"""
GPU 常驻单推理服务（track_p3b 阶段 3，P2 设计第 3 层）。

独占 GPU0 的单实例常驻进程，持有全部三个模型（PP-LCNet 门控 /
YOLO 页检测 / SLANet 结构），对跨页、跨本的推理请求做动态批。
CPU worker 通过 GpuClient 把请求投进 infer_q，阻塞等各自的专属
回包队列；服务按通道（yolo / cls / slanet）凑批，批满或收集窗口
超时即执行。

算法零改动：
  - YOLO：与 detection.detect_tables_image 完全相同的调用
    （ultralytics predict per page, imgsz=1024）与完全相同的后处理
    （detection._extract_yolo_tables，纯搬移）。
  - PP-LCNet：与 models.TableClassifier.predict 逐位一致的预处理
    （RGB→224 BILINEAR→/255→mean/std→CHW），仅把单图 session.run
    换成堆批 run（同一模型、同一 softmax，逐图结果与单图一致，
    属 P3a 已豁免的 CUDA fp32 抖动同类）。
  - SLANet：rapid_table RapidTable.__call__ 的原生 batch_size 参数
    （其内部 TableLabelDecode 本就按 batch_idx 逐图后处理）。

配置全部走 pipeline_config.yaml（gpu_server 段），代码零硬编码。
"""

from __future__ import annotations

import logging
import os
import queue as _queue
import time
from collections import deque
from itertools import count
from typing import Dict, List, Tuple

_log = logging.getLogger(__name__)

# GPU0 only — GPU1 上有常驻租户，绝不触碰。在 import 任何 CUDA 依赖前
# 生效（spawn 子进程入口即调用）。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


class GpuServerError(RuntimeError):
    """GPU 服务端在执行批次时抛出的异常（透传给 worker 的任务重试层）。"""


def gpu_server_main(infer_q, heartbeat, reply_queues, cfg) -> None:
    """
    GPU 服务进程入口。

    Args:
        infer_q:    mp.Queue — worker → 服务的请求队列。请求 dict：
                    {'ch': 'yolo'|'cls'|'slanet', 'rid': int,
                     'wid': int, 'payload': ...}
        reply_queues: per-worker reply queues, passed by
                    inheritance at process creation; routed by request wid.
                    关停哨兵 None → 尽力回包后在制批次后退出。
        heartbeat:  mp.Value('d') — 每个外循环刷新 time.time()，供调度器
                    看门狗检测服务存活（3s 无心跳 → 判定降级重启）。
        cfg:        pipeline_config.yaml['gpu_server'] 段（dict）。
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import numpy as np

    # tqdm 进度条在服务循环里每批打一行 stderr —— UI 噪音，不影响算法。
    try:
        import rapid_table.main as _rtm
        _rtm.tqdm = lambda x, **_kw: x
    except Exception:
        pass

    from assay.detection import _get_yolo, _extract_yolo_tables
    from assay.models import (
        TableClassifier,
        TableStructureModel,
        PPLCNET_INPUT_SIZE,
    )

    heartbeat.value = time.time()
    yolo = _get_yolo()
    heartbeat.value = time.time()
    clf = TableClassifier(use_gpu=True)
    heartbeat.value = time.time()
    slanet = TableStructureModel(use_gpu=True)
    heartbeat.value = time.time()
    # track_p3b 诊断：显式自报各模型实际 EP / 设备（spawn 子进程日志级别
    # 默认 WARNING，_log.info 不可见 —— 用 stderr 直出）
    try:
        print(f"[gpu-server] clf providers="
              f"{clf.session.get_providers()}", flush=True)
    except Exception as e:
        print(f"[gpu-server] clf probe fail: {e}", flush=True)
    try:
        _sl_sess = slanet.engine.table_structure.session.session
        print(f"[gpu-server] slanet providers="
              f"{_sl_sess.get_providers()}", flush=True)
    except Exception as e:
        print(f"[gpu-server] slanet probe fail: {e}", flush=True)
    try:
        import torch
        print(f"[gpu-server] torch cuda={torch.cuda.is_available()} "
              f"dev={torch.cuda.current_device() if torch.cuda.is_available() else -1}",
              flush=True)
    except Exception as e:
        print(f"[gpu-server] torch probe fail: {e}", flush=True)
    _log.info("[gpu-server] models loaded (YOLO + PP-LCNet + SLANet, cuda:0)")

    # track_p3b：凑批窗口废弃。改为每轮每通道执行配额（见主循环）：
    # 批质量来自「上一轮执行期间的自然积压」，无需人为等待窗口。
    _ = cfg.get("collect_window_ms")
    slow_ms = float(cfg.get("slow_batch_ms", 2000))
    slanet_max = int(cfg.get("slanet_batch_max", 32))
    cls_max = int(cfg.get("cls_batch_max", 64))
    yolo_drain = int(cfg.get("yolo_drain_max", 8))

    def _cls_prep(img) -> "np.ndarray":
        # 与 TableClassifier.predict 逐位一致（RGB 224 BILINEAR / 255
        # mean std / CHW），返回 (3,224,224) float32。
        from PIL import Image as _PILImage
        img = img.convert("RGB").resize(
            (PPLCNET_INPUT_SIZE, PPLCNET_INPUT_SIZE), _PILImage.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)

    def _reply(req, ok, payload):
        try:
            reply_queues[int(req.get("wid", 0))].put(
                (req["rid"], ok, payload))
        except Exception:
            pass

    def _run_yolo(buf: deque):
        t0 = time.perf_counter()
        for req in buf:
            try:
                arr, conf = req["payload"]
                results = yolo.predict(source=arr, imgsz=1024,
                                       conf=conf, verbose=False)
                tables = []
                for result in results:
                    tables.extend(_extract_yolo_tables(result))
                _reply(req, True, tables)
            except Exception as e:  # noqa: BLE001
                _log.error(f"[gpu-server] yolo batch item failed: {e}")
                _reply(req, False, repr(e))
        buf.clear()
        dt_ms = (time.perf_counter() - t0) * 1000
        if dt_ms > slow_ms:
            _log.warning(f"[gpu-server] slow yolo drain: {dt_ms:.0f}ms")

    def _run_cls(buf: deque):
        t0 = time.perf_counter()
        # 展开请求 → (req, image)；同批堆 stack 分块 session.run。
        flat: List[Tuple[dict, object]] = []
        for req in buf:
            for img in req["payload"]:
                flat.append((req, img))
        try:
            n = len(flat)
            probs = [0.0] * n
            for s in range(0, n, cls_max):
                chunk = flat[s:s + cls_max]
                batch = np.stack([_cls_prep(_img) for _req, _img in chunk])
                out = clf.session.run([clf.output_name],
                                      {clf.input_name: batch})[0]
                # 与 predict 相同的稳定 softmax（logits − max）
                mx = out.max(axis=1, keepdims=True)
                exp = np.exp(out - mx)
                p = exp / exp.sum(axis=1, keepdims=True)
                for j, (_req, _img) in enumerate(chunk):
                    probs[s + j] = float(p[j][1])
            # 按请求切片回复
            i = 0
            for req in buf:
                k = len(req["payload"])
                _reply(req, True, probs[i:i + k])
                i += k
        except Exception as e:  # noqa: BLE001
            _log.error(f"[gpu-server] cls batch failed: {e}")
            for req in buf:
                _reply(req, False, repr(e))
        buf.clear()
        dt_ms = (time.perf_counter() - t0) * 1000
        if dt_ms > slow_ms:
            _log.warning(f"[gpu-server] slow cls batch: {dt_ms:.0f}ms "
                         f"({n} imgs)")

    def _run_slanet(buf: deque):
        t0 = time.perf_counter()
        # 请求 payload = crop_specs（TableModel.process_crops 契约）；
        # 合并成一次 engine(batch_size=…) 真批调用，再按请求切回。
        req_sizes: List[Tuple[dict, int]] = []
        all_specs: List[Dict] = []
        for req in buf:
            specs = req["payload"]
            req_sizes.append((req, len(specs)))
            all_specs.extend(specs)
        n = len(all_specs)
        try:
            if n == 0:
                for req, _k in req_sizes:
                    _reply(req, True, [])
                buf.clear()
                return
            images = [sp["image"] for sp in all_specs]
            result = slanet.engine(images, batch_size=slanet_max)
            cb_list = result.cell_bboxes
            lp_list = result.logic_points
            i = 0
            for req, k in req_sizes:
                out: List[Dict] = []
                for j in range(k):
                    spec = all_specs[i + j]
                    out.append({
                        "cell_bboxes": (
                            cb_list[i + j] if i + j < len(cb_list)
                            else np.array([])),
                        "logic_points": (
                            lp_list[i + j] if i + j < len(lp_list)
                            else np.array([])),
                        "offset": spec["offset"],
                        "img_size": images[i + j].size,
                    })
                _reply(req, True, out)
                i += k
        except Exception as e:  # noqa: BLE001
            _log.error(f"[gpu-server] slanet batch failed: {e}")
            for req, _k in req_sizes:
                _reply(req, False, repr(e))
        buf.clear()
        dt_ms = (time.perf_counter() - t0) * 1000
        if dt_ms > slow_ms:
            _log.warning(f"[gpu-server] slow slanet batch: {dt_ms:.0f}ms "
                         f"({n} crops)")

    channels = {
        "yolo": (deque(), _run_yolo),
        "cls": (deque(), _run_cls),
        "slanet": (deque(), _run_slanet),
    }
    exit_seen = False
    # track_p3b 诊断：各通道实际执行条数与耗时
    exec_stats: Dict[str, list] = {"yolo": [0, 0.0],
                                   "cls": [0, 0.0],
                                   "slanet": [0, 0.0]}

    # ── track_p3b 终版架构：每通道独立执行线程 ──────────────────────
    # 历史问题（单线程轮转版）：slanet 一个大批次 ~0.7-7.6s 执行期间，
    # cls / yolo 回包全堵，worker 单次 GPU 调用平均往返 ~320ms、detect
    # 单任务最长 4.4s（GPU 真实计算仅 ~14s/本，服务端 duty 70%）。
    # 方案：yolo（torch）/ cls、slanet（ORT）推理均释放 GIL，各通道一个
    # 常驻执行线程；GPU 内核在驱动层排队（不同 stream 甚至可重叠），
    # 通道间不再互相阻塞，单通道延迟 = 自身积压/该通道吞吐。
    import threading

    window_ms_cfg = {
        "yolo": float(cfg.get("yolo_window_ms", 0.0)),
        "cls": float(cfg.get("cls_window_ms", 3.0)),
        "slanet": float(cfg.get("slanet_window_ms", 5.0)),
    }
    # 单批次上限：yolo 按页（无真批，只控单批页数）；cls 按图；
    # slanet 按 crop（engine batch_size 真批）。
    quotas = {"yolo": yolo_drain, "cls": cls_max, "slanet": slanet_max}
    stop_evt = threading.Event()

    def _serve(ch):
        buf, runner = channels[ch]
        window = window_ms_cfg[ch] / 1000.0
        quota = quotas[ch]
        while not stop_evt.is_set():
            if not buf:
                time.sleep(0.002)
                continue
            # 凑批窗口：仅当缓冲里只有零星请求时短暂等待并发聚合
            if window > 0 and len(buf) == 1:
                time.sleep(window)
                if not buf:
                    continue
            part = deque()
            acc = 0
            while buf:
                r = buf[0]
                sz = (len(r["payload"]) if ch in ("cls", "slanet") else 1)
                if acc and acc + sz > quota:
                    break
                part.append(buf.popleft())
                acc += sz
            _t0 = time.perf_counter()
            try:
                runner(part)
            except Exception as e:  # noqa: BLE001
                _log.error(f"[gpu-server] {ch} runner crashed: {e}")
                for r in part:
                    _reply(r, False, repr(e))
            exec_stats[ch][0] += acc
            exec_stats[ch][1] += time.perf_counter() - _t0

    _threads = [threading.Thread(target=_serve, args=(ch,), daemon=True,
                                 name=f"gpu-{ch}")
                for ch in channels]
    for _t in _threads:
        _t.start()

    # 派发线程（主循环）：只做路由 + 心跳
    while not exit_seen:
        heartbeat.value = time.time()
        while True:
            try:
                req = infer_q.get(timeout=0.02)
            except _queue.Empty:
                break
            if req is None:
                exit_seen = True
                break
            ch = req.get("ch")
            entry = channels.get(ch)
            if entry is None:
                _reply(req, False, f"unknown channel {ch!r}")
                continue
            entry[0].append(req)   # deque append 与通道线程 popleft 线程安全

    # 哨兵后：通道线程跑完手头批次退出，剩余缓冲同步清尾
    stop_evt.set()
    for _t in _threads:
        _t.join(timeout=60)
    for ch, (buf, runner) in channels.items():
        if buf:
            runner(buf)
    heartbeat.value = time.time()
    for _ch in ("yolo", "cls", "slanet"):
        _n, _s = exec_stats[_ch]
        print(f"[gpu-server] exec {_ch}: items={_n} total={_s:.2f}s "
              f"avg={(_s / _n * 1000):.2f}ms/item" if _n else
              f"[gpu-server] exec {_ch}: items=0", flush=True)
    _log.info("[gpu-server] drained and exiting")


class GpuClient:
    """
    Worker 侧的 GPU 服务客户端（track_p3b 阶段 3）。

    每个 worker 进程持有一个实例：自带专属回包队列，请求带自增 rid，
    回包按 rid 匹配（陈旧回包——如超时重试后旧批迟到——直接丢弃）。
    服务重启（心跳降级换代）时旧请求或被尽力回包、或随队列消失，
    对应 worker 超时抛 GpuServerError → 任务由调度器重投（任务幂等）。
    """

    def __init__(self, infer_q, reply_q, worker_id,
                 timeout_s: float = 60.0):
        self.infer_q = infer_q
        # reply_q is passed by inheritance at worker process creation
        # (an mp.Queue cannot be pickled as a queue payload); requests
        # carry only the wid and the server routes replies by wid.
        self.reply_q = reply_q
        self.wid = int(worker_id)
        self.timeout_s = float(timeout_s)
        self._rids = count(1)
        # track_p3b 画像：每通道调用次数与等待耗时（含凑批窗口+执行）
        self.stats = {"yolo": [0, 0.0], "cls": [0, 0.0], "slanet": [0, 0.0]}

    def _call(self, ch: str, payload):
        _t0 = time.monotonic()
        try:
            return self._call_impl(ch, payload)
        finally:
            self.stats[ch][0] += 1
            self.stats[ch][1] += time.monotonic() - _t0

    def _call_impl(self, ch: str, payload):
        rid = next(self._rids)
        self.infer_q.put({"ch": ch, "rid": rid, "wid": self.wid,
                          "payload": payload})
        deadline = time.monotonic() + self.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GpuServerError(
                    f"gpu server timeout after {self.timeout_s:.0f}s "
                    f"(ch={ch}, rid={rid})")
            try:
                r_rid, ok, data = self.reply_q.get(timeout=remaining)
            except Exception:
                continue
            if r_rid != rid:
                continue  # 迟到的旧请求回包，丢弃
            if not ok:
                raise GpuServerError(f"gpu server error (ch={ch}): {data}")
            return data

    # ── 三通道 API（与既有单机函数同构） ────────────────────────────────

    def yolo_detect(self, arr, conf: float):
        """detection.detect_tables_image 的服务版：arr=(H,W,3) uint8。"""
        return self._call("yolo", (arr, float(conf)))

    def classify(self, images) -> List[float]:
        """TableClassifier.predict 的批版：返回逐图 P(table)。"""
        return self._call("cls", list(images))

    def process_crops(self, crop_specs) -> List[Dict]:
        """TableModel.process_crops 契约的服务版。"""
        return self._call("slanet", list(crop_specs))
