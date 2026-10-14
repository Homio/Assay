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
    # ── track_p5：合并调用通道（detect 任务 5 跳 → 2 跳） ─────────────
    # yolo2：一次往返携带 std + (条件) fb 双 pass；可选跨请求同 conf
    #   合批（ultralytics list source）。list 批与逐页 predict 存在
    #   ~1e-4 量级 conf fp 抖动（同 P3a/P3b 已豁免的 CUDA 抖动类，
    #   A/B 见 track_p5/ab_yolo_listbatch.py），可经本开关整体回退。
    yolo_list_batch = bool(cfg.get("yolo_list_batch", True))
    yolo2_max = int(cfg.get("yolo2_drain_max", 8))
    # cslanet：一次往返完成整页全部 crop 的 PP-LCNet 门控 + SLANet 结构
    #   （cls 与 slanet 在服务端并行执行，eager：门控在本批内几乎不
    #   裁剪——P3b 实测 337 cls → 337 slanet；多算的 slanet 结果被
    #   worker 丢弃，可观察输出不变）。
    cslanet_max = int(cfg.get("cslanet_batch_max", 32))
    prep_threads = max(1, int(cfg.get("cslanet_prep_threads", 4)))
    # yolo2 std/fb 两组独立 predict 的并行度（fb 组与 std 组的 CPU
    # 前后处理重叠、GPU 内核仍按驱动排队；1 = 串行回退）。
    yolo2_pred_threads = max(1, int(cfg.get("yolo2_pred_threads", 2)))

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

    # ── track_p5：合并调用通道 ──────────────────────────────────────────
    def _run_yolo2(buf: deque):
        """一次往返携带 std + 条件 fb 双 pass。

        payload: {'arr': (H,W,3) uint8, 'conf': float,
                  'conf_fb': float | None}
        reply:   (tables_std, tables_fb_or_None)
        """
        t0 = time.perf_counter()
        std_by_idx: Dict[int, list] = {}
        fb_by_idx: Dict[int, list] = {}
        try:
            if yolo_list_batch:
                # 跨请求同 conf 合批：一组同 conf 页阵列合成一次
                # ultralytics list-source predict，结果按输入顺序切开。
                groups: Dict[Tuple[float, int], List[Tuple[int, object]]] = {}
                for idx, req in enumerate(buf):
                    p = req["payload"]
                    groups.setdefault((float(p["conf"]), 0), []).append(
                        (idx, p["arr"]))
                    if p.get("conf_fb") is not None:
                        groups.setdefault((float(p["conf_fb"]), 1), []).append(
                            (idx, p["arr"]))
                def _predict_group(args_):
                    (conf_, is_fb_), items_ = args_
                    results_ = yolo.predict(
                        source=[a for _, a in items_], imgsz=1024,
                        conf=conf_, verbose=False)
                    out_ = {}
                    for (idx_, _), result_ in zip(items_, results_):
                        out_[idx_] = _extract_yolo_tables(result_)
                    return is_fb_, out_

                group_items = list(groups.items())
                if yolo2_pred_threads > 1 and len(group_items) > 1:
                    # std / fb 两组并行 predict：CPU 前后处理重叠
                    # （letterbox/NMS 在 CPU 上吃 GIL，串行时 fb 组
                    # 纯等待）；GPU 内核由驱动排队，逐位结果不变。
                    futs = [_yolo2_pool.submit(_predict_group, g)
                            for g in group_items]
                    for fut in futs:
                        is_fb_, out_ = fut.result()
                        (fb_by_idx if is_fb_ else std_by_idx).update(out_)
                else:
                    for is_fb_, out_ in map(_predict_group, group_items):
                        (fb_by_idx if is_fb_ else std_by_idx).update(out_)
            else:
                # 逐请求 predict：与 P3b 三通道版的调用序列逐次一致。
                for idx, req in enumerate(buf):
                    p = req["payload"]
                    r = yolo.predict(source=p["arr"], imgsz=1024,
                                     conf=float(p["conf"]), verbose=False)
                    std_by_idx[idx] = _extract_yolo_tables(r[0])
                    if p.get("conf_fb") is not None:
                        r2 = yolo.predict(source=p["arr"], imgsz=1024,
                                          conf=float(p["conf_fb"]),
                                          verbose=False)
                        fb_by_idx[idx] = _extract_yolo_tables(r2[0])
            for idx, req in enumerate(buf):
                _reply(req, True, (std_by_idx.get(idx, []),
                                   fb_by_idx.get(idx)))
        except Exception as e:  # noqa: BLE001
            _log.error(f"[gpu-server] yolo2 batch failed: {e}")
            for req in buf:
                _reply(req, False, repr(e))
        buf.clear()
        dt_ms = (time.perf_counter() - t0) * 1000
        if dt_ms > slow_ms:
            _log.warning(f"[gpu-server] slow yolo2 batch: {dt_ms:.0f}ms "
                         f"({len(std_by_idx)} pages)")

    # cslanet 执行池：cls 与 slanet 两路并行（互不阻塞、无共享池死锁）。
    import concurrent.futures as _cf
    _exec_pool = _cf.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="gpu-cslanet-exec")
    _cls_prep_pool = _cf.ThreadPoolExecutor(
        max_workers=prep_threads, thread_name_prefix="gpu-clsprep")
    _sl_prep_pool = _cf.ThreadPoolExecutor(
        max_workers=prep_threads, thread_name_prefix="gpu-slprep")
    _yolo2_pool = _cf.ThreadPoolExecutor(
        max_workers=yolo2_pred_threads, thread_name_prefix="gpu-yolo2pred")

    _rt = slanet.engine          # RapidTable
    _ts = _rt.table_structure     # PPTableStructurer

    def _cls_probs(crop_list) -> List[float]:
        """PP-LCNet 门控概率（与 _run_cls 同一数学，预处理池化并行）。"""
        n = len(crop_list)
        arrs = list(_cls_prep_pool.map(_cls_prep, crop_list))
        probs = [0.0] * n
        for s in range(0, n, cls_max):
            chunk_arrs = arrs[s:s + cls_max]
            batch = np.stack(chunk_arrs)
            out = clf.session.run([clf.output_name],
                                  {clf.input_name: batch})[0]
            # 与 predict 相同的稳定 softmax（logits − max）
            mx = out.max(axis=1, keepdims=True)
            exp = np.exp(out - mx)
            p = exp / exp.sum(axis=1, keepdims=True)
            for j in range(len(chunk_arrs)):
                probs[s + j] = float(p[j][1])
        return probs

    def _sl_prep_one(ori_img):
        """TablePreprocess 单图版——与 TablePreprocess.__call__ 逐位一致
        （其内部本就逐图循环；此处只是把循环交给线程池）。
        preprocess_op 返回 (imgs_list, shape_arr)，单图调用时 imgs_list
        恰含一图：拆出图像本体，shape 保留 (1,6) 形状以贴合 decode 的
        shape_list[batch_idx] 索引语义。"""
        img_list, shape_arr = _ts.preprocess_op([ori_img])
        return img_list[0], shape_arr

    def _sl_decode_one(args):
        """TableLabelDecode 单图版——decode 内部按 batch_idx 逐图循环，
        此处按 batch=1 切片并行，逐图输出不变。"""
        bbox_preds, struct_probs, shape_list, ori_img = args
        structs, cell_bboxes = _ts.postprocess_op(
            bbox_preds, struct_probs, shape_list, ori_img)
        return structs, cell_bboxes

    def _slanet_all(crop_list):
        """SLANet 全量推理——镜像 RapidTable.__call__ 的逐 chunk 流程，
        仅把单图预处理/解码放进线程池（同一函数、同一顺序、同一输入）。
        返回 (cell_bboxes 列表, logic_points 列表)。"""
        # load_img（PIL→ndarray）也进池：337 crop 串行 ~0.26s，
        # 且在执行线程上占 GIL，池化后与其他 CPU 阶段重叠。
        ori_imgs = list(_sl_prep_pool.map(_rt.load_img, crop_list))
        all_structs: List = []
        all_cell_bboxes: List = []
        all_logic_points: List = []
        for s in range(0, len(ori_imgs), cslanet_max):
            chunk = ori_imgs[s:s + cslanet_max]
            prep = list(_sl_prep_pool.map(_sl_prep_one, chunk))
            imgs = [im for im, _sh in prep]
            shape_arrs = [sh for _im, sh in prep]   # 每图 (1,6)
            # 与 PPTableStructurer.__call__ 相同的 session 调用
            # （imgs 为 (3,488,488) 数组列表，ORT 侧堆成 (N,C,H,W)）
            bbox_preds, struct_probs = _ts.session(imgs.copy())
            dec = list(_sl_prep_pool.map(
                _sl_decode_one,
                [(bbox_preds[i:i + 1], struct_probs[i:i + 1],
                  shape_arrs[i], chunk[i:i + 1])
                 for i in range(len(chunk))]))
            for structs, cell_bboxes in dec:
                all_structs.extend(structs)
                all_cell_bboxes.extend(cell_bboxes)
            # 与 RapidTable.__call__ 相同：逐 chunk 解码 logic points
            all_logic_points.extend(
                _rt.table_matcher.decode_logic_points(
                    [st for sts in dec for st in sts[0]]))
        return all_cell_bboxes, all_logic_points

    def _run_cslanet(buf: deque):
        """一次往返：整批所有 crop 的门控概率 + SLANet 结构。

        payload: {'crops': [PIL], 'offsets': [(x, y), ...]}
        reply:   每 crop 一个 dict
                 {'prob': P(table), 'cell_bboxes': …, 'logic_points': …,
                  'offset': (x, y), 'img_size': (w, h)}
        cls 与 slanet 在两个执行线程并行；门控判定由 worker 端按同一
        阈值比较完成（eager，见上方说明）。
        """
        t0 = time.perf_counter()
        req_sizes: List[Tuple[dict, int]] = []
        all_crops: List = []
        all_offsets: List = []
        for req in buf:
            p = req["payload"]
            req_sizes.append((req, len(p["crops"])))
            all_crops.extend(p["crops"])
            all_offsets.extend(p["offsets"])
        n = len(all_crops)
        try:
            if n == 0:
                for req, _k in req_sizes:
                    _reply(req, True, [])
                buf.clear()
                return
            fut_cls = _exec_pool.submit(_cls_probs, all_crops)
            fut_sl = _exec_pool.submit(_slanet_all, all_crops)
            probs = fut_cls.result()
            cb_list, lp_list = fut_sl.result()
            # 防御层：推理结果数量必须与请求 crop 数一致，否则整批回错
            # （fail-loud），绝不允许静默补空数组把坏结果喂给下游。
            if len(probs) != n or len(cb_list) != n or len(lp_list) != n:
                _log.error(
                    "[gpu-server] cslanet result length mismatch: "
                    f"requested {n} crops, got probs={len(probs)}, "
                    f"cell_bboxes={len(cb_list)}, "
                    f"logic_points={len(lp_list)}; failing whole batch")
                for req, _k in req_sizes:
                    _reply(req, False,
                           "slanet result length mismatch: "
                           f"requested {n}, probs={len(probs)}, "
                           f"cell_bboxes={len(cb_list)}, "
                           f"logic_points={len(lp_list)}")
                buf.clear()
                return
            i = 0
            for req, k in req_sizes:
                out: List[Dict] = []
                for j in range(k):
                    out.append({
                        "prob": probs[i + j],
                        "cell_bboxes": (
                            cb_list[i + j] if i + j < len(cb_list)
                            else np.array([])),
                        "logic_points": (
                            lp_list[i + j] if i + j < len(lp_list)
                            else np.array([])),
                        "offset": all_offsets[i + j],
                        "img_size": all_crops[i + j].size,
                    })
                _reply(req, True, out)
                i += k
        except Exception as e:  # noqa: BLE001
            _log.error(f"[gpu-server] cslanet batch failed: {e}")
            for req, _k in req_sizes:
                _reply(req, False, repr(e))
        buf.clear()
        dt_ms = (time.perf_counter() - t0) * 1000
        if dt_ms > slow_ms:
            _log.warning(f"[gpu-server] slow cslanet batch: {dt_ms:.0f}ms "
                         f"({n} crops)")

    channels["yolo2"] = (deque(), _run_yolo2)
    channels["cslanet"] = (deque(), _run_cslanet)
    exit_seen = False
    # track_p3b 诊断：各通道实际执行条数与耗时
    exec_stats: Dict[str, list] = {"yolo": [0, 0.0, 0],
                                   "cls": [0, 0.0, 0],
                                   "slanet": [0, 0.0, 0],
                                   "yolo2": [0, 0.0, 0],
                                   "cslanet": [0, 0.0, 0]}

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
        # track_p5 合并通道：yolo2 凑批窗口与 yolo 同理取 0（积压来自
        # 自然并发）；cslanet 沿用 slanet 的 5ms 微窗（默认值均可由
        # pipeline_config.yaml 覆盖，零硬编码）。
        "yolo2": float(cfg.get("yolo2_window_ms", 0.0)),
        "cslanet": float(cfg.get("cslanet_window_ms", 5.0)),
    }
    # 单批次上限：yolo 按页（无真批，只控单批页数）；cls 按图；
    # slanet 按 crop（engine batch_size 真批）。
    # yolo2 按页；cslanet 按 crop（slanet_max 同一真批口径）。
    quotas = {"yolo": yolo_drain, "cls": cls_max, "slanet": slanet_max,
              "yolo2": yolo2_max, "cslanet": cslanet_max}
    stop_evt = threading.Event()

    def _serve(ch):
        buf, runner = channels[ch]
        window = window_ms_cfg[ch] / 1000.0
        quota = quotas[ch]
        # track_p5：合并通道用「真·凑批窗口」——首个请求到达后最多等
        # window，凑满 quota（按通道计数口径：yolo2 页 / cslanet crop）
        # 提前执行。GPU session 每次调用有 ~80-160ms 固定成本（idle GPU
        # 实测 batch12≈8.7ms/crop、batch32≈4.9、batch48≈2.9），批越大
        # 摊薄越明显。legacy 三通道保持 P3b 原语义（仅零星请求等一窗）。
        use_collect = ch in ("yolo2", "cslanet")

        def _buf_size():
            if ch == "cslanet":
                return sum(len(r["payload"]["crops"]) for r in buf)
            if ch in ("cls", "slanet"):
                return sum(len(r["payload"]) for r in buf)
            return len(buf)

        while not stop_evt.is_set():
            if not buf:
                time.sleep(0.002)
                continue
            # 凑批窗口：仅当缓冲里只有零星请求时短暂等待并发聚合
            if use_collect and window > 0 and _buf_size() < quota:
                _deadline = time.monotonic() + window
                _last = _buf_size()
                while time.monotonic() < _deadline:
                    if _last >= quota:
                        break
                    time.sleep(0.004)
                    _now = _buf_size()
                    if _now == _last:
                        continue   # 窗口内持续静默也等满 deadline：让
                    # 突发间隙不断流，凑满 quota 再执行（track_p5 s2 实测
                    # 4ms 静默早退使 cslanet 批碎化，avg 21→23ms/crop）
                    _last = _now
            elif window > 0 and len(buf) == 1:
                time.sleep(window)
                if not buf:
                    continue
            part = deque()
            acc = 0
            while buf:
                r = buf[0]
                if ch in ("cls", "slanet"):
                    sz = len(r["payload"])
                elif ch == "cslanet":
                    # payload 为 dict {'crops':…, 'offsets':…}，按 crop 计
                    sz = len(r["payload"]["crops"])
                else:
                    # yolo / yolo2：按页
                    sz = 1
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
            exec_stats[ch][2] += 1   # 执行次数 → 退出时算真实平均批大小

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
    for _ch in ("yolo", "cls", "slanet", "yolo2", "cslanet"):
        _n, _s, _c = exec_stats[_ch]
        print(f"[gpu-server] exec {_ch}: items={_n} total={_s:.2f}s "
              f"avg={(_s / _n * 1000):.2f}ms/item calls={_c} "
              f"batch={(_n / _c):.1f}" if _n else
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
        self.stats = {"yolo": [0, 0.0], "cls": [0, 0.0], "slanet": [0, 0.0],
                      # track_p5 合并通道
                      "yolo2": [0, 0.0], "cslanet": [0, 0.0]}

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

    # ── track_p5 合并通道 API（detect 任务 5 跳 → 2 跳） ─────────────────

    def yolo2_detect(self, arr, conf: float, conf_fb=None):
        """一次往返携带 std + 条件 fb 双 pass。

        返回 (tables_std, tables_fb_or_None)：tables_fb 仅当 conf_fb
        非 None 时计算（对应 heuristic_table=True 的条件 fallback）。
        """
        return self._call("yolo2", {"arr": arr, "conf": float(conf),
                                    "conf_fb":
                                        (None if conf_fb is None
                                         else float(conf_fb))})

    def cslanet_all(self, crops, offsets) -> List[Dict]:
        """整页全部 crop 一次往返：门控概率 + SLANet 结构并行算。

        crops 为 PIL 图列表，offsets 为对应 (x, y) 页内偏移列表。
        返回逐 crop dict：{'prob', 'cell_bboxes', 'logic_points',
        'offset', 'img_size'}；门控判定（prob ≥ 阈值）由调用方完成。
        """
        assert len(crops) == len(offsets), "crops/offsets 长度不一致"
        return self._call("cslanet", {"crops": list(crops),
                                      "offsets": list(offsets)})
