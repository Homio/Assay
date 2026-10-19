"""
CPU 工作池（track_p3b 阶段 3，P2 设计第 2 层）。

无状态 worker 进程：不加载任何模型、不 import torch，只做四类页粒度
任务（全部复用 assay 包既有函数，算法零改动）：

  calib  页眉校准扫描分片（detection.calibrate_page_headers_scan_page）
  text   任务 A：extract_page_text + page_has_table + top_remnant_candidate
  detect 任务 B(+C)：渲染 + detect_tables_on_page(yolo_fn=GpuClient) +
         extractor.prepare_page_slanet_data（PP-LCNet 门控与 SLANet 走
         GpuClient）
  build  任务 D：extractor.finalize_table_from_slanet（构建 + 守卫 +
         complete_table 全套）

worker 与 GPU 服务的通信见 assay.gpu_server.GpuClient。任务失败（含 GPU
超时）以 error 结果回传，由调度器重投（任务幂等）。

配置：pipeline_config.yaml（scheduler 段），零硬编码。
"""

from __future__ import annotations

import logging
import os
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_log = logging.getLogger(__name__)

# P17 spike：多槽 fitz doc 缓存（LRU）+ 任务执行期 pin。
# 单槽缓存在嵌套窃取下会 close 挂起任务正持有的 doc → "page is None"
# （b10_a2_r1 实测 1176 次 → 重投循环 → 挂死）。修复：任务执行期 pin
# 自己的 doc；淘汰只挑未 pin 槽位；全 pin 满载退化为不缓存直开。
from collections import OrderedDict as _OrderedDict

_DOC_CACHE = _OrderedDict()   # path -> doc（插入序 = LRU 冷端）
_DOC_PINS = {}               # path -> pin 计数（同书嵌套可叠）
# P25 A3：缓存槽位上限由 worker_main 按 pipeline_config.yaml
#（scheduler.worker_doc_cache_cap）覆盖；模块默认仅兜底。
_DOC_CAP = 16                # 每 worker 最多常开的 doc 数


def _enable_pdeathsig():
    """P25 A4：孤儿免疫的内核级兜底 —— PR_SET_PDEATHSIG(SIGKILL)。

    主循环 ppid 轮询只覆盖取任务空转点；p25_freeze200_final 实测：
    调度器被杀后 worker 常卡在 mp 队列的 futex（满管道 put /
    队列锁）里，永远回不到轮询点（20 个 worker 仅 1 个因 GPU 超时
    异常解锁而退出，19 个驻留各 ~1GB）。PDEATHSIG 让内核在父进程
    死亡瞬间直接收割本进程，对任何阻塞点免疫。prctl 后立刻自检
    ppid，防父进程在 prctl 前已死的竞态（此时 ppid 已是 1）。"""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, 9)  # PR_SET_PDEATHSIG=1, SIGKILL=9
        if os.getppid() == 1:
            os._exit(0)
    except Exception:  # noqa: BLE001
        pass


def _flush_doc_cache():
    """P25 A2：pause 主动释放——关闭全部未 pin 的缓存 doc、GC、
    malloc_trim（pymalloc arena 高水位还给 OS），让 watchdog pause
    真正降水位。仅关未 pin 槽位（被窃取嵌套任务持有的 doc 不动）。"""
    import gc
    for _p in list(_DOC_CACHE):
        if not _DOC_PINS.get(_p):
            try:
                _DOC_CACHE.pop(_p).close()
            except Exception:  # noqa: BLE001
                pass
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _open_doc(path: str):
    import fitz
    doc = _DOC_CACHE.get(path)
    if doc is not None:
        _DOC_CACHE.move_to_end(path)
        return doc
    while len(_DOC_CACHE) >= _DOC_CAP:
        victim = None
        for p in _DOC_CACHE:
            if not _DOC_PINS.get(p):
                victim = p
                break
        if victim is None:
            # 全部槽位被在途任务 pin 满载：不缓存直开，用毕随 GC。
            return fitz.open(path)
        try:
            _DOC_CACHE.pop(victim).close()
        except Exception:
            pass
    doc = fitz.open(path)
    _DOC_CACHE[path] = doc
    return doc


def _pin_doc(path: str):
    _DOC_PINS[path] = _DOC_PINS.get(path, 0) + 1


def _unpin_doc(path: str):
    n = _DOC_PINS.get(path, 0) - 1
    if n > 0:
        _DOC_PINS[path] = n
    else:
        _DOC_PINS.pop(path, None)


def run_calib(task):
    from assay.detection import calibrate_page_headers_scan_page
    doc = _open_doc(task["path"])
    partials = []
    for i in range(task["page_lo"], min(task["page_hi"], len(doc))):
        partials.append(calibrate_page_headers_scan_page(doc[i]))
    return partials


def run_text(task):
    from assay.extractor import extract_page_text, page_has_table
    from assay.detection import top_remnant_candidate
    doc = _open_doc(task["path"])
    page = doc[task["page_idx"]]
    texts = extract_page_text(page, task["page_idx"],
                              header_sigs=task["header_sigs"])
    cand = page_has_table(page) or top_remnant_candidate(page)
    return {"texts": texts, "w": page.rect.width, "h": page.rect.height,
            "cand": bool(cand)}


def run_detect(task, client, fused_calls=False):
    import fitz
    import numpy as np
    from assay.detection import (
        detect_tables_on_page, YOLO_CONFIDENCE, YOLO_CONFIDENCE_FALLBACK)
    from assay.extractor import (PAGE_ZOOM, GATEKEEPER_THRESHOLD,
                                 prepare_page_slanet_data)
    doc = _open_doc(task["path"])
    page = doc[task["page_idx"]]
    pix = page.get_pixmap(matrix=fitz.Matrix(PAGE_ZOOM, PAGE_ZOOM))

    def _pix_arr(pix):
        # 采样通道裁剪与 detect_tables_image 逐行一致。
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        return arr

    if fused_calls:
        # track_p5：detect 任务 GPU 往返 5 跳 → 2 跳。
        # 第 1 跳：std + 条件 fb 双 pass 一次请求（fb 仅在 heuristic
        # 判真时随行携带，服务端可选同 conf 跨请求合批）。
        def _yolo_fused_fn(pix, need_fallback):
            return client.yolo2_detect(
                _pix_arr(pix), YOLO_CONFIDENCE,
                conf_fb=(YOLO_CONFIDENCE_FALLBACK if need_fallback
                         else None))

        yolo_tables = detect_tables_on_page(
            page, pix, PAGE_ZOOM, yolo_fused_fn=_yolo_fused_fn)
    else:
        # yolo_fn 与 detection.detect_tables_image 同契约（pix, conf=…），
        # 采样通道裁剪与 detect_tables_image 逐行一致；内部走 GPU 服务。
        # track_p5 P13a：单前向双 conf——一次 yolo_detect(conf=0.0) 拿
        # 全 top-300 候选，worker 端按调用方传入的 conf 阈值分流；fused
        # 路径（上方）不动。等价性见 analysis/perf/track_p5/
        # YOLO_DUALCONF_SPIKE.md §一（v10postprocess 顺序验证）+ §二
        # （平安银行 296 页逐位 diff 0/296）。
        _yolo_dual_cache = [None, None]  # [pix_id, all_tables]
        def _yolo_fn(pix, conf=YOLO_CONFIDENCE):
            if _yolo_dual_cache[0] is not pix:
                _yolo_dual_cache[1] = client.yolo_detect(_pix_arr(pix), 0.0)
                _yolo_dual_cache[0] = pix
            return [t for t in _yolo_dual_cache[1] if t[4] > conf]

        yolo_tables = detect_tables_on_page(
            page, pix, PAGE_ZOOM, yolo_fn=_yolo_fn)

    # image-pixel → PDF-point 换算（与 legacy extract() Phase 2 同口径）
    scale_x = page.rect.width / pix.width
    scale_y = page.rect.height / pix.height
    if not yolo_tables:
        return {"pairs": [], "n_crops": 0,
                "scale_x": scale_x, "scale_y": scale_y}

    def _gate(crop_img):
        # 与 TableModel.is_table(threshold=GATEKEEPER_THRESHOLD) 等价：
        # 服务端返回逐图 P(table)，worker 端做同一比较。
        return client.classify([crop_img])[0] >= GATEKEEPER_THRESHOLD

    if fused_calls:
        # 第 2 跳：整页全部 crop 的门控 + SLANet 结构一次请求。
        # 服务端 cls 与 slanet 并行执行；门控判定在本端用同一阈值
        # 比较（fused 路径下 gate_fn/structure_fn 不被调用，仅保
        # legacy 可回退）。
        pairs = prepare_page_slanet_data(
            page, pix, yolo_tables, gate_fn=_gate,
            structure_fn=client.process_crops,
            batch_size=max(1, len(yolo_tables)),
            fused_cslanet_fn=client.cslanet_all,
            gate_threshold=GATEKEEPER_THRESHOLD,
        )
    else:
        # 单请求整页发派：服务端跨页合并凑批（batch_size 由服务端配置）。
        pairs = prepare_page_slanet_data(
            page, pix, yolo_tables, gate_fn=_gate,
            structure_fn=client.process_crops,
            batch_size=max(1, len(yolo_tables)),
        )
    return {"pairs": pairs, "n_crops": len(pairs),
            "scale_x": scale_x, "scale_y": scale_y}


def run_build(task):
    from assay.extractor import finalize_table_from_slanet, _running_header_rects
    doc = _open_doc(task["path"])
    page = doc[task["page_idx"]]
    hdr_rects = _running_header_rects(page, task["header_sigs"])
    tables = []
    for sres, bbox in task["items"]:
        t = finalize_table_from_slanet(
            sres, page, task["page_idx"],
            task["scale_x"], task["scale_y"], bbox,
            page_hdr_rects=hdr_rects)
        if t is not None:
            tables.append(t)
    return {"tables": tables}


def worker_main(worker_id, task_q, result_q, infer_q, reply_q,
                active_slots, cfg, infer_q_csl=None, pause_evt=None):
    """
    Worker 进程入口。

    active_slots：mp.Value('i') —— 看门狗收缩/恢复的活跃槽位数。worker
    序号 ≥ 活跃槽位时让出 CPU（睡眠轮询），被收缩的槽位不取新任务；
    恢复后自动回到循环。哨兵 None → 退出。

    track_p5/P17 spike：async_pipeline=true 时 GpuClient armed 任务窃取
    ——detect 任务等 GPU 回包超 async_wait_ms 即从共享 task_q 偷任务执行
    （页上下文挂起/恢复由调用栈天然承载；detect 在途受 depth 封顶）。
    任务执行统一走 _run_one（正常领取与被窃取同一条路径）。

    P25 A2/A4：pause_evt 置位且空转时清 doc 缓存主动降水位；主循环
    检测 os.getppid()==1（父进程被 kill -9 收编为孤儿）即退出，
    杜绝孤儿舰队驻留内存。
    """
    from assay.gpu_server import GpuClient

    _enable_pdeathsig()
    logging.basicConfig(level=logging.WARNING)
    # P25 A3：死配置接线——MuPDF store 限额（worker_store_mb MB，
    # 0 = 不动）；doc LRU 缓存槽位上限。
    global _DOC_CAP
    _DOC_CAP = max(1, int(cfg.get("doc_cache_cap", _DOC_CAP)))
    store_mb = int(cfg.get("worker_store_mb", 0))
    if store_mb > 0:
        import fitz
        fitz.TOOLS.store_maxsize = store_mb << 20
    release_on_pause = bool(cfg.get("release_on_pause", False))
    timeout_s = float(cfg.get("gpu_timeout_s", 60.0))
    fused_calls = bool(cfg.get("fused_calls", False))
    client = GpuClient(infer_q, reply_q, worker_id, timeout_s=timeout_s,
                       infer_q_csl=infer_q_csl)
    async_on = bool(cfg.get("async_pipeline", False))
    if async_on:
        client.arm_steal(task_q, active_slots,
                         depth=int(cfg.get("async_depth", 2)),
                         wait_ms=float(cfg.get("async_wait_ms", 50.0)),
                         guard_s=float(cfg.get("async_guard_s", 10.0)))

    def _run_one(task):
        kind = task["kind"]
        # P17 spike：执行期 pin 本书 doc —— 等待期嵌套窃取开别的书
        # 也不会 close 本任务正持有的 doc（page/pix 生命周期保障）。
        _pin_doc(task["path"])
        ctx = client.task_begin(kind)
        t0 = time.time()
        try:
            if kind == "calib":
                out = run_calib(task)
            elif kind == "text":
                out = run_text(task)
            elif kind == "detect":
                out = run_detect(task, client, fused_calls=fused_calls)
            elif kind == "build":
                out = run_build(task)
            else:
                raise ValueError(f"unknown task kind {kind!r}")
            # P17 spike：dt 扣除被窃取时间（Σdt 不重复计数）；GPU 记账
            # 已按 ctx 独立累计（嵌套任务不重复计入本任务）。任务异常
            # 照常回传 ok=False（调度器重投/失败），绝不静默补空。
            result_q.put({"task": task, "ok": True, "data": out,
                          "dt": time.time() - t0 - ctx["stolen"],
                          "gpu": {ch: list(ctx["gpu"][ch])
                                  for ch in ctx["gpu"]}})
        except Exception as e:  # noqa: BLE001
            _log.error(f"[W{worker_id:02d}] {kind} task failed: {e}\n"
                       f"{traceback.format_exc(limit=8)}")
            result_q.put({"task": task, "ok": False, "error": repr(e),
                          "dt": time.time() - t0 - ctx["stolen"],
                          "gpu": {ch: list(ctx["gpu"][ch])
                                  for ch in ctx["gpu"]}})
        finally:
            client.task_end(ctx)
            _unpin_doc(task["path"])

    if async_on:
        client._runner = _run_one

    flushed_on_pause = False
    while True:
        # P25 A4：孤儿免疫——父进程被 kill -9 后本进程被 init 收编
        #（ppid==1），立即退出，杜绝孤儿舰队（~10GB/代 驻留）。
        if os.getppid() == 1:
            _log.warning("[W%02d] parent dead (ppid=1) — exiting",
                         worker_id)
            break
        while active_slots.value <= worker_id:
            if os.getppid() == 1:
                _log.warning("[W%02d] parent dead (ppid=1) — exiting",
                             worker_id)
                return
            time.sleep(0.5)
        try:
            task = task_q.get(timeout=1.0)
        except Exception:
            # P25 A2：空转遇 pause → 主动释放（每次 pause  episode 一次，
            # pause 解除后复位）。
            if pause_evt is not None and pause_evt.is_set():
                if release_on_pause and not flushed_on_pause:
                    _flush_doc_cache()
                    flushed_on_pause = True
            else:
                flushed_on_pause = False
            continue
        if task is None:
            break
        _run_one(task)

    if async_on:
        # P17 spike 画像：本 worker 窃取统计（退出时一次性打印）
        print(f"[worker{worker_id:02d}] steal n={client.steal_n} "
              f"s={client.steal_s:.1f} max_detect_depth={client.max_depth} "
              f"stash_left={len(client._stash)} "
              f"reply_cache_left={len(client._reply_cache)} "
              f"sentinel_seen={client._saw_sentinel}", flush=True)
