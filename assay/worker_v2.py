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

# 单槽 fitz doc 缓存：同书任务连续到达时省一次 open（~10-30ms）。
_DOC_CACHE = {"path": None, "doc": None}


def _open_doc(path: str):
    import fitz
    if _DOC_CACHE["path"] == path:
        return _DOC_CACHE["doc"]
    if _DOC_CACHE["doc"] is not None:
        try:
            _DOC_CACHE["doc"].close()
        except Exception:
            pass
    doc = fitz.open(path)
    _DOC_CACHE["path"] = path
    _DOC_CACHE["doc"] = doc
    return doc


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
                 active_slots, cfg, infer_q_csl=None):
    """
    Worker 进程入口。

    active_slots：mp.Value('i') —— 看门狗收缩/恢复的活跃槽位数。worker
    序号 ≥ 活跃槽位时让出 CPU（睡眠轮询），被收缩的槽位不取新任务；
    恢复后自动回到循环。哨兵 None → 退出。
    """
    import multiprocessing as mp

    from assay.gpu_server import GpuClient

    logging.basicConfig(level=logging.WARNING)
    timeout_s = float(cfg.get("gpu_timeout_s", 60.0))
    # track_p5：合并调用开关（scheduler worker_cfg 透传，源自
    # pipeline_config.yaml gpu_server.fused_calls）。false = P3b 三通道
    # 路径，逐位不变。
    fused_calls = bool(cfg.get("fused_calls", False))
    # track_p5/p13b 拆分服务：cls/slanet/cslanet 请求走独立服务队列
    #（None = 单服务 legacy，全部走 infer_q）。
    client = GpuClient(infer_q, reply_q, worker_id, timeout_s=timeout_s,
                       infer_q_csl=infer_q_csl)

    while True:
        # 槽位收缩：序号在槽外的 worker 睡眠让核（恒留 1 核 + 5GB 的
        # 看门狗手段之一）
        while active_slots.value <= worker_id:
            time.sleep(0.5)
        try:
            task = task_q.get(timeout=1.0)
        except Exception:
            continue
        if task is None:
            break
        kind = task["kind"]
        g0 = {ch: tuple(v) for ch, v in client.stats.items()}
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
            gpu = {ch: [client.stats[ch][0] - g0[ch][0],
                         client.stats[ch][1] - g0[ch][1]] for ch in g0}
            result_q.put({"task": task, "ok": True, "data": out,
                          "dt": time.time() - t0, "gpu": gpu})
        except Exception as e:  # noqa: BLE001
            _log.error(f"[W{worker_id:02d}] {kind} task failed: {e}\n"
                       f"{traceback.format_exc(limit=8)}")
            gpu = {ch: [client.stats[ch][0] - g0[ch][0],
                         client.stats[ch][1] - g0[ch][1]] for ch in g0}
            result_q.put({"task": task, "ok": False, "error": repr(e),
                          "dt": time.time() - t0, "gpu": gpu})
