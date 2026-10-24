"""
Pipeline orchestrator with two execution paths.

  - v2 (default for directory mode with >=2 PDFs): page-granular
    scheduling (assay.scheduler_v2) with a resident GPU inference
    service (assay.gpu_server) and CPU-only workers (assay.worker_v2);
    all scheduling/threshold/watchdog knobs live in pipeline_config.yaml.
  - legacy (this module): per-file ProcessPoolExecutor — each worker
    loads its own TableModel (shared-nothing) and processes whole PDFs;
    CUDA time-slicing handles concurrent GPU access.  Still used for
    single-file runs and as the ASSAY_PIPELINE=legacy fallback.
"""

from __future__ import annotations

import gc
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import torch

_log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# track_p3a QW#1: per-worker TableModel singleton
# ──────────────────────────────────────────────────────────────────────────────
#
# In a ProcessPoolExecutor(spawn), each worker process is long-lived:
# it handles many files in series.  Re-loading PP-LCNet + SLANet (≈2.54s,
# plus 600+MB RSS jitter from CUDA context churn) on every _process_one
# call is pure waste.  We cache the model at module scope so the first
# file in a worker pays the load cost and every subsequent file skips it.
#
# ``_model_cache`` is process-local (spawn-forked), so each worker keeps
# its own copy; the cache is correct under concurrent dispatch because
# only one thread in a worker ever touches it (workers are single-
# threaded by construction).
_model_cache: dict = {}


def _get_worker_model(use_gpu: bool):
    """Return the per-worker TableModel singleton."""
    cached = _model_cache.get("model")
    if cached is None or cached.use_gpu != use_gpu:
        from assay.models import TableModel as _TM
        cached = _TM(use_gpu=use_gpu)
        _model_cache["model"] = cached
    return cached


# ──────────────────────────────────────────────────────────────────────────────
# Per-file worker function (runs inside each ProcessPoolExecutor worker)
# ──────────────────────────────────────────────────────────────────────────────


def _process_one(args: Tuple[str, str, int, bool, bool, bool]) -> Dict:
    """
    Process a single PDF inside a worker process.

    This is the entry point for ProcessPoolExecutor. Each worker:
      1. Loads its own TableModel (ONNX on GPU or CPU)
      2. Extracts tables using AssayExtractor
      3. Writes markdown (+ JSON unless no_json=True)
    """
    pdf_path, output_dir, worker_id, no_json, use_gpu, debug_pdf = args

    # Minimize CPU thread contention between workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    torch.set_num_threads(1)

    # track_p3a QW#1: cached worker-level TableModel singleton
    # (avoid reloading PP-LCNet + SLANet on every file).
    model = _get_worker_model(use_gpu=use_gpu)
    from assay.extractor import AssayExtractor
    from assay.output import (write_markdown, write_json,
                               write_debug_pdf, safe_stem)

    file_stem = Path(pdf_path).stem
    start_time = time.time()

    try:
        extractor = AssayExtractor(model)
        doc = extractor.extract(pdf_path)

        # Write output — P16: safe_stem guards >255-byte component names
        safe = safe_stem(file_stem)
        out_dir = Path(output_dir) / safe
        out_dir.mkdir(parents=True, exist_ok=True)
        write_markdown(doc, str(out_dir / f"{safe}.md"))
        if not no_json:
            write_json(doc, str(out_dir / f"{safe}.json"), indent=2)
        if debug_pdf:
            debug_path = str(out_dir / f"{safe}_layout.pdf")
            write_debug_pdf(doc, pdf_path, debug_path)

        elapsed = time.time() - start_time
        return {
            'file_name': file_stem,
            'status': 'SUCCESS',
            'tables': len(doc.tables),
            'time_sec': elapsed,
            'worker': f'W{worker_id:02d}',
        }

    except Exception as e:
        elapsed = time.time() - start_time
        _log.error(f"Worker {worker_id} failed on {file_stem}: {e}", exc_info=True)
        return {
            'file_name': file_stem,
            'status': 'ERROR',
            'error': str(e),
            'time_sec': elapsed,
            'worker': f'W{worker_id:02d}',
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────


def run(
    pdf_dir: str,
    output_dir: str,
    num_workers: int = 12,
    use_gpu: bool = True,
    no_json: bool = False,
    debug_pdf: bool = False,
) -> float:
    """
    Run the table extraction pipeline on all PDFs in a directory.

    Args:
        pdf_dir:     Directory containing PDF files.
        output_dir:  Directory for markdown + JSON output.
        num_workers: Number of parallel CPU workers (default 12).
        use_gpu:     Use GPU for ONNX inference (default True).
        no_json:     Skip JSON output (default False).
        debug_pdf:   Generate layout-annotated PDF for verification (default False).

    Returns:
        Total wall time in seconds.
    """
    # track_p3b stage 4: directory mode (N>=2) with >=2 PDFs switches to
    # the v2 page-granular pipeline (resident GPU inference service +
    # page-level task scheduling).  Set ASSAY_PIPELINE=legacy to fall back
    # to the per-file ProcessPoolExecutor path (single-file mode always
    # uses legacy).  Same CLI, same output directory layout.
    arch = os.environ.get("ASSAY_PIPELINE", "v2")
    if arch == "v2":
        pdfs_pre = sorted(Path(pdf_dir).glob("*.pdf"))
        if len(pdfs_pre) >= 2:
            from assay.scheduler_v2 import run_v2
            return run_v2(pdf_dir, output_dir, num_workers, use_gpu,
                          no_json, debug_pdf)

    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    total = len(pdfs)
    if total == 0:
        print("Error: no PDF files found")
        return 0.0

    print(f"\n{'='*60}")
    print(f"  Assay — Table Extraction Pipeline")
    print(f"  Workers: {num_workers} | GPU: {'ON' if use_gpu else 'OFF'}")
    print(f"  Files: {total} | Output: {output_dir}")
    print(f"{'='*60}")

    tasks = [(str(p), output_dir, i, no_json, use_gpu, debug_pdf) for i, p in enumerate(pdfs)]

    success, fail = 0, 0
    total_tables = 0
    t_start = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=torch.multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(_process_one, task): task for task in tasks}

        for future in as_completed(futures):
            try:
                res = future.result()
                if res['status'] == 'SUCCESS':
                    success += 1
                    total_tables += res['tables']
                    print(f"  [{success+fail:3d}/{total}] {res['worker']} | "
                          f"{res['file_name'][:45]:45s} | "
                          f"{res['time_sec']:5.1f}s | "
                          f"{res['tables']:4d} tables",
                          flush=True)
                else:
                    fail += 1
                    print(f"  [{success+fail:3d}/{total}] {res['worker']} | "
                          f"{res['file_name'][:45]:45s} | FAIL: "
                          f"{res.get('error', 'unknown')[:50]}",
                          flush=True)
            except Exception as e:
                fail += 1
                print(f"  [{success+fail:3d}/{total}] ERROR | {str(e)[:60]}",
                      flush=True)

    elapsed = time.time() - t_start
    print(f"{'='*60}")
    print(f"  {success}/{total} succeeded, {fail} failed")
    print(f"  {total_tables} tables extracted")
    print(f"  Wall time: {elapsed:.1f}s ({elapsed/total:.1f}s/file)")
    print(f"{'='*60}")

    return elapsed
