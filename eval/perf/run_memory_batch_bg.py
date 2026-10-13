#!/usr/bin/env python3
"""
run_memory_batch_bg.py — background-friendly memory batch test.

Usage:
    python3 run_memory_batch_bg.py <assay_root> <pdf_dir> <out_dir> <workers> <samples_csv>

Behavior:
- Spawns `assay.py -w <workers> -p <pdf_dir> -o <out_dir> --gpu 0 -v` as
  a child via subprocess.Popen, NOT captured.
- Spawns a 1-Hz ps RSS sampler that records (timestamp, n_workers, total_rss_mib).
- Daemonizes via os.fork() so this script returns control immediately.
- The actual memory batch will run to completion (~5 min for 12 books, 12 workers)
  while the calling bash session can move on.

After the call returns, monitor via:
    ps -ef | grep assay.py
    tail -f <samples_csv>
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main(assay_root: str, pdf_dir: str, out_dir: str, workers: int, samples_csv: str):
    # Fork so the parent can return immediately; child does the actual work.
    pid = os.fork()
    if pid > 0:
        # Parent — return immediately.
        print(f"[memory-batch-bg] forked child PID={pid}")
        print(f"[memory-batch-bg] csv will be at {samples_csv}")
        sys.exit(0)

    # Child — session leader, fully detach.
    os.setsid()
    sys.stdin.close()
    sys.stdout.close()
    sys.stderr.close()

    log_path = Path(samples_csv).with_suffix(".run.log")
    log = open(log_path, "w")
    log.write(f"[bg] start assay_root={assay_root} pdf_dir={pdf_dir} out_dir={out_dir}\n")
    log.flush()

    # Spawn assay.py
    cmd = [
        "python3", "assay.py",
        "-p", pdf_dir,
        "-o", out_dir,
        "-w", str(workers),
        "--gpu", "0",
        "-v",
    ]
    log.write(f"[bg] cmd: {cmd}\n")
    log.flush()
    assay_proc = subprocess.Popen(cmd, cwd=assay_root,
                                  stdout=log, stderr=subprocess.STDOUT)

    # Sample loop
    csv = open(samples_csv, "w")
    csv.write("timestamp_sec,n_workers,total_rss_mib\n")
    csv.flush()
    t0 = time.time()
    while True:
        ret = assay_proc.poll()
        n_workers = 0
        total_rss_mib = 0.0
        try:
            ps_out = subprocess.check_output(
                ["ps", "-eo", "rss,comm,args"], text=True, timeout=5,
            )
            for line in ps_out.splitlines()[1:]:
                parts = line.strip().split(None, 1)
                if not parts:
                    continue
                try:
                    rss_kb = int(parts[0])
                except ValueError:
                    continue
                rest = parts[1] if len(parts) > 1 else ""
                if "python3" in rest and ("assay" in rest or "spawn" in rest):
                    n_workers += 1
                    total_rss_mib += rss_kb / 1024.0
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            log.write(f"[bg] ps error: {e}\n")
        csv.write(f"{time.time() - t0:.2f},{n_workers},{total_rss_mib:.1f}\n")
        csv.flush()
        if ret is not None:
            log.write(f"[bg] assay exit={ret}\n")
            csv.flush()
            break
        time.sleep(1.0)
    log.close()
    csv.close()
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        sys.exit("usage: run_memory_batch_bg.py <assay_root> <pdf_dir> <out_dir> <workers> <samples_csv>")
    main(*sys.argv[1:])
