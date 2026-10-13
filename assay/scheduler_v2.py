"""
页粒度流水线调度器（track_p3b 阶段 4，P2 设计第 1 层）。

单调度器进程，驱动：
  - GPU 常驻单推理服务（assay.gpu_server，独占 GPU0）
  - CPU 工作池（assay.worker_v2，无模型无 torch）

书生命周期：注册 → 页眉校准分片（并行）→ 合并签名 → 全书文本任务 A
→ 候选页检测任务 B（渲染 + YOLO/门控/结构走 GPU 服务）→ 构建任务 D
（build_table_from_slanet + complete_table 全套）→ 装配导出（md/json，
与 legacy _process_one 相同的嵌套目录与格式）。

调度策略：单队列优先级 D > B > A（A 保底供给防断流）+ 书间 round-robin
公平 + 页预算（在制页 ≤256 / 在制 crop ≤512 / 在制书 ≤50）。

看门狗线程（P2 §watchdog）：MemAvailable 低于水位暂停发新任务 / 收缩
worker 槽位，恢复水位持续 5s 后回满；loadavg 超限暂停扩槽 —— 恒留
1 CPU 核 + 5GB 内存余量防 OOM。GPU 服务心跳超时 → 终止并以同一队列
重启（任务幂等，worker 超时后由调度器重投）。

配置：pipeline_config.yaml（零硬编码）。算法零改动：全部页体复用
assay 包既有函数（calibrate_page_headers_scan_page / extract_page_text /
detect_tables_on_page / prepare_page_slanet_data /
finalize_table_from_slanet），本文件只做编排。
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

PRIO_BUILD = 0
PRIO_DETECT = 1
PRIO_TEXT = 2


def load_pipeline_config(search_dirs: Optional[List[str]] = None) -> dict:
    """读 pipeline_config.yaml（v2 管道唯一阈值源；缺失时用内置默认值）。"""
    import yaml

    candidates = []
    if search_dirs:
        candidates.extend(Path(d) / "pipeline_config.yaml" for d in search_dirs)
    candidates.append(Path(__file__).resolve().parent.parent
                      / "pipeline_config.yaml")
    for c in candidates:
        if c.is_file():
            with open(c, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    _log.warning("pipeline_config.yaml not found; using built-in defaults")
    return {}


class _Book:
    """BookRegistry 单书状态：页状态机 + 归集结果。"""

    def __init__(self, book_id: int, pdf_path: str):
        self.book_id = book_id
        self.pdf_path = pdf_path
        self.stem = Path(pdf_path).stem
        self.n_pages = 0
        self.ok = True
        self.status = "SUCCESS"
        self.error = ""
        self.finished = False
        self.t_start: Optional[float] = None
        self.t_end: Optional[float] = None
        self.calib_pending = 0
        self.calib_partials: List[dict] = []
        self.header_sigs: Optional[set] = None
        self.pending: Dict[int, deque] = {p: deque() for p in
                                          (PRIO_BUILD, PRIO_DETECT, PRIO_TEXT)}
        self.inflight_pages = 0
        self.texts: Dict[int, list] = {}
        self.page_dims: Dict[int, tuple] = {}
        self.cand_pages: set = set()
        self.pairs: Dict[int, list] = {}
        self.scales: Dict[int, tuple] = {}
        self.tables: Dict[int, list] = {}
        self.pages_done = 0

    def has_pending(self) -> bool:
        return any(self.pending[p] for p in self.pending)


class SchedulerV2:
    def __init__(self, output_dir: str, num_workers: int, no_json: bool,
                 debug_pdf: bool, cfg: dict):
        import multiprocessing as mp

        self.cfg = cfg
        self.output_dir = output_dir
        self.no_json = no_json
        self.debug_pdf = debug_pdf
        self.ctx = mp.get_context("spawn")

        sched = cfg.get("scheduler") or {}
        budget = cfg.get("budgets") or {}
        wd = cfg.get("watchdog") or {}
        workers_cfg = int(cfg.get("workers", 17))
        self.num_workers = max(2, num_workers if (num_workers or 0) > 1
                               else workers_cfg)
        self.calib_chunk = max(1, int(sched.get("calib_chunk_pages", 32)))
        self.a_floor_frac = float(sched.get("a_floor_fraction", 0.25))
        self.max_retries = int(sched.get("max_task_retries", 2))
        self.max_active_books = int(sched.get("max_active_books", 50))
        self.max_inflight_pages = int(budget.get("max_inflight_pages", 256))
        self.max_inflight_crops = int(budget.get("max_inflight_crops", 512))
        self.est_crops = int(budget.get("est_crops_per_page", 2))
        self.a_floor = max(2, int(self.a_floor_frac * self.num_workers))
        self.gpu_timeout_s = float(sched.get("gpu_timeout_s", 60.0))

        self.wd_cfg = wd
        self.gpu_cfg = cfg.get("gpu_server") or {}

        self.inflight_pages = 0
        self.inflight_crops = 0
        self.inflight_text = 0

        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.infer_q = self.ctx.Queue()
        # track_p3b 集成修复：mp.Queue 不能作为消息体跨队列 pickle，
        # 每 worker 专属回包队列改为进程创建时按继承传递。
        self.reply_queues = [self.ctx.Queue()
                             for _ in range(self.num_workers)]
        self.heartbeat = self.ctx.Value("d", 0.0)
        self.active_slots = self.ctx.Value("i", self.num_workers)
        self.pause_evt = self.ctx.Event()

        self.books: Dict[int, _Book] = {}
        self.queue_books: deque = deque()
        self.active: List[_Book] = []
        self.rr_ptr = 0

        self.gpu_proc = None
        self.gpu_restarts = 0
        self.gpu_spawn_t = 0.0
        self._restart_requested = threading.Event()
        self._stop = threading.Event()
        self.done_count = 0
        self.fail_count = 0
        self.total_tables = 0
        # track_p3b 画像：按任务类别聚合 worker 执行耗时与 GPU 等待
        self.stage_stats: Dict[str, dict] = {}

    # ── 记账辅助 ────────────────────────────────────────────────────────

    def _weight(self, task) -> int:
        if task["kind"] == "calib":
            return min(self.calib_chunk, task["page_hi"] - task["page_lo"])
        return 1

    def _crop_est(self, task) -> int:
        return self.est_crops if task["kind"] == "detect" else 0

    def _decrement(self, task):
        w = self._weight(task)
        self.inflight_pages -= w
        self.books[task["book"]].inflight_pages -= w
        self.inflight_crops -= self._crop_est(task)
        if task["kind"] in ("text", "calib"):
            self.inflight_text -= w

    # ── GPU 服务管理 ────────────────────────────────────────────────────

    def _spawn_gpu_server(self):
        from assay.gpu_server import gpu_server_main
        self.gpu_spawn_t = time.time()
        self.gpu_proc = self.ctx.Process(
            target=gpu_server_main,
            args=(self.infer_q, self.heartbeat, self.reply_queues,
                  dict(self.gpu_cfg)),
            name="gpu-server", daemon=False)
        self.gpu_proc.start()
        _log.info("[v2] gpu server spawned (pid %s)", self.gpu_proc.pid)

    def _maybe_restart_gpu_server(self):
        if not self._restart_requested.is_set():
            return
        self._restart_requested.clear()
        self.gpu_restarts += 1
        _log.error("[v2] gpu server heartbeat stale — restarting (#%d)",
                   self.gpu_restarts)
        if self.gpu_proc is not None and self.gpu_proc.is_alive():
            self.gpu_proc.terminate()
            self.gpu_proc.join(5)
            if self.gpu_proc.is_alive():
                self.gpu_proc.kill()
                self.gpu_proc.join(5)
        self._spawn_gpu_server()

    # ── 书注册与激活 ────────────────────────────────────────────────────

    def add_books(self, pdf_paths: List[str]):
        for i, p in enumerate(pdf_paths):
            self.books[i] = _Book(i, p)
            self.queue_books.append(self.books[i])

    def _activate_books(self):
        import fitz
        while self.queue_books and len(self.active) < self.max_active_books:
            book = self.queue_books.popleft()
            try:
                doc = fitz.open(book.pdf_path)
                book.n_pages = len(doc)
                doc.close()
            except Exception as e:  # noqa: BLE001
                book.status = "ERROR"
                book.error = f"open failed: {e}"
                book.t_start = book.t_end = time.time()
                self._finish_book(book, failed=True)
                continue
            book.t_start = time.time()
            n_chunks = math.ceil(book.n_pages / self.calib_chunk)
            book.calib_pending = n_chunks
            for c in range(n_chunks):
                lo = c * self.calib_chunk
                hi = min(lo + self.calib_chunk, book.n_pages)
                book.pending[PRIO_TEXT].append({
                    "kind": "calib", "book": book.book_id,
                    "path": book.pdf_path, "page_lo": lo,
                    "page_hi": hi, "chunk_id": c, "retry": 0,
                })
            self.active.append(book)
            _log.info("[v2] book activated: %s (%d pages, %d calib chunks)",
                      book.stem, book.n_pages, n_chunks)

    @property
    def _per_book_page_cap(self) -> int:
        n = max(1, len(self.active))
        return max(4, math.ceil(self.max_inflight_pages / n))

    # ── 派发 ───────────────────────────────────────────────────────────

    def _pop_task(self) -> Optional[dict]:
        """优先级 D>B>A + A 保底 + 书间 round-robin。"""
        order = self.active
        if not order:
            return None
        n = len(order)
        if self.inflight_text < self.a_floor:
            for k in range(n):
                b = order[(self.rr_ptr + k) % n]
                if b.ok and b.pending[PRIO_TEXT] \
                        and b.inflight_pages < self._per_book_page_cap:
                    self.rr_ptr = (self.rr_ptr + k + 1) % n
                    return b.pending[PRIO_TEXT].popleft()
        for prio in (PRIO_BUILD, PRIO_DETECT, PRIO_TEXT):
            for k in range(n):
                b = order[(self.rr_ptr + k) % n]
                if b.ok and b.pending[prio] \
                        and b.inflight_pages < self._per_book_page_cap:
                    self.rr_ptr = (self.rr_ptr + k + 1) % n
                    return b.pending[prio].popleft()
        return None

    @staticmethod
    def _prio_of(kind: str) -> int:
        if kind == "build":
            return PRIO_BUILD
        if kind == "detect":
            return PRIO_DETECT
        return PRIO_TEXT

    def dispatch(self):
        if self.pause_evt.is_set():
            return
        while True:
            if self.inflight_pages >= self.max_inflight_pages \
                    or self.inflight_crops >= self.max_inflight_crops:
                return
            task = self._pop_task()
            if task is None:
                return
            if self.inflight_crops + self._crop_est(task) \
                    > self.max_inflight_crops:
                self.books[task["book"]].pending[self._prio_of(
                    task["kind"])].appendleft(task)
                return
            w = self._weight(task)
            self.inflight_pages += w
            self.inflight_crops += self._crop_est(task)
            self.books[task["book"]].inflight_pages += w
            if task["kind"] in ("text", "calib"):
                self.inflight_text += w
            self.task_q.put(task)

    # ── 结果处理 ────────────────────────────────────────────────────────

    def _page_terminal(self, book: _Book):
        book.pages_done += 1
        if book.pages_done >= book.n_pages:
            book.t_end = time.time()
            self._finish_book(book)

    def _fail_book(self, book: _Book, error: str):
        if not book.ok:
            return
        book.ok = False
        book.status = "ERROR"
        book.error = error
        for p in book.pending:
            book.pending[p].clear()
        book.t_end = time.time()
        self._finish_book(book, failed=True)

    def _finish_book(self, book: _Book, failed: bool = False):
        if book.finished:
            return
        book.finished = True
        from assay.output import (DocumentOutput, PageOutput,
                                  write_markdown, write_json,
                                  write_debug_pdf)
        doc_out = DocumentOutput(file_name=book.stem)
        if failed:
            doc_out.status = book.status
            doc_out.error = book.error
        for p in range(book.n_pages):
            w, h = book.page_dims.get(p, (0.0, 0.0))
            doc_out.pages.append(PageOutput(page_no=p, width=w, height=h))
            doc_out.texts.extend(book.texts.get(p, []))
            doc_out.tables.extend(book.tables.get(p, []))

        out_dir = Path(self.output_dir) / book.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        write_markdown(doc_out, str(out_dir / f"{book.stem}.md"))
        if not self.no_json:
            write_json(doc_out, str(out_dir / f"{book.stem}.json"), indent=2)
        if self.debug_pdf:
            write_debug_pdf(doc_out, book.pdf_path,
                            str(out_dir / f"{book.stem}_layout.pdf"))

        self.done_count += 1
        n_tables = len(doc_out.tables)
        self.total_tables += n_tables
        if failed:
            self.fail_count += 1
            print(f"  [{self.done_count:3d}/{len(self.books)}] FAIL | "
                  f"{book.stem[:45]:45s} | {book.error[:60]}", flush=True)
        else:
            dt = (book.t_end or time.time()) - (book.t_start or time.time())
            print(f"  [{self.done_count:3d}/{len(self.books)}] | "
                  f"{book.stem[:45]:45s} | {dt:5.1f}s | "
                  f"{n_tables:4d} tables", flush=True)
        if book in self.active:
            self.active.remove(book)
            if self.rr_ptr >= len(self.active):
                self.rr_ptr = 0
        self._activate_books()

    def _handle_calib(self, book: _Book, res: dict):
        book.calib_partials.extend(res["data"])
        book.calib_pending -= 1
        if book.calib_pending == 0:
            from assay.detection import register_header_calibration
            # doc.name == 本书任务里的 path 字符串 → 与导出侧
            # statement_caption_info 的 _file_stem 匹配口径一致
            book.header_sigs = register_header_calibration(
                book.pdf_path, book.n_pages, book.calib_partials)
            book.calib_partials = []
            for p in range(book.n_pages):
                book.pending[PRIO_TEXT].append({
                    "kind": "text", "book": book.book_id,
                    "path": book.pdf_path, "page_idx": p,
                    "header_sigs": book.header_sigs, "retry": 0,
                })

    def _handle_text(self, book: _Book, res: dict):
        data = res["data"]
        p = res["task"]["page_idx"]
        book.texts[p] = data["texts"]
        book.page_dims[p] = (data["w"], data["h"])
        if data["cand"]:
            book.cand_pages.add(p)
            book.pending[PRIO_DETECT].append({
                "kind": "detect", "book": book.book_id,
                "path": book.pdf_path, "page_idx": p,
                "header_sigs": book.header_sigs, "retry": 0,
            })
        else:
            self._page_terminal(book)

    def _handle_detect(self, book: _Book, res: dict):
        data = res["data"]
        p = res["task"]["page_idx"]
        # crop 预算校正：派发时按 est 占用，此处换成实际 crop 数；
        # build 完成时再释放（见 _handle_build）
        actual = int(data["n_crops"])
        self.inflight_crops += actual - self.est_crops
        if data["pairs"]:
            book.pairs[p] = data["pairs"]
            book.pending[PRIO_BUILD].append({
                "kind": "build", "book": book.book_id,
                "path": book.pdf_path, "page_idx": p,
                "header_sigs": book.header_sigs,
                "items": data["pairs"],
                "scale_x": data["scale_x"], "scale_y": data["scale_y"],
                "n_crops": actual, "retry": 0,
            })
        else:
            self.inflight_crops -= actual
            self._page_terminal(book)

    def _handle_build(self, book: _Book, res: dict):
        task = res["task"]
        p = task["page_idx"]
        book.tables[p] = res["data"]["tables"]
        self.inflight_crops -= int(task.get("n_crops", 0))
        self._page_terminal(book)

    def handle_result(self, res: dict):
        task = res["task"]
        book = self.books[task["book"]]
        self._decrement(task)
        st = self.stage_stats.setdefault(
            task["kind"], {"n": 0, "dt": 0.0, "dt_max": 0.0,
                           "gpu_n": 0, "gpu_s": 0.0})
        st["n"] += 1
        _dt = float(res.get("dt", 0.0))
        st["dt"] += _dt
        st["dt_max"] = max(st["dt_max"], _dt)
        for _ch, _v in (res.get("gpu") or {}).items():
            st["gpu_n"] += int(_v[0])
            st["gpu_s"] += float(_v[1])
        if not res.get("ok"):
            if book.ok:
                if task.get("retry", 0) >= self.max_retries:
                    self._fail_book(book, f"{task['kind']} task on page "
                                           f"{task.get('page_idx')} failed "
                                           f"after {self.max_retries} "
                                           f"retries: {res.get('error')}")
                else:
                    task["retry"] = task.get("retry", 0) + 1
                    book.pending[self._prio_of(task["kind"])].appendleft(task)
            return
        if not book.ok:
            return  # 迟到的成功结果，忽略
        kind = task["kind"]
        if kind == "calib":
            self._handle_calib(book, res)
        elif kind == "text":
            self._handle_text(book, res)
        elif kind == "detect":
            self._handle_detect(book, res)
        elif kind == "build":
            self._handle_build(book, res)

    # ── 看门狗线程 ──────────────────────────────────────────────────────

    def _watchdog_loop(self):
        cfg = self.wd_cfg
        interval = float(cfg.get("sample_interval_s", 1.0))
        pause_mib = float(cfg.get("pause_mem_available_mib", 6144))
        shrink_mib = float(cfg.get("shrink_mem_available_mib", 5325))
        resume_mib = float(cfg.get("resume_mem_available_mib", 8192))
        sustain_s = float(cfg.get("resume_sustain_s", 5.0))
        min_active = int(cfg.get("min_active_workers", 8))
        load_pause = float(cfg.get("loadavg_pause", 19.0))
        stale_s = float(cfg.get("heartbeat_stale_s", 3.0))
        grace_s = float(cfg.get("startup_grace_s", 180.0))
        resume_since: Optional[float] = None
        last_heartbeat = self.heartbeat.value

        while not self._stop.is_set():
            try:
                mem_mib = _mem_available_mib()
                load1 = _load1()
            except Exception:  # noqa: BLE001
                time.sleep(interval)
                continue

            if mem_mib < shrink_mib:
                if self.active_slots.value > min_active:
                    self.active_slots.value -= 1
                    _log.warning("[watchdog] MemAvailable %.0fMiB < "
                                 "%.0fMiB — shrink slots to %d",
                                 mem_mib, shrink_mib,
                                 self.active_slots.value)
                resume_since = None
            elif mem_mib < pause_mib or load1 > load_pause:
                if not self.pause_evt.is_set():
                    _log.warning("[watchdog] mem %.0fMiB / load1 %.1f — "
                                 "pause dispatch", mem_mib, load1)
                self.pause_evt.set()
                resume_since = None
            elif mem_mib > resume_mib and load1 <= load_pause:
                if resume_since is None:
                    resume_since = time.monotonic()
                elif time.monotonic() - resume_since >= sustain_s:
                    self.pause_evt.clear()
                    if self.active_slots.value < self.num_workers:
                        self.active_slots.value = self.num_workers
                        _log.info("[watchdog] recovered — slots back to %d",
                                  self.num_workers)
                    resume_since = None

            hb = self.heartbeat.value
            if hb != last_heartbeat:
                last_heartbeat = hb
            elif (hb > 0 and time.time() - hb > stale_s
                  and time.time() - self.gpu_spawn_t > grace_s
                  and not self._restart_requested.is_set()):
                # 心跳停摆（加载宽限期之后）：请求主循环重启 GPU 服务
                self._restart_requested.set()
            time.sleep(interval)

    # ── 主入口 ──────────────────────────────────────────────────────────

    def run(self, pdf_paths: List[str]) -> float:
        import queue as _queue

        t_start = time.time()
        self.add_books(pdf_paths)
        self._activate_books()

        from assay.worker_v2 import worker_main
        worker_cfg = {
            "gpu_timeout_s": self.gpu_timeout_s,
        }
        workers = []
        for i in range(self.num_workers):
            p = self.ctx.Process(target=worker_main,
                                 args=(i, self.task_q, self.result_q,
                                       self.infer_q, self.reply_queues[i],
                                       self.active_slots, worker_cfg),
                                 name=f"worker-{i:02d}", daemon=False)
            p.start()
            workers.append(p)
        self._spawn_gpu_server()

        wd_thread = threading.Thread(target=self._watchdog_loop,
                                     name="watchdog", daemon=True)
        wd_thread.start()

        print(f"\n{'='*60}")
        print(f"  Assay — Table Extraction Pipeline (v2 page-granular)")
        print(f"  Workers: {self.num_workers} | GPU: resident server "
              f"(cuda:0)")
        print(f"  Files: {len(pdf_paths)} | Output: {self.output_dir}")
        print(f"{'='*60}", flush=True)

        idle_ticks = 0
        while self.done_count < len(self.books):
            self._maybe_restart_gpu_server()
            try:
                res = self.result_q.get(timeout=0.25)
            except _queue.Empty:
                res = None
            if res is not None:
                idle_ticks = 0
                self.handle_result(res)
            else:
                idle_ticks += 1
            self.dispatch()
            if idle_ticks > 0 and idle_ticks % 40 == 0:
                # 周期性诊断：完全无结果且无待派任务 = 卡死信号
                if self.inflight_pages == 0 and not any(
                        b.has_pending() for b in self.active) \
                        and not self.queue_books and self.active:
                    _log.error("[v2] deadlock suspected: active books with "
                               "no inflight/pending tasks")
                    for b in self.active:
                        self._fail_book(b, "scheduler deadlock detected")

        # ── 收尾 ──
        self._stop.set()
        for _ in workers:
            self.task_q.put(None)
        for p in workers:
            p.join(30)
        self.infer_q.put(None)
        if self.gpu_proc is not None:
            self.gpu_proc.join(30)

        elapsed = time.time() - t_start
        total = len(self.books)
        print(f"{'='*60}")
        print("  stage profile (worker-exec wall, incl. GPU waits):")
        for _k in ("calib", "text", "detect", "build"):
            _st = self.stage_stats.get(_k)
            if not _st:
                continue
            print(f"    {_k:6s} n={_st['n']:5d} sum={_st['dt']:8.1f}s "
                  f"max={_st['dt_max']:6.2f}s gpu_calls={_st['gpu_n']:6d} "
                  f"gpu_wait={_st['gpu_s']:8.1f}s")
        print(f"{'='*60}", flush=True)
        print(f"  {total - self.fail_count}/{total} succeeded, "
              f"{self.fail_count} failed")
        print(f"  {self.total_tables} tables extracted")
        if self.gpu_restarts:
            print(f"  GPU server restarts: {self.gpu_restarts}")
        print(f"  Wall time: {elapsed:.1f}s "
              f"({elapsed / max(1, total):.1f}s/file)")
        print(f"{'='*60}", flush=True)
        return elapsed


def _mem_available_mib() -> float:
    with open("/proc/meminfo", "r", encoding="ascii") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / 1024.0
    return float("inf")


def _load1() -> float:
    with open("/proc/loadavg", "r", encoding="ascii") as f:
        return float(f.read().split()[0])


def run_v2(pdf_dir: str, output_dir: str, num_workers: int,
           use_gpu: bool, no_json: bool, debug_pdf: bool,
           arch: str = "v2") -> float:
    """pipeline.run 的 v2 入口：目录模式切到页粒度流水线。"""
    from pathlib import Path as _P
    pdfs = sorted(_P(pdf_dir).glob("*.pdf"))
    total = len(pdfs)
    if total == 0:
        print("Error: no PDF files found")
        return 0.0
    cfg = load_pipeline_config()
    sched = SchedulerV2(output_dir=output_dir,
                        num_workers=num_workers, no_json=no_json,
                        debug_pdf=debug_pdf, cfg=cfg)
    return sched.run([str(p) for p in pdfs])
