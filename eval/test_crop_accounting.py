"""P33: scheduler inflight_crops returns to 0 on every task outcome; _table_scale_cached keyed per Page (CPU only)."""
import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from assay import scheduler_v2 as S  # noqa: E402


def _sched(est=2, retries=2):
    s = object.__new__(S.SchedulerV2)
    s.est_crops = est
    s.calib_chunk = 32
    s.max_retries = retries
    s.inflight_pages = s.inflight_crops = s.inflight_text = 0
    s.stage_stats = {}
    s.books = {}
    s.finished = []
    s._finish_book = lambda b, failed=False: s.finished.append((b.book_id, failed))
    b = S._Book(0, "/x/book.pdf")
    b.n_pages = 10
    s.books[0] = b
    return s, b


def _send(s, task):
    # mirrors SchedulerV2.dispatch accounting
    w = s._weight(task)
    s.inflight_pages += w
    s.inflight_crops += s._crop_est(task)
    s.books[task["book"]].inflight_pages += w
    return task


def _detect(p):
    return {"kind": "detect", "book": 0, "page_idx": p, "retry": 0}


def _det_res(task, n, pairs=True):
    return {"task": task, "ok": True, "data": {
        "n_crops": n, "pairs": [object()] * n if pairs and n else [],
        "scale_x": 1.0, "scale_y": 1.0}}


def _pop_build(b):
    return b.pending[S.PRIO_BUILD].popleft()


def test_detect_then_build_ok():
    s, b = _sched()
    s.handle_result(_det_res(_send(s, _detect(0)), 3))
    assert s.inflight_crops == 3
    bt = _send(s, _pop_build(b))
    s.handle_result({"task": bt, "ok": True, "data": {"tables": []}})
    assert s.inflight_crops == 0 and s.inflight_pages == 0


def test_detect_without_pairs():
    s, b = _sched()
    s.handle_result(_det_res(_send(s, _detect(0)), 0, pairs=False))
    assert s.inflight_crops == 0 and b.pages_done == 1


def test_build_retry_then_ok():
    s, b = _sched()
    s.handle_result(_det_res(_send(s, _detect(0)), 4))
    bt = _send(s, _pop_build(b))
    s.handle_result({"task": bt, "ok": False, "error": "gpu timeout"})
    assert s.inflight_crops == 4          # still owed by the requeued build
    bt = _send(s, _pop_build(b))
    s.handle_result({"task": bt, "ok": True, "data": {"tables": []}})
    assert s.inflight_crops == 0


def test_build_retries_exhausted():
    s, b = _sched(retries=0)
    s.handle_result(_det_res(_send(s, _detect(0)), 5))
    bt = _send(s, _pop_build(b))
    s.handle_result({"task": bt, "ok": False, "error": "boom"})
    assert s.inflight_crops == 0 and not b.ok


def test_book_fail_clears_pending_builds_and_late_results():
    s, b = _sched(retries=0)
    s.handle_result(_det_res(_send(s, _detect(0)), 2))     # build queued, not sent
    s.handle_result(_det_res(_send(s, _detect(1)), 3))
    late = _send(s, b.pending[S.PRIO_BUILD].pop())         # page 1 build in flight
    d2 = _send(s, _detect(2))
    s.handle_result({"task": d2, "ok": False, "error": "x"})  # fails the book
    assert not b.ok and not b.has_pending()
    assert s.inflight_crops == 3                            # only the in-flight build
    s.handle_result({"task": late, "ok": True, "data": {"tables": []}})
    assert s.inflight_crops == 0 and s.inflight_pages == 0


def test_late_failed_build_after_book_failure():
    s, b = _sched(retries=0)
    s.handle_result(_det_res(_send(s, _detect(0)), 2))
    bt = _send(s, _pop_build(b))
    s._fail_book(b, "other")
    s.handle_result({"task": bt, "ok": False, "error": "late"})
    assert s.inflight_crops == 0


class _Q:
    def __init__(self):
        self.items = deque()

    def put(self, t):
        self.items.append(t)


class _Evt:
    def is_set(self):
        return False


def _dispatching(est=2, max_pages=256, max_crops=512):
    s, b = _sched(est=est)
    s.pause_evt = _Evt()
    s.task_q = _Q()
    s.max_inflight_pages = max_pages
    s.max_inflight_crops = max_crops
    s.a_floor = 0
    s.rr_ptr = 0
    s.active = [b]
    return s, b


def test_pending_builds_over_budget_still_dispatch():
    s, b = _dispatching()
    b.n_pages = 300
    for p in range(200):
        s.handle_result(_det_res(_send(s, _detect(p)), 3))
    assert s.inflight_crops == 600 and s.inflight_pages == 0
    s.dispatch()
    assert len(s.task_q.items) == 200
    assert all(t["kind"] == "build" for t in s.task_q.items)


def test_detect_deferred_text_flows_when_crops_full():
    s, b = _dispatching(max_crops=4)
    s.inflight_crops = 4                     # owed by builds already in flight
    b.pending[S.PRIO_DETECT].append(_detect(1))
    b.pending[S.PRIO_TEXT].append({"kind": "text", "book": 0, "page_idx": 2, "retry": 0})
    s.dispatch()
    assert [t["kind"] for t in s.task_q.items] == ["text"]
    assert len(b.pending[S.PRIO_DETECT]) == 1
    s.inflight_crops = 0
    s.dispatch()
    assert [t["kind"] for t in s.task_q.items] == ["text", "detect"]


def test_dense_book_drains_through_dispatch():
    s, b = _dispatching()
    b.n_pages = 600
    for p in range(600):
        b.pending[S.PRIO_DETECT].append(_detect(p))
    for _ in range(20000):
        s.dispatch()
        if not s.task_q.items:
            break
        t = s.task_q.items.popleft()
        if t["kind"] == "detect":
            s.handle_result(_det_res(t, 4))
        else:
            s.handle_result({"task": t, "ok": True, "data": {"tables": []}})
    assert b.pages_done == 600 and s.finished == [(0, False)]
    assert s.inflight_crops == 0 and s.inflight_pages == 0


def test_many_pages_never_negative():
    import random
    rng = random.Random(3)
    s, b = _sched()
    b.n_pages = 400
    builds = deque()
    lo = 0
    for p in range(400):
        s.handle_result(_det_res(_send(s, _detect(p)), rng.choice([0, 0, 1, 2, 5, 9])))
        while b.pending[S.PRIO_BUILD]:
            builds.append(_send(s, _pop_build(b)))
        if builds and rng.random() < 0.6:
            s.handle_result({"task": builds.popleft(), "ok": True, "data": {"tables": []}})
        lo = min(lo, s.inflight_crops)
    while builds:
        s.handle_result({"task": builds.popleft(), "ok": True, "data": {"tables": []}})
    assert lo >= 0 and s.inflight_crops == 0 and s.inflight_pages == 0


def test_scale_cache_per_page():
    import fitz
    from assay import completion as C
    calls = []
    orig = C._table_scale
    C._table_scale = lambda page, box: calls.append(id(page)) or float(len(calls))
    try:
        d = fitz.open()
        p1, p2 = d.new_page(), d.new_page()
        box = (10.2, 20.0, 300.0, 400.0)
        v1 = C._table_scale_cached(p1, box)
        assert C._table_scale_cached(p1, (10.4, 20.0, 300.0, 400.0)) == v1   # rounded key hit
        v2 = C._table_scale_cached(p2, box)
        assert v2 != v1 and len(calls) == 2
        assert not hasattr(C, "_SCALE_CACHE")
        p1b = d[0]                                     # fresh Page object: no stale hit
        C._table_scale_cached(p1b, box)
        assert len(calls) == 3
    finally:
        C._table_scale = orig
