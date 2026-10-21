"""track_p5/P31: gpu_server._drain quota semantics + exactly-once under concurrent lanes (CPU only)."""
import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from assay.gpu_server import _drain, _req_size  # noqa: E402


def _legacy_drain(buf, ch, quota):
    # pre-P31 inline loop from _serve, kept verbatim as the reference
    part = deque()
    acc = 0
    while buf:
        r = buf[0]
        if ch in ("cls", "slanet"):
            sz = len(r["payload"])
        elif ch == "cslanet":
            sz = len(r["payload"]["crops"])
        else:
            sz = 1
        if acc and acc + sz > quota:
            break
        part.append(buf.popleft())
        acc += sz
    return part, acc


def _req(ch, rid, k):
    if ch == "cslanet":
        return {"rid": rid, "payload": {"crops": [0] * k, "offsets": [0] * k}}
    if ch in ("cls", "slanet"):
        return {"rid": rid, "payload": [0] * k}
    return {"rid": rid, "payload": None}


def test_req_size():
    assert _req_size("slanet", _req("slanet", 0, 5)) == 5
    assert _req_size("cls", _req("cls", 0, 3)) == 3
    assert _req_size("cslanet", _req("cslanet", 0, 7)) == 7
    assert _req_size("yolo", _req("yolo", 0, 9)) == 1
    assert _req_size("yolo2", _req("yolo2", 0, 9)) == 1


def test_matches_legacy_sequence():
    import random
    rng = random.Random(7)
    for ch in ("slanet", "cls", "cslanet", "yolo", "yolo2"):
        for quota in (1, 8, 32):
            sizes = [rng.choice([0, 1, 2, 5, 17, 40]) for _ in range(300)]
            a = deque(_req(ch, i, k) for i, k in enumerate(sizes))
            b = deque(_req(ch, i, k) for i, k in enumerate(sizes))
            lk = threading.Lock()
            while a or b:
                pa, aa = _drain(a, ch, quota, lk)
                pb, ab = _legacy_drain(b, ch, quota)
                assert [r["rid"] for r in pa] == [r["rid"] for r in pb]
                assert aa == ab
                assert pa, "non-empty buffer must yield at least one request"


def test_oversized_request_taken_alone():
    buf = deque([_req("slanet", 0, 50), _req("slanet", 1, 1)])
    part, acc = _drain(buf, "slanet", 32, threading.Lock())
    assert [r["rid"] for r in part] == [0] and acc == 50
    assert len(buf) == 1


def test_empty_buffer():
    part, acc = _drain(deque(), "slanet", 32, threading.Lock())
    assert not part and acc == 0


def test_concurrent_lanes_exactly_once():
    n_req = 20000
    buf = deque()
    lk = threading.Lock()
    got = [[] for _ in range(4)]
    done = threading.Event()

    def producer():
        for i in range(n_req):
            buf.append(_req("slanet", i, 1 + i % 23))
            if i % 97 == 0:
                time.sleep(0)
        done.set()

    def lane(j):
        while True:
            part, acc = _drain(buf, "slanet", 32, lk)
            if not part:
                if done.is_set() and not buf:
                    return
                time.sleep(0)
                continue
            assert acc == sum(len(r["payload"]) for r in part)
            assert len(part) == 1 or acc <= 32
            got[j].extend(r["rid"] for r in part)

    th = [threading.Thread(target=lane, args=(j,)) for j in range(4)]
    for t in th:
        t.start()
    producer()
    for t in th:
        t.join(timeout=60)
    allr = sorted(r for g in got for r in g)
    assert allr == list(range(n_req))
    for g in got:
        assert g == sorted(g), "each lane must see FIFO order"


def test_config_has_slanet_lanes():
    import yaml
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    with open(os.path.join(root, "pipeline_config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert "slanet_lanes" in cfg["gpu_server"]
    assert int(cfg["gpu_server"]["slanet_lanes"]) >= 1
