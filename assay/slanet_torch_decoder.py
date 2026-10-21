"""SLANet torch decoder with CUDA Graph (P27).

Replaces the ONNX Runtime Loop that decodes the SLANet-plus structure
head.  The encoder stays in ORT (truncated graph, output
``transpose_0.tmp_0.0``); only the autoregressive GRU decoder moves to
torch, where a fixed-step CUDA Graph removes the per-step kernel launch
overhead that dominates the ORT Loop (P22R: 161.8ms kernel time of which
the launch gaps are the bulk).

Semantics replicated from the ONNX graph (verified against ORT in
track_p5/p27_equiv.py, token sequences identical on 96 real crops):

* step limit: the Loop body runs for step = 0 .. ``assign_0`` (= 500)
  inclusive, i.e. at most ``max_steps + 1`` = 501 iterations writing rows
  0..500 (the Loop trip count 1024 is dead);
* ``all_done`` = every row has emitted EOS (argmax == 49) at least once
  in its history, checked after each step's write;
* output length = steps_executed + 1, clamped to the 501-row buffer: when
  EOS stops the loop the ORT post-loop Slice keeps one extra trailing row
  of the pre-loop zero buffer (logits 0 -> softmax 0.02 uniform, loc 0);
  when no EOS ever appears all 501 rows are real tokens.

Decoder matmul precision follows ``slanet_decoder_tf32``.  Full FP32 is
batch-invariant and bit-identical to ORT run with TF32 disabled; TF32 is
not batch-invariant (tensor-core rounding depends on the other rows in
the batch and flips near-tie tokens: crop 1497 has 360 cells alone, 366
inside a batch of 48), but it is what ORT's CUDA EP uses by default.

Nothing here is hardcoded policy: slot sizes, step
limit and the on/off switch all come from pipeline_config.yaml's
``gpu_server`` section (see ``maybe_install``).
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch

_log = logging.getLogger("assay.slanet_torch_decoder")

_ENC_OUTPUT = "transpose_0.tmp_0.0"
_EOS = 49
_VOCAB = 50
_HIDDEN = 256
_FEAT = 96
_LOC_DIM = 8


def _const(graph, name: str) -> np.ndarray:
    """Fetch a weight that may live as an initializer or a Constant node."""
    for init in graph.initializer:
        if init.name == name:
            from onnx import numpy_helper
            return numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            from onnx import numpy_helper
            for attr in node.attribute:
                if attr.name == "value":
                    return numpy_helper.to_array(attr.t)
    raise KeyError(name)


def build_encoder_session(model_path: str, providers):
    """ORT session for the encoder only: graph truncated at the Loop.

    Appending the encoder tensor as an extra output still executes the
    Loop (ORT runs the whole graph), so the graph is physically cut: the
    Loop node and everything downstream of it is removed and
    ``transpose_0.tmp_0.0`` becomes the sole output.
    """
    import onnx
    from onnx import helper

    model = onnx.load(model_path)
    graph = model.graph
    # Keep only the nodes that feed the encoder tensor.  The Loop and
    # everything downstream of it (plus the ConstantOfShape scratch
    # buffers only the Loop consumes) is dropped, so the session never
    # enters the decode loop.
    producers = {o: n for n in graph.node for o in n.output}
    needed = set()
    stack = [_ENC_OUTPUT]
    while stack:
        name = stack.pop()
        node = producers.get(name)
        if node is None or id(node) in needed:
            continue
        needed.add(id(node))
        stack.extend(list(node.input))
    keep = [n for n in graph.node if id(n) in needed]
    consumed = {i for n in keep for i in n.input}
    del graph.node[:]
    graph.node.extend(keep)
    alive = [i for i in graph.initializer if i.name in consumed]
    del graph.initializer[:]
    graph.initializer.extend(alive)
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info(
        _ENC_OUTPUT, onnx.TensorProto.FLOAT, ["b", _HIDDEN, _FEAT]))
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), opts,
                                providers=providers)


class _Slot:
    """One fixed-batch CUDA Graph decoder.  Padding rows are pre-marked
    done so they never delay the batch-level stop, and the replay loop
    stops at the same step the ORT Loop would."""

    def __init__(self, B: int, max_steps: int, weights: Dict[str, torch.Tensor]):
        self.B = B
        self.max_steps = max_steps
        dev = weights["W0"].device
        self.dev = dev
        W = weights
        self.W0, self.W1, self.b1, self.W2 = W["W0"], W["W1"], W["b1"], W["W2"]
        self.G0t, self.G1t = W["G0t"], W["G1t"]
        self.gb0, self.gb1 = W["gb0"], W["gb1"]
        self.W3, self.b3, self.W4, self.b4 = W["W3"], W["b3"], W["W4"], W["b4"]
        self.W5, self.b5, self.W6, self.b6 = W["W5"], W["b5"], W["W6"], W["b6"]
        self.eye = torch.eye(_VOCAB, device=dev)

        self.F = torch.zeros(B, _HIDDEN, _FEAT, device=dev)
        self.h = torch.zeros(B, _HIDDEN, device=dev)
        self.pre = torch.zeros(B, dtype=torch.long, device=dev)
        self.FW0 = torch.empty(B, _HIDDEN, _HIDDEN, device=dev)
        self.step = torch.zeros((), dtype=torch.long, device=dev)
        self.done = torch.zeros(B, dtype=torch.bool, device=dev)
        # rows 0..max_steps (max_steps + 1 decode steps at most).
        self.logits = torch.zeros(B, max_steps + 1, _VOCAB, device=dev)
        self.loc = torch.zeros(B, max_steps + 1, _LOC_DIM, device=dev)
        # host mirror of `done`, refreshed after every replayed step so the
        # stop falls on the same step the ORT Loop would stop on.
        self.done_host = torch.zeros(B, dtype=torch.bool, pin_memory=True)
        self.graph = self._capture()

    def _step_fn(self):
        a = torch.tanh(self.FW0 + (self.h @ self.W1 + self.b1).unsqueeze(1))
        alpha = torch.softmax(a @ self.W2, dim=1)
        ctx = (alpha.transpose(1, 2) @ self.F).squeeze(1)
        gi = torch.cat([ctx, self.eye[self.pre]], 1) @ self.G0t + self.gb0
        gh = self.h @ self.G1t + self.gb1
        ir, iz, in_ = gi.chunk(3, 1)
        hr, hz, hn_ = gh.chunk(3, 1)
        r = torch.sigmoid(ir + hr)
        z = torch.sigmoid(iz + hz)
        n = torch.tanh(in_ + r * hn_)
        self.h.copy_((self.h - n) * z + n)
        s = (self.h @ self.W3 + self.b3) @ self.W4 + self.b4
        lo = torch.sigmoid((self.h @ self.W5 + self.b5) @ self.W6 + self.b6)
        idx = self.step.view(1)
        self.logits.index_copy_(1, idx, s.unsqueeze(1))
        self.loc.index_copy_(1, idx, lo.unsqueeze(1))
        self.pre.copy_(s.argmax(1))
        self.done.logical_or_(self.pre == _EOS)
        self.step.add_(1)

    def _capture(self) -> "torch.cuda.CUDAGraph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step_fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step_fn()
        return graph

    @torch.no_grad()
    def run(self, feats: torch.Tensor) -> tuple:
        """feats: [b, 256, 96] with b <= B.  Returns (loc, probs) numpy
        arrays of shape [b, L+1, *] where L is the executed step count."""
        b = feats.shape[0]
        self.F[:b].copy_(feats)
        if b < self.B:
            self.F[b:].zero_()
        self.h.zero_()
        self.pre.zero_()
        # padding rows are done from the start: they must not hold the
        # batch-level stop open (mirrors ORT, where the batch is exactly b).
        self.done.zero_()
        if b < self.B:
            self.done[b:] = True
        self.logits.zero_()
        self.loc.zero_()
        self.FW0.copy_(self.F @ self.W0)
        self.step.zero_()

        # one replay = one decode step, so the stop lands on the same step
        # as the ORT Loop (overshooting would append tokens past EOS, which
        # the label decoder would read as extra structure).
        executed = self.max_steps + 1
        for _ in range(self.max_steps + 1):
            self.graph.replay()
            self.done_host.copy_(self.done, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            if bool(self.done_host[:b].all()):
                executed = int(self.step.item())
                break
        # trailing row at position `executed` stays the zero buffer row; on
        # the no-EOS path the slice clamps to the max_steps + 1 real rows.
        logits = self.logits[:b, :executed + 1]
        loc = self.loc[:b, :executed + 1]
        return (loc.float().cpu().numpy(),
                torch.softmax(logits.float(), dim=-1).cpu().numpy())


class TorchDecoderSession:
    """Duck-typed stand-in for the ORT InferenceSession that rapid_table's
    OrtInferSession drives.  ``run`` returns ``[loc, struct_probs]`` in
    the model's output-name order; everything else delegates to the real
    ORT session so metadata (character dictionary) stays identical.
    """

    def __init__(self, model_path: str, providers, slot_sizes: List[int],
                 max_steps: int, fallback_session, tf32: bool = False):
        import onnx
        model = onnx.load(model_path)
        dev = "cuda"
        T = lambda name: torch.from_numpy(
            np.array(_const(model.graph, name), copy=True)).to(dev)
        G0 = T("gru_cell_0.w_0")
        G1 = T("gru_cell_0.w_1")
        weights = {
            "W0": T("linear_0.w_0"), "W1": T("linear_1.w_0"),
            "b1": T("linear_1.b_0"), "W2": T("linear_2.w_0"),
            "G0t": G0.t().contiguous(), "G1t": G1.t().contiguous(),
            "gb0": T("gru_cell_0.b_0"), "gb1": T("gru_cell_0.b_1"),
            "W3": T("linear_3.w_0"), "b3": T("linear_3.b_0"),
            "W4": T("linear_4.w_0"), "b4": T("linear_4.b_0"),
            "W5": T("linear_5.w_0"), "b5": T("linear_5.b_0"),
            "W6": T("linear_6.w_0"), "b6": T("linear_6.b_0"),
        }
        # Precision is baked into the CUDA Graphs at capture time, so it is
        # set before the slots are built.  This process only hosts cls +
        # SLANet (split_servers), so the global flag reaches nothing else.
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision("high" if tf32 else "highest")
        self.encoder = build_encoder_session(model_path, providers)
        self.fallback = fallback_session
        self.slots = {
            B: _Slot(B, max_steps, weights) for B in slot_sizes
        }
        self._slot_sizes = sorted(slot_sizes)
        self.fallback_count = 0
        _log.info("SLANet torch decoder ready (slots=%s, max_steps=%d)",
                  self._slot_sizes, max_steps)

    # ── ORT session surface used by rapid_table ─────────────────────────
    def get_inputs(self):
        return self.fallback.get_inputs()

    def get_outputs(self):
        return self.fallback.get_outputs()

    def get_modelmeta(self):
        return self.fallback.get_modelmeta()

    def get_providers(self):
        return self.fallback.get_providers()

    def run(self, output_names, input_feed):
        try:
            return self._run(input_feed)
        except Exception as exc:  # quality over speed: never fail a batch
            self.fallback_count += 1
            _log.warning("torch decoder failed (%s); ORT fallback #%d",
                         exc, self.fallback_count)
            return self.fallback.run(output_names, input_feed)

    def _run(self, input_feed) -> list:
        x = next(iter(input_feed.values()))
        if isinstance(x, (list, tuple)):
            x = np.stack([np.asarray(t) for t in x])
        x = np.ascontiguousarray(x, dtype=np.float32)
        feats = self.encoder.run([_ENC_OUTPUT], {"x": x})[0]
        b = feats.shape[0]
        slot = self._pick_slot(b)
        loc, probs = slot.run(torch.from_numpy(feats))
        return [loc, probs]

    def _pick_slot(self, b: int) -> _Slot:
        for size in self._slot_sizes:
            if size >= b:
                return self.slots[size]
        raise RuntimeError(
            f"batch {b} exceeds largest decoder slot {self._slot_sizes[-1]}")


def maybe_install(table_model, cfg: Optional[dict]) -> bool:
    """Swap the ORT session for the torch decoder when the config says so.

    Called from the GPU server after ``TableStructureModel`` built its ORT
    session, which becomes the fallback.  Returns True when the swap
    landed; any failure leaves the ORT session in place and logs why, so
    a broken decoder never takes the pipeline down.
    """
    cfg = cfg or {}
    if not cfg.get("slanet_torch_decoder", False):
        return False
    try:
        session = table_model.engine.table_structure.session.session
        model_path = table_model.engine.table_structure.session.model_path \
            if hasattr(table_model.engine.table_structure.session, "model_path") \
            else None
        slots = sorted({int(s) for s in cfg.get(
            "slanet_decoder_slots",
            [cfg.get("slanet_batch_max", 32), cfg.get("cslanet_batch_max", 48)])})
        from assay.models import SLANET_PATH, _SLANET_CUDA_EP_OPTS
        providers = [("CUDAExecutionProvider", dict(_SLANET_CUDA_EP_OPTS)),
                     ("CPUExecutionProvider", {})]
        session = TorchDecoderSession(
            model_path=model_path or SLANET_PATH,
            providers=providers,
            slot_sizes=slots,
            max_steps=int(cfg.get("slanet_decoder_max_steps", 500)),
            fallback_session=table_model.engine.table_structure.session.session,
            tf32=bool(cfg.get("slanet_decoder_tf32", False)),
        )
        table_model.engine.table_structure.session.session = session
        return True
    except Exception as exc:
        _log.warning("torch decoder install failed (%s); staying on ORT", exc)
        return False
