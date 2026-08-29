"""
Lightweight table structure recognition via SLANet ONNX + PP-LCNet gatekeeper.

MinerU ONNX models (all run via ONNX Runtime, GPU or CPU):
  - slanet-plus.onnx (7.4MB) — table structure recognition
  - PP-LCNet_x1_0_table_cls.onnx (6.5MB) — table/non-table classifier

GPU support requires onnxruntime-gpu + CUDA 12 + cuDNN 9.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# cuDNN path fix for onnxruntime-gpu
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_cudnn_path():
    """Add nvidia-cudnn-cu12 lib to LD_LIBRARY_PATH so onnxruntime-gpu can load it."""
    import glob as _glob
    import site as _site

    _candidates = []
    for _sp in _site.getsitepackages():
        _candidates.append(os.path.join(_sp, "nvidia", "cudnn", "lib"))
    _candidates.extend(
        _glob.glob(os.path.expanduser("~") + "/**/nvidia/cudnn/lib", recursive=True)
    )

    for _pkg in _candidates:
        _so = os.path.join(_pkg, "libcudnn.so.9")
        if os.path.isfile(_so):
            _existing = os.environ.get("LD_LIBRARY_PATH", "")
            if _pkg not in _existing:
                os.environ["LD_LIBRARY_PATH"] = f"{_pkg}:{_existing}".strip(":")
            return

_ensure_cudnn_path()

# ──────────────────────────────────────────────────────────────────────────────
# Model paths
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path("/home/homio/sda/1111/Assay/model")
SLANET_PATH = str(MODEL_DIR / "slanet-plus.onnx")
PPLCNET_PATH = str(MODEL_DIR / "PP-LCNet_x1_0_table_cls.onnx")

# ──────────────────────────────────────────────────────────────────────────────
# Inference parameters
# ──────────────────────────────────────────────────────────────────────────────

GPU_BATCH_SIZE = 8
PPLCNET_INPUT_SIZE = 224


# ──────────────────────────────────────────────────────────────────────────────
# PP-LCNet table classifier (Step 4: gatekeeper)
# ──────────────────────────────────────────────────────────────────────────────


class TableClassifier:
    """
    PP-LCNet ONNX classifier: table vs non-table.

    Input:  PIL Image (any size, will be resized to 224x224)
    Output: float in [0, 1] — probability that the image IS a table.
    """

    def __init__(self, model_path: str = PPLCNET_PATH, use_gpu: bool = True):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # suppress warnings

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        _actual = self.session.get_providers()[0]
        _log.info(f"PP-LCNet loaded ({_actual})")

    def predict(self, image: Image.Image) -> float:
        """Return P(table | image)."""
        img = image.convert("RGB").resize(
            (PPLCNET_INPUT_SIZE, PPLCNET_INPUT_SIZE), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: arr})
        logits = outputs[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        return float(probs[1])

    def is_table(self, image: Image.Image, threshold: float = 0.5) -> bool:
        return self.predict(image) >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# SLANet table structure recognizer (Step 5: structure parsing)
# ──────────────────────────────────────────────────────────────────────────────


class TableStructureModel:
    """
    SLANet ONNX table structure recognition.

    Input:  List[PIL.Image] — table crop images
    Output: cell_bboxes (Nx8 per image, crop-local coords),
            logic_points (Nx4: [start_row, end_row, start_col, end_col])
    """

    def __init__(self, model_path: str = SLANET_PATH, use_gpu: bool = True):
        from rapid_table import RapidTable, RapidTableInput, ModelType, EngineType

        engine_cfg = {}
        if use_gpu:
            # rapid_table uses ONNXRUNTIME engine; configure GPU providers
            engine_cfg = {
                "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [{"device_id": 0}],
            }

        cfg = RapidTableInput(
            model_type=ModelType.SLANETPLUS,
            model_dir_or_path=model_path,
            engine_type=EngineType.ONNXRUNTIME,
            use_ocr=False,
            engine_cfg=engine_cfg,
        )
        self.engine = RapidTable(cfg)

        # Check what provider was actually used
        try:
            session = self.engine.table_structure.session.session
            _actual = session.get_providers()[0]
        except Exception:
            _actual = "unknown"
        _log.info(f"SLANet loaded ({_actual})")

    def __call__(
        self, images: List[Image.Image]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if not images:
            return [], []
        result = self.engine(images)
        return result.cell_bboxes, result.logic_points


# ──────────────────────────────────────────────────────────────────────────────
# Unified TableModel (replaces both TATR models)
# ──────────────────────────────────────────────────────────────────────────────


class TableModel:
    """
    All-in-one table processing: PP-LCNet gatekeeper + SLANet structure.

    Args:
        use_gpu: If True (default), run ONNX models on CUDA.
                 Falls back to CPU if CUDA unavailable.
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self.classifier = TableClassifier(use_gpu=use_gpu)
        self.structure = TableStructureModel(use_gpu=use_gpu)

    def process_crops(self, crop_specs: List[Dict]) -> List[Dict]:
        """
        Run SLANet structure recognition on pre-cropped table images.

        Each spec: {'image': PIL.Image, 'offset': (crop_x0, crop_y0)}

        Returns one dict per crop:
            {'cell_bboxes':  np.ndarray (N,8), 'logic_points': np.ndarray (N,4),
             'offset': (cx, cy), 'img_size': (w, h)}
        """
        if not crop_specs:
            return []

        images = [s['image'] for s in crop_specs]
        cell_bboxes_list, logic_points_list = self.structure(images)

        results = []
        for i, spec in enumerate(crop_specs):
            results.append({
                'cell_bboxes': (
                    cell_bboxes_list[i] if i < len(cell_bboxes_list)
                    else np.array([])
                ),
                'logic_points': (
                    logic_points_list[i] if i < len(logic_points_list)
                    else np.array([])
                ),
                'offset': spec['offset'],
                'img_size': images[i].size,
            })
        return results

    def classify(self, image: Image.Image) -> float:
        return self.classifier.predict(image)

    def is_table(self, image: Image.Image, threshold: float = 0.5) -> bool:
        return self.classifier.is_table(image, threshold)
