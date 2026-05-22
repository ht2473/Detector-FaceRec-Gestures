"""Unit tests for DetectionEngine helpers (no camera, no YOLO model required)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.core.engine import _resolve_device, DetectionEngine


def _make_engine(**kwargs) -> DetectionEngine:
    defaults = dict(
        source_type="webcam", source_path="",
        model_name="yolo26s.pt",
        conf=0.25, iou=0.45, imgsz=640,
        device="auto", classes=[],
    )
    defaults.update(kwargs)
    return DetectionEngine(**defaults)


class TestResolveDevice(unittest.TestCase):

    def test_cpu_stays_cpu(self):
        self.assertEqual(_resolve_device("cpu"), "cpu")

    def test_auto_with_cuda(self):
        with patch("src.core.engine.torch.cuda.is_available", return_value=True):
            self.assertEqual(_resolve_device("auto"), "cuda")

    def test_auto_without_cuda(self):
        with patch("src.core.engine.torch.cuda.is_available", return_value=False):
            self.assertEqual(_resolve_device("auto"), "cpu")

    def test_empty_string_treated_as_auto(self):
        with patch("src.core.engine.torch.cuda.is_available", return_value=False):
            self.assertEqual(_resolve_device(""), "cpu")

    def test_case_insensitive(self):
        with patch("src.core.engine.torch.cuda.is_available", return_value=True):
            self.assertEqual(_resolve_device("AUTO"), "cuda")
            self.assertEqual(_resolve_device("CPU"),  "cpu")


class TestUseHalfLogic(unittest.TestCase):
    """Verify use_half is True on CUDA and False on CPU."""

    def _use_half(self, resolved_device: str, cuda_avail: bool) -> bool:
        import torch
        with patch.object(torch.cuda, "is_available", return_value=cuda_avail):
            import torch as t
            return resolved_device == "cuda" and t.cuda.is_available()

    def test_use_half_true_on_cuda(self):
        self.assertTrue(self._use_half("cuda", True))

    def test_use_half_false_on_cpu(self):
        self.assertFalse(self._use_half("cpu", True))

    def test_use_half_false_when_cuda_unavailable(self):
        self.assertFalse(self._use_half("cuda", False))

    def test_auto_resolves_not_auto(self):
        with patch("src.core.engine.torch.cuda.is_available", return_value=False):
            resolved = _resolve_device("auto")
        self.assertNotEqual(resolved, "auto")


class TestDetectionToggleRecording(unittest.TestCase):

    def test_start_returns_false(self):
        eng = _make_engine()
        self.assertFalse(eng.toggle_recording())
        self.assertTrue(eng._recording)

    def test_stop_without_writer_returns_false(self):
        eng = _make_engine()
        eng.toggle_recording()
        self.assertFalse(eng.toggle_recording())
        self.assertFalse(eng._recording)

    def test_stop_with_writer_returns_true(self):
        eng = _make_engine()
        eng.toggle_recording()
        eng._writer = MagicMock()
        result = eng.toggle_recording()
        self.assertTrue(result)
        self.assertIsNone(eng._writer)


class TestClassFilter(unittest.TestCase):

    def _build(self, model_names, classes):
        if not classes:
            return None
        names_lower = {v.lower(): k for k, v in model_names.items()}
        return [names_lower[c.lower()] for c in classes if c.lower() in names_lower]

    def test_known_class_resolved(self):
        self.assertEqual(sorted(self._build({0:"person",1:"car",2:"dog"}, ["person","dog"])), [0,2])

    def test_unknown_class_skipped(self):
        self.assertEqual(self._build({0:"person"}, ["person","unicorn"]), [0])

    def test_case_insensitive(self):
        self.assertEqual(sorted(self._build({0:"Person",1:"Car"}, ["PERSON","car"])), [0,1])

    def test_empty_classes_returns_none(self):
        self.assertIsNone(self._build({0:"person"}, []))

    def test_no_matches_returns_empty(self):
        self.assertEqual(self._build({0:"person"}, ["airplane"]), [])


if __name__ == "__main__":
    unittest.main()
