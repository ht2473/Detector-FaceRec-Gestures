"""Unit tests for FaceRecognitionEngine logic (no camera, no InsightFace required)."""
from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.core.face_engine import FaceRecognitionEngine


def _make_engine(face_db=None, rec_thresh=0.35):
    eng = FaceRecognitionEngine.__new__(FaceRecognitionEngine)
    eng._face_db   = face_db or {}
    eng._app       = None
    eng._cap       = None
    eng._writer    = None
    eng._recording = False
    eng.rec_thresh = rec_thresh
    return eng


def _unit(dim=512, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestNormalise(unittest.TestCase):

    def test_output_is_unit(self):
        v = _unit(128, seed=1)
        n = FaceRecognitionEngine._normalise(v)
        np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-6)

    def test_known_values(self):
        v = np.array([3.0, 4.0], dtype=np.float32)
        n = FaceRecognitionEngine._normalise(v)
        np.testing.assert_allclose(n, [0.6, 0.8], atol=1e-6)

    def test_zero_vector_no_crash(self):
        v = np.zeros(64, dtype=np.float32)
        n = FaceRecognitionEngine._normalise(v)
        self.assertEqual(n.shape, (64,))


class TestMatchFace(unittest.TestCase):

    def test_empty_db_unknown(self):
        eng = _make_engine()
        name, sim = eng._match_face(_unit())
        self.assertEqual(name, "Unknown")
        self.assertEqual(sim, 0.0)

    def test_exact_match(self):
        vec = _unit(512, seed=42)
        eng = _make_engine({"Alice": vec.copy()}, rec_thresh=0.0)
        name, sim = eng._match_face(vec)
        self.assertEqual(name, "Alice")
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_best_of_two(self):
        alice = _unit(512, seed=1)
        bob   = _unit(512, seed=2)
        eng   = _make_engine({"Alice": alice, "Bob": bob}, rec_thresh=0.0)
        query = alice + 0.01 * np.random.default_rng(99).standard_normal(512)
        query = (query / np.linalg.norm(query)).astype(np.float32)
        name, _ = eng._match_face(query)
        self.assertEqual(name, "Alice")

    def test_similarity_bounded(self):
        vecs = {f"p{i}": _unit(128, seed=i) for i in range(10)}
        eng  = _make_engine(vecs, rec_thresh=0.0)
        _, sim = eng._match_face(_unit(128, seed=99))
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim,  1.0)

    def test_returns_str_and_float(self):
        eng = _make_engine({"Bob": _unit()}, rec_thresh=0.0)
        name, sim = eng._match_face(_unit())
        self.assertIsInstance(name, str)
        self.assertIsInstance(sim, float)


class TestFaceToggleRecording(unittest.TestCase):

    def test_start_false(self):
        eng = _make_engine()
        self.assertFalse(eng.toggle_recording())
        self.assertTrue(eng._recording)

    def test_stop_no_writer_false(self):
        eng = _make_engine()
        eng.toggle_recording()
        self.assertFalse(eng.toggle_recording())
        self.assertFalse(eng._recording)

    def test_stop_with_writer_true(self):
        eng = _make_engine()
        eng.toggle_recording()
        eng._writer = MagicMock()
        self.assertTrue(eng.toggle_recording())
        self.assertIsNone(eng._writer)


if __name__ == "__main__":
    unittest.main()
