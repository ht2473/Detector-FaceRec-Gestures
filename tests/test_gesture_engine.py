"""Unit tests for GestureRecognitionEngine logic (no camera, no MediaPipe required)."""
from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock

from src.core.gesture_engine import _GESTURE_MAP, GestureRecognitionEngine


def _lm(x=0.0, y=0.0, z=0.0):
    pt = MagicMock(); pt.x = x; pt.y = y; pt.z = z
    return pt


def _landmarks(thumb, index, middle, ring, pinky, label="Right"):
    lms = [_lm() for _ in range(21)]
    # Thumb: Right extended = tip.x < mcp.x
    if label == "Right":
        lms[3].x = 0.5; lms[4].x = 0.3 if thumb else 0.7
    else:
        lms[3].x = 0.5; lms[4].x = 0.7 if thumb else 0.3
    # 4 fingers: extended = tip.y < pip.y
    for ext, tip, pip in zip([index,middle,ring,pinky],[8,12,16,20],[6,10,14,18]):
        lms[pip].y = 0.5; lms[tip].y = 0.3 if ext else 0.7
    return lms


def _make_engine():
    eng = GestureRecognitionEngine.__new__(GestureRecognitionEngine)
    eng._recording   = False
    eng._writer      = None
    eng._record_lock = threading.Lock()
    return eng


class TestFingersUp(unittest.TestCase):

    def _f(self, t, i, m, r, p, label="Right"):
        return GestureRecognitionEngine._fingers_up(_landmarks(t,i,m,r,p,label), label)

    def test_fist(self):           self.assertEqual(self._f(0,0,0,0,0), (0,0,0,0,0))
    def test_open_palm(self):      self.assertEqual(self._f(1,1,1,1,1), (1,1,1,1,1))
    def test_pointing(self):       self.assertEqual(self._f(0,1,0,0,0), (0,1,0,0,0))
    def test_victory(self):        self.assertEqual(self._f(0,1,1,0,0), (0,1,1,0,0))
    def test_thumbs_up(self):      self.assertEqual(self._f(1,0,0,0,0), (1,0,0,0,0))
    def test_i_love_you(self):     self.assertEqual(self._f(1,1,0,0,1), (1,1,0,0,1))
    def test_rock_on(self):        self.assertEqual(self._f(0,1,0,0,1), (0,1,0,0,1))
    def test_call_me(self):        self.assertEqual(self._f(1,0,0,0,1), (1,0,0,0,1))
    def test_pinky(self):          self.assertEqual(self._f(0,0,0,0,1), (0,0,0,0,1))
    def test_thumbs_up_left(self): self.assertEqual(self._f(1,0,0,0,0,"Left"), (1,0,0,0,0))
    def test_fist_left(self):      self.assertEqual(self._f(0,0,0,0,0,"Left"), (0,0,0,0,0))

    def test_returns_tuple_of_5_bits(self):
        r = self._f(1,1,1,1,1)
        self.assertIsInstance(r, tuple)
        self.assertEqual(len(r), 5)
        for v in r: self.assertIn(v, (0,1))


class TestGestureMap(unittest.TestCase):

    def _g(self, *f): return _GESTURE_MAP.get(f, "Unknown")

    def test_fist(self):      self.assertEqual(self._g(0,0,0,0,0), "Fist")
    def test_open_palm(self): self.assertEqual(self._g(1,1,1,1,1), "Open Palm")
    def test_pointing(self):  self.assertEqual(self._g(0,1,0,0,0), "Pointing")
    def test_victory(self):   self.assertEqual(self._g(0,1,1,0,0), "Victory / Peace")
    def test_thumbs_up(self): self.assertEqual(self._g(1,0,0,0,0), "Thumbs Up")
    def test_i_love_you(self):self.assertEqual(self._g(1,1,0,0,1), "I Love You")
    def test_rock_on(self):   self.assertEqual(self._g(0,1,0,0,1), "Rock On")
    def test_call_me(self):   self.assertEqual(self._g(1,0,0,0,1), "Call Me")
    def test_unknown_combo(self): self.assertEqual(self._g(0,0,1,0,1), "Unknown")

    def test_all_values_are_nonempty_strings(self):
        for k, v in _GESTURE_MAP.items():
            with self.subTest(key=k):
                self.assertIsInstance(v, str); self.assertTrue(len(v) > 0)


class TestGestureToggleRecording(unittest.TestCase):

    def test_start_returns_false(self):
        eng = _make_engine()
        self.assertFalse(eng.toggle_recording())
        self.assertTrue(eng._recording)

    def test_stop_no_writer_returns_false(self):
        eng = _make_engine()
        eng.toggle_recording()
        self.assertFalse(eng.toggle_recording())
        self.assertFalse(eng._recording)

    def test_stop_with_writer_returns_true(self):
        eng = _make_engine()
        eng.toggle_recording()
        eng._writer = MagicMock()
        self.assertTrue(eng.toggle_recording())
        eng._writer  # now None
        self.assertIsNone(eng._writer)


if __name__ == "__main__":
    unittest.main()
