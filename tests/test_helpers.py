"""Unit tests for src/utils/helpers.py."""
from __future__ import annotations

import pathlib
import tempfile
import unittest

from src.utils.helpers import (
    ensure_dir,
    format_size,
    get_file_info,
    get_timestamp,
    load_yaml_config,
)


class TestGetTimestamp(unittest.TestCase):

    def test_returns_string(self):
        ts = get_timestamp()
        self.assertIsInstance(ts, str)

    def test_no_path_separators(self):
        ts = get_timestamp()
        self.assertNotIn("/", ts)
        self.assertNotIn("\\", ts)

    def test_length_reasonable(self):
        ts = get_timestamp()
        self.assertGreater(len(ts), 10)


class TestFormatSize(unittest.TestCase):

    def test_bytes(self):
        self.assertIn("B", format_size(500))

    def test_kilobytes(self):
        result = format_size(2048)
        self.assertIn("KB", result)

    def test_megabytes(self):
        result = format_size(5 * 1024 * 1024)
        self.assertIn("MB", result)

    def test_gigabytes(self):
        result = format_size(2 * 1024 ** 3)
        self.assertIn("GB", result)

    def test_zero(self):
        result = format_size(0)
        self.assertIn("B", result)


class TestEnsureDir(unittest.TestCase):

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = pathlib.Path(tmp) / "a" / "b" / "c"
            self.assertFalse(new_dir.exists())
            result = ensure_dir(new_dir)
            self.assertTrue(new_dir.exists())
            self.assertEqual(result, new_dir)

    def test_existing_dir_no_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            ensure_dir(tmp)   # must not raise

    def test_accepts_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / "strpath")
            ensure_dir(path)
            self.assertTrue(pathlib.Path(path).exists())


class TestGetFileInfo(unittest.TestCase):

    def test_nonexistent_returns_none(self):
        result = get_file_info("/nonexistent/path/file.txt")
        self.assertIsNone(result)

    def test_existing_file_returns_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            path = f.name
        info = get_file_info(path)
        self.assertIsNotNone(info)
        self.assertIn("name", info)
        self.assertIn("size", info)
        self.assertIn("size_human", info)
        self.assertEqual(info["size"], 5)
        pathlib.Path(path).unlink(missing_ok=True)


class TestLoadYamlConfig(unittest.TestCase):

    def test_nonexistent_file_returns_empty(self):
        result = load_yaml_config("/no/such/file.yaml")
        self.assertEqual(result, {})

    def test_valid_yaml_returns_dict(self):
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("app:\n  name: Test\n  version: 1.0\n")
            path = f.name
        result = load_yaml_config(path)
        self.assertEqual(result["app"]["name"], "Test")
        self.assertEqual(result["app"]["version"], 1.0)
        pathlib.Path(path).unlink(missing_ok=True)

    def test_invalid_yaml_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("key: [\n")   # broken YAML
            path = f.name
        result = load_yaml_config(path)
        self.assertEqual(result, {})
        pathlib.Path(path).unlink(missing_ok=True)

    def test_empty_yaml_returns_empty(self):
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            path = f.name
        result = load_yaml_config(path)
        self.assertEqual(result, {})
        pathlib.Path(path).unlink(missing_ok=True)

    def test_default_yaml_structure(self):
        """Smoke-test: actual configs/default.yaml has the expected top-level keys."""
        import pathlib
        config_path = pathlib.Path(__file__).parent.parent / "configs" / "default.yaml"
        if not config_path.exists():
            self.skipTest("default.yaml not found")
        cfg = load_yaml_config(config_path)
        for key in ("app", "model", "detection", "face_recognition", "gesture", "device"):
            with self.subTest(key=key):
                self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main()
