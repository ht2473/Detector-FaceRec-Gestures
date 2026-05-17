"""Utility / helper functions."""
from __future__ import annotations

import pathlib
import yaml
from datetime import datetime
from typing import Dict, Optional


def get_timestamp() -> str:
    """Return a filesystem-safe timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str | pathlib.Path) -> pathlib.Path:
    """Create *path* (and all parents) if they don't exist. Returns the Path."""
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_size(byte_count: int) -> str:
    """Convert a byte count to a human-readable string (B → TB)."""
    size = float(byte_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def get_file_info(filepath: str | pathlib.Path) -> Optional[Dict]:
    """Return basic metadata for *filepath*, or ``None`` if it doesn't exist."""
    path = pathlib.Path(filepath)
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "size_human": format_size(stat.st_size),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
    }


def load_yaml_config(filepath: str | pathlib.Path) -> dict:
    """Safely load a YAML config file.  Returns an empty dict on any error."""
    path = pathlib.Path(filepath)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # noqa: BLE001
        # Avoid crashing at import time; caller can check for empty dict
        print(f"[helpers] Failed to load YAML '{path}': {exc}")
        return {}
