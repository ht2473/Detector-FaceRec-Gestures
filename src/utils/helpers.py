"""Utility functions"""
import pathlib
import yaml
from datetime import datetime
from typing import Optional, Dict

def get_timestamp() -> str:
    """Return current timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str | pathlib.Path) -> None:
    """Create directory if not exists"""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def format_size(bytes_size: int) -> str:
    """Format size to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def get_file_info(filepath: str) -> Optional[Dict]:
    """Return file information"""
    path = pathlib.Path(filepath)
    if path.exists():
        return {
            'name': path.name,
            'size': path.stat().st_size,
            'created': datetime.fromtimestamp(path.stat().st_ctime),
            'modified': datetime.fromtimestamp(path.stat().st_mtime)
        }
    return None

def load_yaml_config(filepath: str) -> dict:
    """Safely load YAML configuration file"""
    path = pathlib.Path(filepath)
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error loading YAML {filepath}: {e}")
    return {}