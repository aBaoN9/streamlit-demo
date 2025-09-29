# src/paths.py
from pathlib import Path

# Project root = thư mục chứa các folder: app/, data/, src/, notebooks/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def artifact_path(name: str) -> Path:
    return PROJECT_ROOT / "app" / "artifacts" / name

def data_path(name: str) -> Path:
    return PROJECT_ROOT / "data" / name
