# src/history.py
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from src.paths import artifact_path

HIST_PATH = artifact_path("history.json")

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_history() -> List[Dict[str, Any]]:
    if HIST_PATH.exists():
        return json.loads(HIST_PATH.read_text(encoding="utf-8"))
    return []

def append_history(item: Dict[str, Any]) -> None:
    hist = read_history()
    item["ts"] = _now()
    hist.insert(0, item)  # newest first
    HIST_PATH.write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

def clear_history() -> None:
    if HIST_PATH.exists():
        HIST_PATH.unlink()
