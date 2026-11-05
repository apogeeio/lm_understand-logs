"""Persist execution enablement independently of process lifetime."""
import json


def math_paused(root):
    path = root / "replication/configs/math-pause.json"
    return path.exists() and json.loads(path.read_text()).get("paused", False)
