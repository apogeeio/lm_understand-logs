import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def connection():
    path=ROOT/"configs/local.json"
    data=json.loads(path.read_text()) if path.exists() else {}
    host=os.environ.get("WSP_SSH_HOST",data.get("ssh_host"))
    remote=os.environ.get("WSP_REMOTE_ROOT",data.get("remote_root"))
    if not host or not remote:raise RuntimeError("Configure configs/local.json or WSP_SSH_HOST/WSP_REMOTE_ROOT")
    return host,remote.rstrip("/")
