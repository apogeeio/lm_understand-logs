"""Load the updated worker after its current practice pass, preserving receipts."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from .launch_math import launch,ROOT

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--old-pid",type=int,required=True);args=ap.parse_args()
    state=ROOT/"replication/artifacts/math-gpu-state.json";record=state.parent/f"gpu-upgrade-{args.old_pid}.json"
    if record.exists():return
    while True:
        s=json.loads(state.read_text())
        if s["pid"]!=args.old_pid:
            record.write_text(json.dumps({"status":"already_replaced","observed":s,"updated":time.time()}));return
        proc=Path(f"/proc/{args.old_pid}/cmdline")
        if not proc.exists():break
        cmd=proc.read_bytes().split(b"\0")
        if b"wsp_repro.math_gpu" not in cmd:raise RuntimeError("PID no longer belongs to this worker")
        # Never cancel an editor's practice pass. An incomplete low-priority evaluation
        # chunk has no saved score and can be recovered with the same UID and seed.
        if s["phase"]=="ready" or (s["phase"]=="running" and s.get("request","").startswith("eval-")):
            record.write_text(json.dumps({"status":"restarting","interrupted_state":s,
                "reason":"load final grader and official IFEval adapter; recover incomplete heldout receipt",
                "updated":time.time()}))
            os.kill(args.old_pid,signal.SIGTERM)
            for _ in range(30):
                if not proc.exists():break
                time.sleep(1)
            if proc.exists():raise RuntimeError("Old GPU controller did not exit")
            break
        time.sleep(5)
    launch("math-gpu","wsp_repro.math_gpu",gpu=True)
    from .audit_math_grades import main as audit
    audit()
    r=json.loads(record.read_text()) if record.exists() else {};r.update(status="restarted",updated=time.time())
    record.write_text(json.dumps(r,indent=2))

if __name__=="__main__":main()
