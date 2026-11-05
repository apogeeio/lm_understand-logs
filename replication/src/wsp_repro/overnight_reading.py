"""Durable serial queues for the original reading experiments."""
import argparse
import json
import time
from .run_reading import generate,ROOT
from .spec import make_spec

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["astra","deepseek"],required=True)
    args=ap.parse_args();state=ROOT/"replication/artifacts"/("queue-reading-"+args.model+".json")
    jobs=[j for j in make_spec()["jobs"] if j["model"]==args.model and j["experiment"] in (1,2,3)]
    for job in jobs:
        while True:
            state.write_text(json.dumps({"job":job["id"],"phase":"running","updated":time.time()}))
            try:generate(job);break
            except Exception as e:
                error={"job":job["id"],"phase":"error","type":type(e).__name__,"message":str(e),"updated":time.time()}
                state.write_text(json.dumps(error));print(json.dumps(error),flush=True)
                if type(e).__name__ in {"RateLimited","TransientProviderError","QuotaExhausted","URLError","TimeoutError"}:
                    time.sleep(1800);continue
                folder=ROOT/"replication/runs"/job["id"];folder.mkdir(parents=True,exist_ok=True)
                (folder/"reading-error.json").write_text(json.dumps(error));break
    state.write_text(json.dumps({"phase":"all_attempts_finished","updated":time.time()}))

if __name__=="__main__":main()
