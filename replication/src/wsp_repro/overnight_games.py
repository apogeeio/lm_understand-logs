"""One serial subject queue per provider, durable artifacts and bounded-rate retries."""
import argparse
import json
from pathlib import Path
import time
from .spec import make_spec
from .run_games import generate,ROOT
from .dispatch_eval import evaluate_frozen


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=["astra","deepseek"],required=True)
    args=ap.parse_args()
    status=ROOT/"artifacts"/("queue-"+args.model+".json");status.parent.mkdir(exist_ok=True)
    jobs=[j for j in make_spec()["jobs"] if j["experiment"]==5 and j["model"]==args.model]
    for index,job in enumerate(jobs):
        folder=ROOT/"runs"/job["id"]
        while True:
            status.write_text(json.dumps({"status":"running","job":job["id"],"index":index,"total":len(jobs),"updated":time.time()}))
            _,state=generate(job)
            if state in ("frozen","already_frozen"):
                try:evaluate_frozen(job)
                except Exception as e:
                    (folder/"evaluation-error.json").write_text(json.dumps({"error_type":type(e).__name__,"message":str(e)}))
                break
            details=json.loads((folder/"state.json").read_text())
            kind=details.get("error_type")
            if kind in ("RateLimited","TransientProviderError","URLError","TimeoutError"):
                status.write_text(json.dumps({"status":"backoff","job":job["id"],"next_retry":time.time()+1800,"reason":kind}))
                time.sleep(1800);continue
            if kind=="QuotaExhausted":
                status.write_text(json.dumps({"status":"waiting_provider_quota","job":job["id"],"reason":kind,"updated":time.time()}))
                # Do not switch models or spin through all remaining jobs on depleted quota.
                time.sleep(1800);continue
            print(json.dumps({"job":job["id"],"state":state,"details":details}),flush=True)
            break  # invalid generated output is retained; never generate a replacement to improve score
        print(json.dumps({"job":job["id"],"state":"attempt_finished"}),flush=True)
    status.write_text(json.dumps({"status":"all_attempts_finished","total":len(jobs),"updated":time.time()}))


if __name__=="__main__":main()
