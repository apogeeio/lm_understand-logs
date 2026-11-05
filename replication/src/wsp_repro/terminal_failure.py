"""Keep terminal native model outputs across search-controller restarts."""
import hashlib
import json
import time


def terminal_native_output(error):
    return error.get("type")=="IncompleteModelOutput" or (
        error.get("type")=="RuntimeError" and error.get("message","").startswith(
            ("DeepSeek response ended:","OpenAI response incomplete:")))


def record_terminal(folder,error):
    marker=folder/"terminal-failure.json"
    if marker.exists():return
    evidence=[p for pattern in ("search-error.json","turn-*/call-failure.json")
        for p in folder.glob(pattern)]
    failures=[p for p in evidence if p.name=="call-failure.json"]
    usages=[json.loads(p.read_text()).get("usage",{}) for p in failures]
    usage_available=any(isinstance(u,dict)
        and any(k in u for k in ("input_tokens","prompt_tokens"))
        and any(k in u for k in ("output_tokens","completion_tokens")) for u in usages)
    record={"job":folder.name,"status":"model_failure","reason":error["message"],
        "error_type":error["type"],"replacement_accepted":False,
        "resample_on_restart":False,"score":None,
        "evidence_files":{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in evidence},
        "native_failure_metadata": [str(p.relative_to(folder)) for p in failures],
        "original_usage_available":usage_available,
        "raw_reply_usage_evidence":("See retained native failure metadata and provider response"
            if failures else "Legacy adapter did not retain terminal raw reply, returned model or usage; evidence gap"),
        "updated":time.time()}
    marker.write_text(json.dumps(record,indent=2)+"\n")


def retained_terminal(folder):
    if (folder/"terminal-failure.json").exists():return True
    path=folder/"search-error.json"
    if path.exists():
        error=json.loads(path.read_text())
        if terminal_native_output(error):
            record_terminal(folder,error)
            return True
    return False
