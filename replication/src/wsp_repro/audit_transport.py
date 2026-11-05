"""Read saved wire payloads/Responses output kinds without exposing their content or keys."""
from collections import Counter
import json
import time
from .run_math import ROOT

def main():
    types=Counter();checked=Counter();violations=[];native=[];ds_returns=0
    runs=ROOT/"replication/runs"
    for path in [*runs.glob("**/call.json"),*runs.glob("**/call-failure.json")]:
        meta=json.loads(path.read_text());transport=meta.get("transport")
        request=path.with_name("request.json")
        if not request.exists():continue
        payload=json.loads(request.read_text());checked[transport]+=1
        fields=[]
        if transport in ("openai_api","coding_plan_responses_bridge"):
            if payload.get("tools")!=[]:fields.append("tools not empty")
            if payload.get("store") is not False:fields.append("store not false")
            if payload.get("instructions")!="Answer the user's request.":fields.append("unexpected instructions")
            if {"previous_response_id","conversation"}&payload.keys():fields.append("implicit state identifier")
            raw=path.with_name('provider-response.json')
            output=json.loads(raw.read_text()).get('output',[]) if raw.exists() else meta.get('continuation_items',[])
            for item in output:
                kind=item.get("type","missing");types[kind]+=1
                if kind not in ("message","reasoning","compaction"):
                    native.append({"call":str(path.relative_to(ROOT)),"item_type":kind})
        elif transport=="deepseek_api":
            if payload.get("tools"):fields.append("native tools present")
            if payload.get("reasoning_effort")!="max":fields.append("effort changed")
            messages=payload.get("messages",[])
            if not messages or messages[0]!={"role":"system","content":"Answer the user's request."}:fields.append("unexpected system envelope")
            if any("reasoning_content" in m for m in messages):fields.append("hidden reasoning replayed")
            raw=path.with_name('provider-response.json')
            if raw.exists():
                ds_returns+=1
                for choice in json.loads(raw.read_text()).get('choices',[]):
                    message=choice.get('message',{})
                    if message.get('tool_calls') or message.get('function_call'):
                        native.append({'call':str(path.relative_to(ROOT)),'item_type':'DeepSeek returned native tool request'})
        if fields:violations.append({"call":str(path.relative_to(ROOT)),"fields":fields})
    report={"checked_calls":dict(checked),"responses_output_kinds":dict(types),
        "request_violations":violations,"unexpected_responses_items":native,
        "deepseek_returned_tool_fields":"not separately retained in old call metadata; requests contain no native tools",
        "deepseek_raw_returns_checked":ds_returns,
        "content_printed":False,"updated":time.time()}
    (ROOT/"replication/artifacts/transport-audit.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=="__main__":main()
