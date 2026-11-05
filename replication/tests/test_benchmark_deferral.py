import json
from wsp_repro import math_evaluate
from wsp_repro.progress import snapshot


def test_gpqa_deferral_preserves_eight_benchmark_completion_and_can_resume(tmp_path,monkeypatch):
    monkeypatch.setattr(math_evaluate,"ROOT",tmp_path)
    queue=tmp_path/"queue";queue.mkdir();monkeypatch.setattr(math_evaluate,"QUEUE",queue)
    config=tmp_path/"replication/configs/benchmark-deferrals.json";config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"gpqa":{"status":"deferred","requested_samples_per_problem":4}}))
    job={"id":"e8-astra-p10-r0"};folder=tmp_path/"replication/runs"/job["id"];folder.mkdir(parents=True)
    (folder/"frozen.json").write_text(json.dumps({"edited":{}}))
    manifest={name:{"count":1,"sha256":"fixed-data"} for name in math_evaluate.SAMPLES}
    for name,n in math_evaluate.SAMPLES.items():
        if name=="gpqa":continue
        result={"data_sha256":"fixed-data","start":0,"end":1,"generations":n,
            "ok":n,"cut":0,"tokens":100,"seconds":1}
        (queue/f"eval-{job['id']}-{name}-0000-0001.response.json").write_text(json.dumps({"ok":True,"result":result}))
    math_evaluate.advance(job,manifest)
    path=folder/"heldout/summary.json";report=json.loads(path.read_text())
    assert report["current_phase_benchmarks_complete"]
    assert not report["all_requested_benchmarks_complete"]
    assert len(report["sets"])==7 and report["deferred_benchmarks"]["gpqa"]["status"]=="deferred"
    assert not list(queue.glob("*gpqa*"))  # Available data must not override the configured deferral.
    case=next(c for c in snapshot(tmp_path)["cases"] if c["id"]==job["id"])
    assert case["state"]=="scored_with_deferred_benchmarks"
    config.write_text("{}")  # A later explicit scope update resumes the same frozen checkpoint.
    math_evaluate.advance(job,manifest);report=json.loads(path.read_text())
    assert not report["current_phase_benchmarks_complete"] and not report["all_requested_benchmarks_complete"]
    assert report["sets"]["gpqa"]["samples"]==4
    assert len(list(queue.glob("*gpqa*.request.json")))==1


def test_deferred_math_search_does_not_start_model(tmp_path,monkeypatch):
    from wsp_repro import run_math
    monkeypatch.setattr(run_math,"ROOT",tmp_path)
    config=tmp_path/"replication/configs/search-deferrals.json";config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"math_pass_budgets":[25,50]}))
    def forbidden(*a,**k):raise AssertionError("Deferred search must not call model")
    monkeypatch.setattr(run_math,"call",forbidden)
    run_math.run({"id":"e8-astra-p25-r0","passes":25})
    assert not (tmp_path/"replication/runs/e8-astra-p25-r0").exists()
    case=next(c for c in snapshot(tmp_path)["cases"] if c["id"]=="e8-astra-p25-r0")
    assert case["state"]=="deferred"


def test_partial_math500_retained_without_new_chunks(tmp_path,monkeypatch):
    monkeypatch.setattr(math_evaluate,"ROOT",tmp_path)
    queue=tmp_path/"queue";queue.mkdir();monkeypatch.setattr(math_evaluate,"QUEUE",queue)
    config=tmp_path/"replication/configs/benchmark-deferrals.json";config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"math500":{"status":"deferred"}}))
    job={"id":"eval-base"}
    result={"data_sha256":"fixed","start":0,"end":4,"generations":16,
        "ok":8,"cut":2,"tokens":100,"seconds":1}
    (queue/"eval-eval-base-math500-0000-0004.response.json").write_text(json.dumps({"ok":True,"result":result}))
    math_evaluate.advance(job,{"math500":{"count":500,"sha256":"fixed"},
        "amc23":{"count":40,"sha256":"amc"},"gsm8k":{"count":1319,"sha256":"gsm"}})
    report=json.loads((tmp_path/"replication/runs/eval-base/heldout/summary.json").read_text())
    assert report["sets"]["math500"]["completed_problems"]==4
    assert report["sets"]["math500"]["deferred"]
    assert report["sets"]["math500"]["accuracy"]==.5
    assert not list(queue.glob("*math500*.request.json"))
    assert len(list(queue.glob("*amc23*.request.json")))==1
    assert not list(queue.glob("*gsm8k*.request.json"))


def test_whole_math_pause_blocks_search_eval_and_launcher(tmp_path, monkeypatch):
    from wsp_repro import run_math, launch_math
    config=tmp_path/"replication/configs/math-pause.json";config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"paused":True}))
    for module in [run_math,math_evaluate,launch_math]:monkeypatch.setattr(module,"ROOT",tmp_path)
    def forbidden(*a,**k):raise AssertionError("Paused work must not start")
    monkeypatch.setattr(run_math,"call",forbidden)
    monkeypatch.setattr(math_evaluate,"freeze_evaluation",forbidden)
    monkeypatch.setattr(launch_math.subprocess,"Popen",forbidden)
    run_math.run({"id":"e8-astra-p10-r0","passes":10})
    math_evaluate.advance({"id":"eval-base"},{})
    launch_math.launch("math-gpu","wsp_repro.math_gpu",gpu=True)
    cases=[c for c in snapshot(tmp_path)["cases"] if c["experiment"]==8]
    assert len(cases)==18 and all(c["state"]=="paused" for c in cases)
    assert not (tmp_path/"replication/runs").exists()
