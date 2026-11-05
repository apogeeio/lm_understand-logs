import json
import pytest
from wsp_repro import run_go,run_math
from wsp_repro.providers import IncompleteModelOutput
from wsp_repro.terminal_failure import retained_terminal


@pytest.mark.parametrize("controller",[run_go,run_math])
def test_legacy_terminal_search_never_calls_model_or_restarts_kernel(tmp_path,monkeypatch,controller):
    monkeypatch.setattr(controller,"ROOT",tmp_path)
    folder=tmp_path/"replication/runs/failed-job";folder.mkdir(parents=True)
    error={"type":"RuntimeError","message":"DeepSeek response ended: length","updated":1}
    (folder/"search-error.json").write_text(json.dumps(error))
    def forbidden(*args,**kwargs):raise AssertionError("Failed search must not be resampled")
    monkeypatch.setattr(controller,"call",forbidden)
    controller.run({"id":"failed-job"})
    marker=folder/"terminal-failure.json";saved=marker.read_bytes()
    controller.run({"id":"failed-job"})
    assert marker.read_bytes()==saved
    assert json.loads(saved)["score"] is None
    assert json.loads(saved)["original_usage_available"] is False
    assert "evidence gap" in json.loads(saved)["raw_reply_usage_evidence"]
    assert json.loads((folder/"search-error.json").read_text())==error


def test_native_terminal_failure_advances_to_next_independent_go_job(tmp_path,monkeypatch):
    monkeypatch.setattr(run_go,"ROOT",tmp_path)
    jobs=[{"id":name,"experiment":6,"model":"deepseek"} for name in ("first","next")]
    monkeypatch.setattr(run_go,"make_spec",lambda:{"jobs":jobs})
    monkeypatch.setattr("sys.argv",["run_go","--model","deepseek"])
    seen=[]
    def simulate(job):
        seen.append(job["id"])
        if job["id"]=="first":raise IncompleteModelOutput("DeepSeek response ended: length")
    monkeypatch.setattr(run_go,"run",simulate)
    run_go.main()
    assert seen==["first","next"]
    assert json.loads((tmp_path/"replication/runs/first/terminal-failure.json").read_text())["replacement_accepted"] is False


def test_infrastructure_error_is_not_a_model_failure(tmp_path):
    (tmp_path/"search-error.json").write_text(json.dumps({"type":"RuntimeError",
        "message":"CUDA worker unavailable"}))
    assert not retained_terminal(tmp_path)
    assert not (tmp_path/"terminal-failure.json").exists()
