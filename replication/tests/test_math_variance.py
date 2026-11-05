import json
import hashlib
import pytest
from wsp_repro.aggregate_math import aggregate
from wsp_repro.audit_records import review_status
from wsp_repro.progress import snapshot


def review(folder,verdict):
    trace=folder/"turn-000/answer.md";trace.parent.mkdir(exist_ok=True)
    trace.write_text("immutable experiment reply")
    (folder/"semantic-review.json").write_text(json.dumps({"verdict":verdict,
        "checked_files":{"turn-000/answer.md":hashlib.sha256(trace.read_bytes()).hexdigest()}}))

def test_independent_search_variance_and_transport_groups(tmp_path):
    for repeat,accuracy in enumerate([.1,.2,.6]):
        p=tmp_path/"replication/runs"/f"e8-astra-p10-r{repeat}"
        (p/"heldout").mkdir(parents=True)
        (p/"frozen.json").write_text(json.dumps({"transport":"coding_plan_responses_bridge","semantic_protocol_audit":"pending"}))
        (p/"heldout/summary.json").write_text(json.dumps({"sets":{"math500":{"complete":True,"accuracy":accuracy,"cut_share":.5}}}))
        review(p,"passed")
    result=aggregate(tmp_path);row=result["rows"][0]
    assert row["n_independent_searches"]==3 and row["complete_three_repeats"]
    assert row["mean_accuracy"]==pytest.approx(.3) and row["run_sample_variance"]==pytest.approx(.07)
    # Moving one run to another transport must not pool it into a three-run variance.
    p=tmp_path/"replication/runs/e8-astra-p10-r2/frozen.json"
    p.write_text(json.dumps({"transport":"openai_api","semantic_protocol_audit":"pending"}))
    rows=aggregate(tmp_path)["rows"]
    assert sorted(r["n_independent_searches"] for r in rows)==[1,2]
    assert not any(r["complete_three_repeats"] for r in rows)


def test_rejected_and_stale_reviews_are_not_accepted_scores(tmp_path):
    runs=tmp_path/"replication/runs"
    for repeat,verdict in enumerate(("passed","protocol_violation","passed")):
        p=runs/f"e8-astra-p10-r{repeat}";(p/"heldout").mkdir(parents=True)
        (p/"frozen.json").write_text(json.dumps({"transport":"coding_plan_responses_bridge",
            "semantic_protocol_audit":"passed"}))
        (p/"heldout/summary.json").write_text(json.dumps({"all_requested_benchmarks_complete":True,
            "sets":{"math500":{"complete":True,"accuracy":.9,"cut_share":0,
                "completed_problems":500,"expected_problems":500}}}))
        review(p,verdict)
    (runs/"e8-astra-p10-r2/turn-000/answer.md").write_text("changed after audit")
    result=aggregate(tmp_path)
    assert result["rows"][0]["n_independent_searches"]==1
    assert result["provisional_rows"][0]["n_independent_searches"]==1
    assert result["excluded_runs"][0]["job"]=="e8-astra-p10-r1"
    assert review_status(runs/"e8-astra-p10-r1")=="protocol_violation"
    assert review_status(runs/"e8-astra-p10-r2")=="pending"
    cases={c["id"]:c for c in snapshot(tmp_path)["cases"]}
    assert cases["e8-astra-p10-r1"]["state"]=="protocol_violation"
    assert cases["e8-astra-p10-r2"]["audit"]=="pending"
