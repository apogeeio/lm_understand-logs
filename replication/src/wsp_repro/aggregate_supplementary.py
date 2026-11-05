"""Audited repeat statistics and a separate high-output-budget game arm."""
import hashlib
import json
from pathlib import Path
from statistics import mean, variance
from .audit_records import valid_review

ROOT = Path(__file__).resolve().parents[3]


def stats(values):
    valid = [float(v) for v in values if v is not None]
    return {"values": values, "attempts": len(values), "n": len(valid),
            "failures": len(values) - len(valid),
            "mean": mean(valid) if valid else None,
            "sample_variance": variance(valid) if len(valid) > 1 else None}


def reading_record(folder, repeat):
    raw = json.loads((folder / "result.json").read_text())
    review = valid_review(folder)
    if repeat:
        assert review and review["verdict"] == "passed", folder.name
        assert review["raw_result_sha256"] == hashlib.sha256((folder / "result.json").read_bytes()).hexdigest()
        metrics = review["metrics"]
        outcome = review["outcome"]
    else:
        assert raw["semantic_label_review"] == "passed", folder.name
        if folder.name.startswith("e3-"):
            assert review and review["verdict"] == "passed", folder.name
        metrics = {k: raw[k] for k in ("correct", "order_correct", "copy_exact_match") if k in raw}
        outcome = "valid"
    final = sorted(folder.glob("turn-*/answer.md"))[-1]
    return {"id": folder.name, "repeat": repeat, "outcome": outcome,
            "metrics": metrics, "raw_metrics": {k: raw[k] for k in metrics},
            "answer_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
            "protocol_audit": "passed",
            "notes": review.get("notes", []) if repeat else []}


def build():
    p = ROOT / "replication"
    cfg = json.loads((p / "configs/supplementary.json").read_text())
    evidence = json.loads((p / "artifacts/supplementary-verification.json").read_text())
    evidence_by_id = {r["id"]: r for r in evidence["rows"]}
    groups = []
    unique = sorted({(j["experiment"], j["model"], j["case"]) for j in cfg["jobs"] if j["experiment"] < 5})
    for exp, model, case in unique:
        runs = []
        for repeat in range(3):
            folder = p / "runs" / f"e{exp}-{model}-{case}-r{repeat}"
            if exp < 4:
                row = reading_record(folder, repeat)
            else:
                a = json.loads((folder / "artifact.json").read_text())
                e = json.loads((folder / "evaluation.json").read_text())
                assert a["sha256"] == e["frozen_sha256"] and a["frozen_before_evaluation"]
                t = e["test"]
                assert t["episodes"] == len(t["records"]) == 1024 and t["seed"] == 1401
                assert t["success"] == sum(r["success"] for r in t["records"]) / 1024
                if repeat:
                    audit = json.loads((folder / "semantic-audit.json").read_text())
                    assert audit["verdict"] == "passed" and audit["artifact_sha256"] == a["sha256"]
                    assert evidence_by_id[folder.name]["frozen_sha256"] == a["sha256"]
                row = {"id": folder.name, "repeat": repeat, "outcome": "valid",
                       "metrics": {"success_rate": t["success"]}, "frozen_sha256": a["sha256"],
                       "episodes": 1024, "seed": 1401, "successes": sum(r["success"] for r in t["records"]),
                       "invalid": t["invalid"], "protocol_audit": "passed"}
            if repeat:
                row["calls"] = evidence_by_id[folder.name]["calls"]
            runs.append(row)
        groups.append({"experiment": exp, "model": model, "case": case, "runs": runs,
                       "metrics": {k: stats([r["metrics"][k] for r in runs]) for k in runs[0]["metrics"]}})
    high = []
    receipts = {r["job"]: r for r in json.loads((p / "artifacts/highbudget-verification.json").read_text())["rows"]}
    for j in cfg["jobs"]:
        if j["experiment"] != 5:
            continue
        folder = p / "runs" / j["id"]
        r = receipts[j["id"]]
        a = json.loads((folder / "artifact.json").read_text())
        audit = json.loads((folder / "semantic-audit.json").read_text())
        assert audit["verdict"] == "passed" and audit["artifact_sha256"] == a["sha256"] == r["sha256"]
        high.append({"id": j["id"], "source_attempt": j["source_attempt"], "game": j["game"],
                     "width": j["width"], "repeat": j["repeat"], "model": j["model"],
                     "max_output_tokens": j["max_output_tokens"], "mean": r["mean"],
                     "episode_standard_error": r["standard_error"], "episodes": r["episodes"],
                     "frozen_sha256": r["sha256"], "protocol_audit": "passed",
                     "calls": evidence_by_id[j["id"]]["calls"]})
    high_groups = []
    for game, width in sorted({(r["game"], r["width"]) for r in high}):
        rows = [r for r in high if r["game"] == game and r["width"] == width]
        high_groups.append({"game": game, "width": width, **stats([r["mean"] for r in rows])})
    usage = {}
    for model in ("astra", "deepseek"):
        calls = [c for r in evidence["rows"] if r["model"] == model for c in r["calls"]]
        usage[model] = {"calls": len(calls), **{k: sum(c[k] for c in calls) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "seconds")}}
    rerun_folder = p / "runs/e3-deepseek-romaji-r1-rerun-v1"
    panel = None
    if (rerun_folder / "semantic-review.json").exists():
        replacement = reading_record(rerun_folder, 1)
        replacement["source_attempt"] = "e3-deepseek-romaji-r1"
        panel_runs = []
        for group in groups:
            if group["experiment"] == 3 and group["model"] == "deepseek":
                panel_runs.extend(replacement if r["id"] == replacement["source_attempt"] else r for r in group["runs"])
        panel = {"generation_attempts": 10, "valid_outputs": 9, "empty_outputs": 1,
                 "replacement": replacement, "runs": panel_runs,
                 "metrics": {k: stats([r["metrics"][k] for r in panel_runs]) for k in replacement["metrics"]},
                 "original_panel_preserved": True}
    return {"deepseek_experiment3_completed_panel": panel, "schema": 1, "date": "2026-11-05", "additional_attempts": 50,
            "additional_valid_outputs": 49, "additional_empty_final_outputs": 1,
            "wall_seconds": evidence["wall_seconds"], "provider_concurrency": cfg["provider_concurrency"],
            "variance_ddof": 1, "variance_unit": "fraction squared for reading/car; score squared for games",
            "reading_and_car": groups, "highbudget_game_runs": high, "highbudget_game_groups": high_groups,
            "supplementary_usage": usage, "billed_cost": None,
            "cost_note": "API usage recorded; no billing statement available. Coding-plan tokens are not converted to API prices.",
            "highbudget_comparison": "Fresh independent generations for six previously failed conditions; kept separate from the original 128000-token arm.",
            "math_status": "paused_incomplete"}


def main():
    result = build()
    out = ROOT / "replication/results/supplementary-summary.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ("additional_attempts", "additional_valid_outputs", "additional_empty_final_outputs", "wall_seconds")}))


if __name__ == "__main__":
    main()
