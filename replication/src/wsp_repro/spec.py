"""Frozen run matrix. Repetitions are independent model generations, not eval reruns."""
import json
from pathlib import Path

UPSTREAM = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
MODELS = {
    "astra": {"model": "gpt-6-astra", "transport": "coding_plan_responses_bridge"},
    "deepseek": {"model": "deepseek-flash", "expected_version": "DeepSeek-V4.1-Flash-1017",
                 "transport": "deepseek_api", "effort": "max"},
}
GAMES = ("breakout", "seaquest", "asterix", "snake")
WIDTHS = (32, 16, 8, 4)


def parameter_count(width):
    return 600 * width + width + width * width + width + width * 6 + 6


def make_spec():
    jobs = []
    for model in MODELS:
        for game in GAMES:
            for width in WIDTHS:
                for repeat in range(3):
                    jobs.append(dict(id=f"e5-{model}-{game}-w{width}-r{repeat}", experiment=5,
                        model=model, game=game, kind="network", width=width, repeat=repeat,
                        effort="xhigh" if model=="astra" else "max", episodes=1000, max_steps=10000,
                        sticky_probability=0.1, parameters=parameter_count(width)))
            for repeat in range(3):
                jobs.append(dict(id=f"e5-{model}-{game}-python-r{repeat}", experiment=5, model=model,
                    game=game, kind="python", repeat=repeat, max_code_lines=40,
                    effort="xhigh" if model=="astra" else "max", episodes=1000, max_steps=10000,
                    sticky_probability=0.1))
        for passes in (10, 25, 50):
            for repeat in range(3):
                jobs.append(dict(id=f"e8-{model}-p{passes}-r{repeat}", experiment=8, model=model,
                    passes=passes, repeat=repeat, effort="max", max_python_runs=300,
                    target_model="Qwen/Qwen3.5-0.8B",
                    target_revision="2fc06364715b967f1860aea9cf38778875588b17",
                    eval_protocol="think35", eval_max_new_tokens=32768,
                    benchmark_samples={"math500":4,"gsm8k":2,"ifeval":2,"amc23":16,
                        "minerva":4,"olympiad":4,"mmlupro1k":1,"gpqa":4}))
        cases = {1:("breakout","pong","pacman","freeway"), 2:("circle","square","triangle"),
                 3:("python","romaji","english"), 4:("oneshot_repair",),
                 6:("budget_0","budget_10","budget_25","budget_50")}
        for experiment, variants in cases.items():
            for case in variants:
                jobs.append(dict(id=f"e{experiment}-{model}-{case}-r0",experiment=experiment,
                    model=model,case=case,repeat=0,
                    effort=("high" if experiment in (1,2,4) else "xhigh") if model=="astra" else "max"))
    return {"schema":2,"upstream_commit":UPSTREAM,"models":MODELS,
        "skipped_experiments":[{"experiment":7,"reason":"Gemma experiments are outside the replication scope"}],
        "excluded_target_models":["google/gemma-3-270m-it"],
        "gemma_math_followup":"outside scope; mathematical weight editing targets Qwen only",
        "repetition_unit":"independent model generation/edit search, fresh context",
        "variance":{"across_runs":"sample variance ddof=1 over three run-level means",
                    "within_run":"episode variance/SE reported separately; no pseudo-replication"},
        "evaluation_seeds":list(range(1000)),"math_test_hidden_from_editor":True,
        "transport_policy":"restart an incomplete run on API quota fallback; never pool transports silently",
        "network_widths":{str(w):{"parameters":parameter_count(w),
            "parameter_fraction":parameter_count(w)/parameter_count(32)} for w in WIDTHS},
        "jobs":jobs}


if __name__ == "__main__":
    root=Path(__file__).resolve().parents[2]
    path=root/"configs/study.json"
    path.write_text(json.dumps(make_spec(),ensure_ascii=False,indent=2)+"\n")
    print(path)
