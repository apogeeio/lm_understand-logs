# Weight-space programming replication

Replication and extensions of [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs), fixed at commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`.

[Results](RESULTS.md) cover 120 game-policy generation attempts, 20 weight-reading cases, two car repairs and eight Go attempts. Game evaluation produced 114 valid policies and six output-limit failures. Mathematical weight editing has no completed benchmark comparison in this release. Gemma experiments are outside the replication scope.

## Experimental design

| Component | Conditions | Independent generations |
|---|---|---:|
| Weight reading, experiments 1–3 | Original game, shape and language cases | One per case and model |
| Car repair, experiment 4 | Original one-shot task | One per model |
| Game networks, experiment 5 | Breakout, Seaquest, Asterix, custom Snake; hidden widths 32/16/8/4 | Three per cell and model |
| Python game baseline | Same four games; stateless policy, at most 40 physical code lines | Three per game and model |
| Go, experiment 6 | Feedback budgets 0/10/25/50 | One per budget and model |

Models are `gpt-6-astra` and `deepseek-flash`. Astra uses a native Responses coding-plan bridge; DeepSeek uses its Chat Completions API with thinking enabled and `reasoning_effort=max`. Astra game generation uses `xhigh`; other tasks retain the effort recorded in [study.json](configs/study.json). Returned model identifiers and transport are recorded per response.

Each independent generation starts with a fresh context. Requests contain the task and neutral instructions, with no native tools, repository instructions or unrelated history. Multi-turn searches retain only their own explicit history and opaque reasoning state. The bridge uses its provider-native output limit; DeepSeek's configured output limit is 128,000 tokens.

## Game protocol

- MinAtar 1.0.15; 10×10×6 observations and six actions. Original games preserve the upstream channel mapping and dynamics.
- Network parameters start at zero and are assigned from the final numeric edits block. No training or evaluator feedback is provided during generation.
- Each policy is frozen and hashed before evaluation on 1,000 episodes, seeds 0–999, sticky probability 0.1 and a 10,000-step cap. Action sampling uses a separate seeded random stream.
- Python policies receive the same current observation and action space, with a 100-ms per-action limit. They execute in a Linux bubblewrap sandbox without files outside the sandbox, network access, persistent memory or online models. Observation-derived prediction and planning are permitted.
- Parsing, execution and output-limit failures are retained as failed attempts. Infrastructure failures may be retried; completed replies and frozen policies are reused. Missing outputs are not replaced or scored as zero.

Snake is a custom extension: a 10×10 board with walls, initial length three, +1 reward per food item and no survival reward. Immediate reversal is ignored; entering a vacating tail is allowed. Filling the board wins. Channels encode head, body, food, empty cells, previous head and tail. No-op and fire continue the current heading. Sticky actions repeat the last executed action.

| Hidden width | Parameters | Fraction of width-32 network |
|---|---:|---:|
| 32 | 20,486 | 1.0000 |
| 16 | 9,990 | 0.4877 |
| 8 | 4,934 | 0.2408 |
| 4 | 2,454 | 0.1198 |

## Statistics and validity

Each cell reports the mean and sample variance (`ddof=1`) of the independently generated policies' episode means. Episode variance and standard error are stored separately. A group with fewer than three valid outputs reports its actual sample count and failures; variance is undefined at n=1. Channels are not pooled.

Generated Python strategies and weight-analysis trajectories undergo semantic review against task restrictions. Hash validation binds reviews to their recorded artifacts. Go's original weak/strong practice and small hidden opponent weights were not released; reported Go results use the available same-size opponents. The failed and excluded Go attempts are listed in the results.

## Calibration

All three released game networks were reconstructed bit-for-bit from their numeric edits. Environment tests check observations, dynamics and sticky-action random streams. Released-policy replays use the replication seeds and are separate from newly generated strategies.

| Released-policy replay, 1,000 episodes | Replication mean ± SE | Published mean ± SE |
|---|---:|---:|
| Breakout | 502.880 ± 7.747 | 517.411 ± 7.417 |
| Seaquest | 53.675 ± 1.347 | 53.110 ± 1.357 |
| Asterix | 55.362 ± 1.070 | 53.178 ± 1.023 |

The released car patch achieved 1,024/1,024 successful test episodes. Published PPO scores for experiment 5 remain source references because the corresponding checkpoints were not released.

## Running the game experiments

Python 3.12, the pinned game dependencies and Linux bubblewrap are required for Python-policy evaluation. Model generation uses external services; game evaluation runs on CPU.

```bash
uv venv --python 3.12 .venv-games
uv pip install --python .venv-games/bin/python -r replication/requirements-games.txt
export PYTHONPATH=replication/src
```

Create the Git-ignored `replication/configs/local.json` with the execution host and absolute checkout path:

```json
{
  "ssh_host": "evaluation-host",
  "remote_root": "/path/to/checkout",
  "coding_plan_base_url": "http://127.0.0.1:23880/v1"
}
```

`WSP_SSH_HOST` and `WSP_REMOTE_ROOT` can override the host and checkout path. Provider credentials are read from `~/.config/wsp-replication/providers.env` with mode 600; `DEEPSEEK_API_KEY` is required for DeepSeek. The coding-plan bridge supplies its own authentication. Credentials and local configuration are excluded from Git and policy sandboxes.

```bash
.venv-games/bin/python -m wsp_repro.spec
.venv-games/bin/python -m wsp_repro.run_games --workers 2 --evaluate
.venv-games/bin/python -m wsp_repro.aggregate
.venv-games/bin/python -m pytest replication/tests -q
```

`runs/` stores responses, frozen policies and per-episode evaluations. `artifacts/` stores operational and audit records. Both are ignored by Git. Public aggregate data and artifact hashes are in [results/](results/).

## Mathematical evaluation implementation

The implementation targets `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17`, using the original `think35` protocol and 32,768-new-token cap. The benchmark plan is retained in the manifest; this release reports no completed mathematical benchmark comparison.

Mathematical execution is disabled by `configs/math-pause.json`. Launch, search, evaluation and GPU entry points all enforce this configuration. Benchmark and search deferral files specify omitted evaluations and budgets. Changing execution configuration does not alter saved results.

The GPU environment is specified in `requirements-math.txt`: Python 3.12, torch 2.14/CUDA 13.0 and vLLM 0.33. Compiler components must use compatible versions: NVCC/NVVM/CRT 13.0.88 and CCCL 13.0.85. Evaluation data is isolated from the editing process, and submitted weights are frozen before held-out evaluation.

GPQA requires separately authorized data access. Original MMLU-Pro question IDs were recovered, but the private prompt helper is unavailable. The official IFEval checker agreed with 399 of 400 released sample flags; the remaining discrepancy is retained. These data and evaluator differences are tracked independently from experiment outcomes.
