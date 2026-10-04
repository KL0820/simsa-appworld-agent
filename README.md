# SIMSA: an API task agent with reusable skills

This is the code for my master's research at National Taipei University of
Technology. Given an AppWorld task, the agent plans steps, finds APIs, executes
Python in AppWorld's sandbox, and checks progress before submitting. A separate
pipeline turns successful training trajectories into reusable instructions
("skills") for planning and execution.

The research question was whether selected skills could replace much of the
fixed, hand-written prompt while retaining task completions. The thesis is
available as a [thesis catalog record](https://hdl.handle.net/11296/k4pk4g),
linked from the [NTUT CSIE thesis list](https://csie.ntut.edu.tw/p/406-1070-151927%2Cr2190.php?Lang=zh-tw).

## Try one task

This is a benchmark prototype, not a hosted chatbot. A live task needs both
AppWorld's source/data and a Gemini API key (or configured Vertex AI access).
Model calls may incur charges. The walkthrough is generated from the **actual
run**, not a pre-scripted animation.

1. Install AppWorld and its task data following the [official setup guide](https://github.com/StonyBrookNLP/appworld/blob/main/README.md#installation).
2. Install this project's dependencies: `uv sync --frozen --extra dev --extra runtime`.
3. Copy `.env.example` to `.env`; set `APPWORLD_ROOT` and `GOOGLE_API_KEY`.
   Keep `.env` private. If AppWorld uses a different Python environment, set
   `APPWORLD_PYTHON` too.
4. From this repository, run:

```bash
uv run python -m scripts.demo
```

The launcher starts a local AppWorld RPC process, runs one previously tested
task (`13547f5_2`), stops the process, and prints paths to a task summary and
`task_view.html`. Open that HTML file in a browser to see the plan, retrieved
APIs, executed APIs, and evaluator outcome. A different installed task can be
selected with `--task-id`. Runs are under the git-ignored `logs/` directory.
The local RPC service is bound to `127.0.0.1` only. Review task text before
sharing a generated page; the page intentionally omits raw prompts, API
arguments, model responses, and credentials.

If AppWorld or a model key is unavailable, the [recorded task](docs/example-run.md)
shows a condensed, real execution. The [reproduction guide](docs/reproduction.md)
has the longer setup and five-task smoke command.

## What the experiments showed

| Experiment | Completed tasks |
|---|---:|
| Full AppWorld `test_normal` run | 96/168 |
| Fixed 56-task split, thin prompts without skills | 24/56 |
| Same split, thin prompts with curated skills | 32/56 |
| Same split, full hand-written prompts | 33/56 |

Across three components, fixed instruction text went from 4,184 to 297 words.
That is a reduction in **fixed prompt text**, not proof of lower total token use,
cost, or latency. These are single-seed thesis results; a one-task gap does not
establish that the two approaches are equivalent. The released library has 187
curated skills. The [evaluation notes](docs/evaluation.md) explain the splits,
the later smoke run, and what the numbers do not show.

## Where to look in the code

- [`AppWorldAgent._run_one_cycle`](src/adk_appworld_agent/agent.py) runs one
  planning/retrieval/execution/progress cycle. The framework owns task state;
  model workers return proposals and results.
- [`progress_policy.py`](src/adk_appworld_agent/orchestration/progress_policy.py)
  checks proposed next steps before they change the task state, including
  submission and retry limits.
- [`mind_skill/loop.py`](src/mind_skill/loop.py) coordinates skill induction,
  deduction, evaluation, and refinement. This adapts the MIND-Skill method;
  it does not train model weights.
- [One recorded task](docs/example-run.md) shows the APIs selected, what the
  agent executed, and the benchmark's independent result. It also shows a real
  limitation: even this small task took about two minutes.

For the component boundaries and design tradeoffs, see
[architecture](docs/architecture.md). For the complete test and live-run setup,
see [reproduction](docs/reproduction.md).

## Try the contract tests

With [uv](https://docs.astral.sh/uv/) installed, these tests need no model key or
AppWorld data:

```bash
uv sync --frozen --extra dev --extra runtime
uv run pytest -q \
  tests/test_completion_gate.py tests/test_variable_store.py \
  tests/test_scenario_goal_completion.py tests/test_demo_setup.py tests/test_task_viewer.py
```

The benchmark data, training trajectories, raw logs, and credentials are not
included here.

## About this public copy

This public repository begins with a source import in October 2026. It came
from a separate thesis working repository; that history remains private because
it includes local configuration, raw trajectories, and research logs.

The project uses Google ADK and AppWorld, adapts MIND-Skill, and includes a
CUGA-derived final-answer prompt. See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution. This is a benchmark prototype, not a production service.
