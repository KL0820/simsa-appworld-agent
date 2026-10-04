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

## First look: Python only

Clone the repository and start the local viewer. Python 3.12 or newer is enough
for this **illustrative preview**: it does not install packages, call a model,
or run an AppWorld task.

```bash
git clone https://github.com/KL0820/simsa-appworld-agent.git
cd simsa-appworld-agent
python3 -m scripts.demo --preview
```

The browser should open a `127.0.0.1` page. If it does not, open the printed
URL yourself. It shows a synthetic task progressing through planning, candidate
API retrieval, execution, progress control, and submission. The amber
**ILLUSTRATIVE PREVIEW** label distinguishes it from an actual run. Leave the
command running to keep the page open; press Ctrl-C when finished.

## Run a real task

This is a benchmark prototype, not a hosted chatbot. A live run needs
AppWorld's **source checkout and task data** plus a Gemini API key or Vertex AI
access. Model calls may incur charges. The viewer updates as the real agent
works; it is not a replay of the preview.

1. Install [AppWorld from source](https://github.com/StonyBrookNLP/appworld/blob/main/README.md#installation), including its `git lfs`, `appworld install --repo`, and `appworld download data` steps. Use an AppWorld checkout with `src/appworld` and `data` directories. For example, alongside this repository on macOS/Linux:

   ```bash
   git lfs install
   git clone https://github.com/StonyBrookNLP/appworld.git ../appworld
   cd ../appworld
   python3 -m venv .venv
   .venv/bin/python -m pip install -e .
   .venv/bin/appworld install --repo
   .venv/bin/appworld download data
   cd ../simsa-appworld-agent
   ```

2. Install this project's packages in its own environment:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -e '.[dev,runtime]'
   ```

3. Copy `.env.example` to `.env`. Set `APPWORLD_ROOT` to the **absolute path** of the AppWorld checkout and set `GOOGLE_API_KEY`. If AppWorld's interpreter is not at `APPWORLD_ROOT/.venv/bin/python`, set `APPWORLD_PYTHON` too. Keep `.env` private. The example file also describes the Vertex AI alternative.

4. Start one actual task:

   ```bash
   .venv/bin/python -m scripts.demo --live
   ```

The local page shows a loading state while the model plans, then its task
breakdown, candidate API names, API names actually called, execution summaries,
control decisions, and the independent evaluator result. It polls local events
about every 0.6 seconds: each stage appears as it starts and fills in when that
stage returns, not token by token. Retrieval, execution, and progress decisions
are grouped under the plan item they are working on; another pass over the same
item is labeled as a new cycle. After a real task completes, section 03 shows
task wall time, recorded LLM calls, and token counts. These subagent aggregates
are not complete billing or a cost estimate. The default task is
`13547f5_2`; choose another installed task with `--task-id`. A run can take
minutes. The launcher starts and stops AppWorld's RPC process and writes logs,
a summary, and a static `task_view.html` under the git-ignored `logs/` folder.
The default task is deliberately short and usually produces just one plan
item. To observe several plan items and control decisions, try
`--task-id 09b0ee6_2`; that task took about five minutes in the
[five-task smoke run](docs/evaluation.md), and live results may vary.
You can later reopen the timeline without spending model calls:
`.venv/bin/python -m scripts.demo --replay logs/dashboard_.../events.jsonl`
(replace the path with the one from your run under `logs/`).
For an older timeline made before metrics were recorded, pass its matching
`task_summary.json` with `--summary` to fill section 03.
Both HTTP services bind to `127.0.0.1` only. The live page omits raw prompts,
API arguments and response bodies, but task and step text can still be
sensitive: review it before sharing screenshots or generated files.

The [recorded task](docs/example-run.md) shows a condensed actual execution,
and the [reproduction guide](docs/reproduction.md) covers the longer experiment
setup and five-task smoke command.

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
