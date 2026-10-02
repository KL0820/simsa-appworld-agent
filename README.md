# SIMSA: reusable skills for an API task agent

This is the code behind my computer science master's thesis at National Taipei
University of Technology. The agent works on AppWorld tasks: it plans what to do,
finds relevant APIs, writes and runs Python in the benchmark sandbox, and checks
whether the task is finished. A separate pipeline turns training trajectories
into short, reusable instructions ("skills") for the agent's components.

The question I tested was practical: can selected skills replace much of the
hand-written prompt without giving up task completions?

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
uv run pytest -q tests/test_completion_gate.py tests/test_variable_store.py tests/test_scenario_goal_completion.py
```

Live tasks require a separate AppWorld installation and model credentials.
The benchmark data, training trajectories, raw logs, and credentials are not
included here.

## About this public copy

This public repository begins with a source import in October 2026. It came
from a separate thesis working repository; that history remains private because
it includes local configuration, raw trajectories, and research logs.

The project uses Google ADK and AppWorld, adapts MIND-Skill, and includes a
CUGA-derived final-answer prompt. See [third-party notices](THIRD_PARTY_NOTICES.md)
for attribution. This is a benchmark prototype, not a production service.
