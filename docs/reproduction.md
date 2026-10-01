# Setup and reproduction

## Dependencies and external data

The source snapshot pins the main libraries to the environment used during
cleanup validation. AppWorld runs in a separate environment; install its source
and required data using the instructions shipped with your AppWorld checkout.
This repository does not bundle AppWorld task databases, solutions or credentials.

Python 3.13 is selected in `.python-version` for the portable setup. On this Mac,
a fresh Python 3.14 install required compiling a transitive `watchdog` extension,
which was blocked by an unaccepted Xcode license. No system license settings were
changed; using Python 3.13 avoids that particular source-build requirement.

Set `APPWORLD_ROOT` to the AppWorld checkout containing both `src/appworld` and
`data`. Several offline tests read its API catalog; they require the data even
though they do not call a model.

```bash
uv sync --frozen --extra dev --extra runtime
export APPWORLD_ROOT=/absolute/path/to/appworld
cp .env.example .env
```

Choose one authentication route in `.env`: Gemini API key or Vertex AI ADC.
Do not commit the populated file. No credentials are needed for the offline tests.

## Offline validation

For a first review, these **20 core contract tests** require neither model
credentials nor AppWorld data; the GitHub Actions workflow runs this same subset:

```bash
uv run pytest -q tests/test_completion_gate.py tests/test_variable_store.py tests/test_scenario_goal_completion.py
```

For the full suite, set `APPWORLD_ROOT` to the installed data/source checkout:

```bash
uv run pytest -q -rs
```

Raw training trajectories are intentionally omitted, so corpus-dependent
deduction tests are marked skipped. The optional real embedding test also skips
when sentence-transformers or its model is unavailable. These are separate from
live AppWorld task success; see the evaluation report for exact validation counts.

The exported snapshot passed **600 tests, with 6 skips**, in a fresh Python 3.13
environment using the installed AppWorld catalog. Five skips concern the omitted
training corpus; one concerns the optional embedding model. The research checkout's
634-test count also includes parametrized corpus cases and other corpus checks
that cannot run without those private inputs.

## Live smoke

Use a dedicated RPC process and port. From the **AppWorld environment**, launch
the bundled adapter by absolute path, with `APPWORLD_ROOT` still exported:

```bash
cd "$APPWORLD_ROOT"
.venv/bin/python /absolute/path/to/simsa/scripts/appworld_rpc_server.py
```

The adapter listens only on `127.0.0.1:4244`. It is not an authenticated public
service; do not expose it outside the local machine. AppWorld APIs modify a
benchmark sandbox, not your personal accounts.

In a second terminal, from the SIMSA repository:

```bash
uv run scripts/run_batch.py \
  --task_ids 13547f5_2,024c982_2,fd1f8fa_2,042a9fc_2,09b0ee6_2 \
  --config configs/portfolio_smoke.json \
  --experiment_name portfolio_smoke --rpc_url tcp://127.0.0.1:4244 --keep-debug
```

This makes paid model calls. Inspect `logs/<experiment>/<run>/` for:

- `batch_results.txt`: batch summary;
- `<task>/task_summary.txt`: milestones and task evaluation;
- `<task>/artifacts/resolved_config.json`: actual settings;
- `<task>/artifacts/ledger.jsonl`: transitions;
- `<task>/artifacts/stderr.log`: provider failures/retries;
- `<task>/artifacts/evaluation.txt`: evaluator output.

Stop the RPC terminal with Ctrl-C when finished. Do not reuse the same server
concurrently for unrelated experiments.

## What is reproducible from this snapshot

The online runtime, released skills, fixed task IDs and tests are included.
Full offline retraining requires the separately retained 90-task source corpus
and additional embedding dependencies (`--extra community-pipeline`).
Historical hosted-model behavior and SDK versions differ from today's runtime;
fixed seeds do not guarantee identical trajectories or scores.
