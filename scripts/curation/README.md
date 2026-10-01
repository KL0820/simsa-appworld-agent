# Skill-library curation tools

These are research runners around `src/mind_skill/curation/`, not prerequisites
for running the released online agent. They require the original training
artifacts and may make paid model calls.

| Script | Purpose |
|---|---|
| `run_curate_engine.py` | Executor-library consolidation and validation |
| `run_curate_planner.py` | Planner-library consolidation |
| `run_stage2_sharpen.py` | Refine confusing skill names and descriptions |
| `run_full90_curation.py` | Coordinate curation over the 90-task corpus |
| `continue_planner_round.py` | Resume planner curation from saved state |
| `gen_curation_summary.py` | Summarize accepted and rejected merges |
| `assemble_baseline_runs.py` | Prepare baseline records for deduction |

Run from the repository root with `PYTHONPATH=src`. Inspect each runner's
argument parser and required artifact paths before execution; the retained
research scripts assume training state that is not bundled in the portfolio.
Use a dedicated RPC port for executor deduction, separate from live evaluation.

For the online demo, use `scripts/run_batch.py` with
`configs/portfolio_smoke.json` and the released `thesis_final` skill library.
