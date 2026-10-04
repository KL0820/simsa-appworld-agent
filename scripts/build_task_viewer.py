"""Build a small, local HTML walkthrough from one task_summary.json.

Only a few summary fields are rendered. Raw prompts, API arguments, model
responses, and sandbox records are deliberately excluded.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _text(value: object) -> str:
    return html.escape(str(value if value is not None else "—"))


def _list_items(values: object) -> str:
    if not isinstance(values, list) or not values:
        return '<span class="muted">None recorded</span>'
    return ", ".join(f"<code>{_text(value)}</code>" for value in values)


def _milestone_card(index: int, milestone: dict) -> str:
    finder = milestone.get("find") or {}
    executor = milestone.get("execute") or {}
    return f"""
    <section class="card">
      <div class="eyebrow">Milestone {index}</div>
      <h3>{_text(milestone.get('goal'))}</h3>
      <div class="row"><strong>Target app</strong><span>{_text(milestone.get('app'))}</span></div>
      <div class="row"><strong>Retrieved APIs</strong><span>{_list_items(finder.get('selected_apis'))}</span></div>
      <div class="row"><strong>Called APIs</strong><span>{_list_items(executor.get('called_apis'))}</span></div>
      <div class="row"><strong>Execution</strong><span>{_text(executor.get('status'))}</span></div>
    </section>"""


def render(summary: dict) -> str:
    milestones = summary.get("milestones") or []
    if not isinstance(milestones, list):
        milestones = []
    cards = "\n".join(
        _milestone_card(index, item)
        for index, item in enumerate(milestones, 1)
        if isinstance(item, dict)
    ) or '<p class="muted">No milestones recorded.</p>'
    evaluation = summary.get("eval") or {}
    submission = summary.get("submission") or {}
    passed = evaluation.get("passed", submission.get("passed"))
    total = evaluation.get("total", submission.get("total"))
    score = f"{_text(passed)}/{_text(total)}" if total is not None else "Not available"
    status = str(summary.get("status") or "Unknown")
    status_class = "good" if passed is not None and total is not None and passed == total else "neutral"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SIMSA task {_text(summary.get('task_id'))}</title>
  <style>
    :root {{ color-scheme: light; font: 16px/1.55 system-ui, sans-serif; background: #f6f7f8; color: #202833; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 48px 24px 72px; }}
    h1 {{ margin: 4px 0 12px; font-size: clamp(2rem, 5vw, 3rem); letter-spacing: -.035em; line-height: 1.1; }}
    h2 {{ margin: 36px 0 16px; font-size: 1.25rem; }}
    h3 {{ margin: 4px 0 18px; font-size: 1.05rem; }}
    p {{ margin: 0 0 14px; }}
    .eyebrow {{ color: #536f86; font-size: .78rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; }}
    .intro {{ max-width: 760px; font-size: 1.1rem; }}
    .summary, .pipeline {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin-top: 24px; }}
    .tile, .step, .card {{ background: white; border: 1px solid #dfe4e8; border-radius: 10px; padding: 18px; }}
    .tile strong {{ display: block; margin-top: 4px; font-size: 1.25rem; }}
    .good {{ color: #176d47; }}
    .neutral, .muted {{ color: #667582; }}
    .step {{ border-top: 3px solid #607d92; font-weight: 650; }}
    .cards {{ display: grid; gap: 14px; }}
    .row {{ display: grid; grid-template-columns: 135px 1fr; gap: 12px; margin-top: 10px; }}
    .row strong {{ color: #52616d; font-size: .9rem; }}
    code {{ display: inline-block; margin: 2px 3px 2px 0; padding: 2px 6px; border-radius: 4px; background: #edf2f5; font-size: .88rem; overflow-wrap: anywhere; }}
    footer {{ margin-top: 32px; color: #63717b; font-size: .85rem; }}
    @media (max-width: 600px) {{ .row {{ grid-template-columns: 1fr; gap: 0; }} }}
  </style>
</head>
<body><main>
  <div class="eyebrow">SIMSA · Local task walkthrough</div>
  <h1>Task {_text(summary.get('task_id'))}</h1>
  <p class="intro">{_text(summary.get('instruction'))}</p>
  <div class="summary">
    <div class="tile"><div class="eyebrow">Run status</div><strong>{_text(status)}</strong></div>
    <div class="tile"><div class="eyebrow">Evaluator</div><strong class="{status_class}">{score}</strong></div>
    <div class="tile"><div class="eyebrow">Wall time</div><strong>{_text(summary.get('wall_s'))} s</strong></div>
  </div>
  <h2>How the agent worked</h2>
  <div class="pipeline"><div class="step">1 · Plan</div><div class="step">2 · Find APIs</div><div class="step">3 · Execute</div><div class="step">4 · Check progress</div><div class="step">5 · Evaluate</div></div>
  <h2>Milestones</h2><div class="cards">{cards}</div>
  <footer>Generated from a local task summary. Raw prompts, API arguments, results and credentials are not embedded. The benchmark evaluator, not the agent's own statement, determines task success. Review the task text before sharing this page.</footer>
</main></body></html>"""


def build_viewer(summary_path: Path, output_path: Path | None = None) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("Expected a JSON task summary object")
    output_path = output_path or summary_path.with_name("task_view.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(summary), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Path to artifacts/task_summary.json")
    parser.add_argument("--out", type=Path, help="HTML output path (default: beside summary)")
    args = parser.parse_args()
    print(build_viewer(args.summary, args.out))


if __name__ == "__main__":
    main()
