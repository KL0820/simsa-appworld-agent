"""Config preflight rendering and interactive launcher state."""

from __future__ import annotations

import html
import json
import shlex
from pathlib import Path
from typing import Any

from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.registry import workflow_subagent_impls
from adk_appworld_agent.task_sets import (
    DATASET_CHOICES,
    VARIANT_CHOICES,
    task_set_counts,
)

_NESTED_SECTIONS = ("model", "retry", "retrieval", "paths", "cache", "debug", "log")
_LAUNCHER_FIELD_HELP = {
    "preset": {
        "en": ["Starting template for all settings.", "Example: default_thin_skill"],
        "zh": ["所有設定的起始模板；選取後仍可逐欄修改。", "範例：default_thin_skill"],
    },
    "model_name": {
        "en": [
            "Provider model identifier used by LLM workers.",
            "Example: gemini-2.5-flash",
        ],
        "zh": ["LLM worker 使用的供應商模型識別名稱。", "範例：gemini-2.5-flash"],
    },
    "temperature": {
        "en": [
            "Sampling randomness. Lower values are more deterministic.",
            "Example: 0 for reproducible evaluation; 0.7 for more variation.",
        ],
        "zh": [
            "控制取樣隨機性；數值越低，輸出越穩定且容易重現。",
            "範例：評估使用 0；需要更多變化時可用 0.7。",
        ],
    },
    "top_p": {
        "en": [
            "Nucleus-sampling probability mass. Usually tune this or temperature, not both.",
            "Example: 1.0 keeps the full probability mass.",
        ],
        "zh": [
            "核取樣的累積機率範圍；通常只調整 Top P 或 Temperature 其中一個。",
            "範例：1.0 表示保留完整機率範圍。",
        ],
    },
    "top_k": {
        "en": [
            "Restricts sampling to the K most likely tokens at each step.",
            "Example: 1 is greedy/deterministic; 40 allows wider choices.",
        ],
        "zh": [
            "每一步只從機率最高的 K 個 token 中取樣。",
            "範例：1 接近 deterministic；40 允許更廣的選擇。",
        ],
    },
    "seed": {
        "en": [
            "Requested random seed for reproducible sampling when supported by the provider.",
            "Example: 123",
        ],
        "zh": ["供應商支援時，用於重現取樣結果的隨機種子。", "範例：123"],
    },
    "candidate_count": {
        "en": [
            "Number of response candidates requested per model call. Higher values increase cost.",
            "Example: 1 for normal runs.",
        ],
        "zh": [
            "每次模型呼叫要求產生的候選回覆數；增加此值會提高成本。",
            "範例：一般執行使用 1。",
        ],
    },
    "max_output_tokens": {
        "en": [
            "Maximum generated tokens per model response. Zero leaves the limit unset.",
            "Example: 0 uses the provider default; 4096 applies an explicit cap.",
        ],
        "zh": [
            "每次模型回覆最多可產生的 token 數；0 代表不主動指定限制。",
            "範例：0 使用 provider default；4096 明確限制為 4096。",
        ],
    },
    "experiment_name": {
        "en": [
            "Stable name used to group the batch under the log root.",
            "Example: thin_skill_ablation",
        ],
        "zh": [
            "用來在 log root 下分組本次 batch 的名稱。",
            "範例：thin_skill_ablation",
        ],
    },
    "dataset": {
        "en": [
            "Fixed AppWorld task collection. Quick smoke is a tracked three-task regression set.",
            "Example: test_normal for thesis evaluation; quick_smoke after a code change.",
        ],
        "zh": [
            "固定的 AppWorld task collection；Quick smoke 是 tracked 的三題快速回歸集合。",
            "範例：論文評估使用 test_normal；程式修改後使用 quick_smoke。",
        ],
    },
    "variant": {
        "en": [
            "Choose the complete dataset or only task variants ending in _1, _2, or _3.",
            "Example: 2 selects only IDs ending in _2; full selects every variant.",
        ],
        "zh": [
            "選擇完整 dataset，或只執行結尾為 _1、_2、_3 的 task variant。",
            "範例：2 只選 `_2`；full 包含全部 variant。",
        ],
    },
    "extra_task_ids": {
        "en": [
            "Comma-separated targeted tasks appended to the selected dataset. Duplicates are removed.",
            "Example: smoke three tasks plus 07b42fd_2,229360a_2.",
        ],
        "zh": [
            "以逗號分隔、附加到所選 dataset 後方的 targeted tasks；重複 ID 會去除。",
            "範例：smoke 三題再加上 07b42fd_2,229360a_2。",
        ],
    },
    "offset": {
        "en": [
            "Number of tasks to skip from the start of a split.",
            "Example: 10 starts from the 11th task.",
        ],
        "zh": ["從 split 開頭略過的 task 數量。", "範例：10 代表從第 11 題開始。"],
    },
    "limit": {
        "en": [
            "Maximum number of split tasks to run. Zero means all remaining tasks.",
            "Example: 5 runs at most five tasks.",
        ],
        "zh": [
            "最多執行多少個 split task；0 代表執行剩餘全部。",
            "範例：5 代表最多執行五題。",
        ],
    },
    "log_root": {
        "en": ["Directory containing experiment and run artifacts.", "Example: logs"],
        "zh": ["存放 experiment 與 run artifacts 的根目錄。", "範例：logs"],
    },
    "rpc_url": {
        "en": [
            "Address of the AppWorld RPC server used by the batch.",
            "Example: tcp://127.0.0.1:4242",
        ],
        "zh": [
            "本次 batch 使用的 AppWorld RPC server 位址。",
            "範例：tcp://127.0.0.1:4242",
        ],
    },
    "find_impl": {
        "en": [
            "FIND retrieval implementation. community is the production default; the others are controlled retrieval ablations.",
            "Example: community_forward adds forward dependency expansion; app_scope_nodep removes community and dependency expansion.",
        ],
        "zh": [
            "FIND retrieval implementation。community 是正式預設；其餘選項是受控 retrieval ablation。",
            "範例：community_forward 加入 forward dependency expansion；app_scope_nodep 同時移除 community 與 dependency expansion。",
        ],
    },
    "plan_impl": {
        "en": [
            "Base PLAN implementation. The complete workflow uses rough; Skill mode derives its push/native variants.",
            "Example: rough produces the ordered milestone plan.",
        ],
        "zh": [
            "PLAN 的 base implementation。完整 workflow 使用 rough；Skill mode 會衍生其 push／native variants。",
            "範例：rough 產生有順序的 milestone plan。",
        ],
    },
    "execute_impl": {
        "en": [
            "Base EXECUTE implementation. The current complete workflow has one production executor; Skill mode derives its push/native variants.",
            "Example: code_plan_execute.",
        ],
        "zh": [
            "EXECUTE 的 base implementation。目前完整 workflow 只有一個正式 executor；Skill mode 會衍生 push／native variants。",
            "範例：code_plan_execute。",
        ],
    },
    "timeout_s": {
        "en": ["Hard wall-clock timeout for one task, in seconds.", "Example: 600"],
        "zh": ["單一 task 的 wall-clock 硬性逾時秒數。", "範例：600"],
    },
    "executor_timeout_s": {
        "en": ["Maximum time allowed for one executor code run.", "Example: 240"],
        "zh": ["單次 executor code run 最多允許的秒數。", "範例：240"],
    },
    "attempt_budget": {
        "en": [
            "Maximum EXECUTE attempts for one stable milestone. Zero disables this bound.",
            "Example: 6",
        ],
        "zh": [
            "單一 stable milestone 最多可進行的 EXECUTE 次數；0 會停用此限制。",
            "範例：6",
        ],
    },
    "skills_library": {
        "en": [
            "Skill condition and library used by planner/executor.",
            "Example: best enables the selected best-per-task skills; off disables skills.",
        ],
        "zh": [
            "Planner／Executor 使用的 skill condition 與 library。",
            "範例：best 啟用 best-per-task skills；off 關閉 skills。",
        ],
    },
    "skills_prompt_variant": {
        "en": [
            "Prompt base used when injecting deterministic skills.",
            "Example: minimal for the thin prompt; full for the full constraint prompt.",
        ],
        "zh": [
            "注入 deterministic skills 時使用的 prompt 基底。",
            "範例：minimal 使用 thin prompt；full 使用完整 constraint prompt。",
        ],
    },
    "skills_root": {
        "en": [
            "Root directory containing component/library/task skill files.",
            "Example: data/mind_skill/skills/release/thesis_final",
        ],
        "zh": [
            "存放 component／library／task skill files 的根目錄。",
            "範例：data/mind_skill/skills/release/thesis_final",
        ],
    },
    "keep_debug": {
        "en": [
            "Keeps detailed workflow and diagnostic artifacts. Uses more disk space.",
            "Example: enable for troubleshooting a smoke run.",
        ],
        "zh": [
            "保留詳細 workflow 與診斷 artifacts，會使用更多硬碟空間。",
            "範例：debug smoke run 時開啟。",
        ],
    },
    "executor_self_assess": {
        "en": [
            "Runs one grounded review after a clean executor result. Adds one model call.",
            "Example: enabled for current convergence diagnostics.",
        ],
        "zh": [
            "Executor clean result 後執行一次 grounded review；每次會增加一個 model call。",
            "範例：目前 convergence diagnostics 建議開啟。",
        ],
    },
    "config_json": {
        "en": [
            "Authoritative full RunConfig submitted to validation and launch.",
            "Example: use this for settings not exposed by the primary controls.",
        ],
        "zh": [
            "送交 validation 與 launch 的完整 authoritative RunConfig。",
            "範例：用來修改主畫面尚未提供的進階設定。",
        ],
    },
}

_LAUNCHER_I18N = {
    "en": {
        "subtitle": "Choose, validate, then start the batch.",
        "precedence_label": "Initial precedence:",
        "step_model_title": "1 · Model",
        "step_model_text": "Complete generation settings",
        "step_run_title": "2 · Run",
        "step_run_text": "Task, controller, skills, launch",
        "model_heading": "Model configuration",
        "model_intro": "All seven fields of the typed ModelConfig. Preset changes reload this page; explicit launcher CLI values remain higher priority.",
        "model_hint": "0 = provider default (no explicit limit)",
        "model_review": "Review the complete model configuration.",
        "next": "Next: Run settings",
        "task_heading": "Task selection",
        "controller_heading": "Controller & budgets",
        "skills_heading": "Skills & diagnostics",
        "advanced_summary": "Advanced: complete resolved RunConfig JSON",
        "advanced_text": "This is authoritative. Primary controls update it automatically; direct JSON edits are validated when starting.",
        "command_preview": "Command preview",
        "idle": "No batch has been started.",
        "back": "Back: Model",
        "validate": "Validate",
        "launch": "Start batch",
        "cancel": "Cancel batch",
        "info_label": "More information",
        "example": "Example",
    },
    "zh": {
        "subtitle": "選擇設定、驗證，然後啟動 batch。",
        "precedence_label": "初始設定優先序：",
        "step_model_title": "1 · Model",
        "step_model_text": "完整設定模型生成參數",
        "step_run_title": "2 · Run",
        "step_run_text": "設定 task、controller、skills 並啟動",
        "model_heading": "模型設定",
        "model_intro": "這裡包含 typed ModelConfig 的全部七個欄位。切換 preset 會重新載入設定；launcher CLI 明確指定的值仍有較高優先序。",
        "model_hint": "0 = provider default（不主動指定上限）",
        "model_review": "請確認完整模型設定。",
        "next": "下一步：Run settings",
        "task_heading": "Task 設定",
        "controller_heading": "Controller 與 budgets",
        "skills_heading": "Skills 與 diagnostics",
        "advanced_summary": "進階：完整 resolved RunConfig JSON",
        "advanced_text": "這是實際送交執行的 authoritative config。主要欄位會自動同步；直接修改 JSON 也會在啟動前驗證。",
        "command_preview": "執行指令預覽",
        "idle": "尚未啟動 batch。",
        "back": "返回：Model",
        "validate": "驗證設定",
        "launch": "開始執行",
        "cancel": "取消執行",
        "info_label": "更多說明",
        "example": "範例",
    },
}
_CLI_CONFIG_PATHS = {
    "model_name": ("model", "name"),
    "temperature": ("model", "temperature"),
    "top_p": ("model", "top_p"),
    "top_k": ("model", "top_k"),
    "seed": ("model", "seed"),
    "candidate_count": ("model", "candidate_count"),
    "max_output_tokens": ("model", "max_output_tokens"),
    "timeout": ("timeout_s",),
    "find_impl": ("impls", Phase.FIND.value),
    "plan_impl": ("impls", Phase.PLAN.value),
    "verify_impl": ("impls", Phase.VERIFY.value),
    "execute_impl": ("impls", Phase.EXECUTE.value),
    "skills_root": ("skills_root",),
    "skills_prompt": ("skills_prompt_variant",),
    "keep_debug": ("log", "keep_debug"),
}


def apply_cli_overrides(
    config: RunConfig,
    overrides: dict[str, Any],
) -> tuple[RunConfig, dict[str, str]]:
    """Overlay explicitly supplied launcher CLI values onto a base config."""
    data = config.model_dump(mode="json")
    sources: dict[str, str] = {}
    for cli_name, path in _CLI_CONFIG_PATHS.items():
        value = overrides.get(cli_name)
        if value is None:
            continue
        _set_nested(data, path, value)
        sources[".".join(path)] = "cli"

    skills = overrides.get("skills")
    if skills is not None:
        _apply_skills_mode(data, str(skills))
        sources["skills_library"] = "cli"
        sources[f"impls.{Phase.PLAN.value}"] = "cli"
        sources[f"impls.{Phase.EXECUTE.value}"] = "cli"
    return RunConfig.model_validate(data), sources


def discover_config_presets(
    configs_dir: Path,
    *,
    selected_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load selectable config presets, always including typed defaults."""
    presets = [
        {
            "name": "RunConfig defaults",
            "path": "",
            "config": RunConfig().model_dump(mode="json"),
        }
    ]
    paths = sorted(configs_dir.glob("*.json")) if configs_dir.exists() else []
    if (
        selected_path is not None
        and selected_path.exists()
        and selected_path not in paths
    ):
        paths.insert(0, selected_path)
    for path in paths:
        config = RunConfig.from_file(path)
        presets.append(
            {
                "name": path.stem,
                "path": str(path.resolve()),
                "config": config.model_dump(mode="json"),
            }
        )
    return presets


def build_launcher_view(
    config: RunConfig,
    *,
    config_path: Path | None,
    presets: list[dict[str, Any]],
    experiment_name: str,
    dataset: str | None = None,
    variant: str = "full",
    extra_task_ids: str | None = None,
    rpc_url: str,
    log_root: str,
    offset: int,
    limit: int,
    cli_sources: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the editable launcher payload consumed by the local webpage."""
    dataset = dataset or "quick_smoke"
    path = config_path or Path("<RunConfig defaults>")
    view = build_view_model(
        config,
        config_path=path,
        experiment_name=experiment_name,
        dataset=dataset,
        variant=variant,
        extra_task_ids=extra_task_ids,
    )
    view.update(
        {
            "launcher": True,
            "presets": presets,
            "selected_preset": str(config_path.resolve()) if config_path else "",
            "batch": {
                "experiment_name": experiment_name,
                "dataset": dataset,
                "variant": variant,
                "extra_task_ids": extra_task_ids or "",
                "rpc_url": rpc_url,
                "log_root": log_root,
                "offset": offset,
                "limit": limit,
            },
            "config": config.model_dump(mode="json"),
            "cli_sources": cli_sources or {},
            "precedence": "page edits > launcher CLI > selected preset > RunConfig defaults",
            "field_help": _LAUNCHER_FIELD_HELP,
            "i18n": _LAUNCHER_I18N,
            "dataset_options": list(DATASET_CHOICES),
            "variant_options": list(VARIANT_CHOICES),
            "task_set_counts": task_set_counts(),
            "workflow_impl_options": {
                phase.value: list(workflow_subagent_impls(phase))
                for phase in (Phase.FIND, Phase.PLAN, Phase.EXECUTE)
            },
        }
    )
    view["selection"] = f"{dataset}:{variant}"
    if extra_task_ids:
        view["selection"] += f" + {extra_task_ids}"
    view["selection_kind"] = "dataset"
    view["command"] = build_batch_command(
        path,
        experiment_name=experiment_name,
        dataset=dataset,
        variant=variant,
        extra_task_ids=extra_task_ids,
    )
    return view


def build_batch_command(
    config_path: Path,
    *,
    experiment_name: str,
    dataset: str = "quick_smoke",
    variant: str = "full",
    extra_task_ids: str | None = None,
) -> str:
    """Build the exact batch command represented by the viewer header."""
    command = [
        "uv",
        "run",
        "scripts/run_batch.py",
        "--config",
        str(config_path),
        "--experiment_name",
        experiment_name,
    ]
    command.extend(["--dataset", dataset, "--variant", variant])
    if extra_task_ids:
        command.extend(["--extra_task_ids", extra_task_ids])
    return shlex.join(command)


def build_view_model(
    config: RunConfig,
    *,
    config_path: Path,
    experiment_name: str,
    dataset: str = "quick_smoke",
    variant: str = "full",
    extra_task_ids: str | None = None,
) -> dict[str, Any]:
    """Convert typed config into render-ready sections and preflight facts."""
    resolved = config.model_dump(mode="json")
    defaults = RunConfig().model_dump(mode="json")
    sections = [
        _section(name, resolved[name], defaults[name], getattr(config, name).__doc__)
        for name in _NESTED_SECTIONS
    ]
    controller = {
        key: value for key, value in resolved.items() if key not in _NESTED_SECTIONS
    }
    controller_defaults = {
        key: value for key, value in defaults.items() if key not in _NESTED_SECTIONS
    }
    sections.append(
        _section(
            "controller",
            controller,
            controller_defaults,
            "Controller policy, implementations, prompt variants, and skill injection.",
        )
    )
    changed_count = sum(
        1 for section in sections for field in section["fields"] if field["changed"]
    )
    return {
        "config_path": str(config_path),
        "experiment_name": experiment_name,
        "selection": (
            f"{dataset}:{variant}" + (f" + {extra_task_ids}" if extra_task_ids else "")
        ),
        "selection_kind": "dataset",
        "command": build_batch_command(
            config_path,
            experiment_name=experiment_name,
            dataset=dataset,
            variant=variant,
            extra_task_ids=extra_task_ids,
        ),
        "sections": sections,
        "changed_count": changed_count,
        "field_count": sum(len(section["fields"]) for section in sections),
        "warnings": _preflight_warnings(config, config_path),
        "resolved_json": json.dumps(resolved, ensure_ascii=False, indent=2),
    }


def render_config_html(view: dict[str, Any]) -> str:
    """Render the preflight page, with launcher controls when enabled."""
    if view.get("launcher"):
        return _render_launcher_html(view)
    return _render_readonly_html(view)


def _render_readonly_html(view: dict[str, Any]) -> str:
    """Render the original standalone read-only page."""
    payload = json.dumps(view, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"Config preflight · {view['experiment_name']}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme: dark; --bg:#0a0d12; --panel:#111722; --line:#273244;
  --text:#e8eef7; --muted:#91a0b5; --accent:#7dd3fc; --changed:#fbbf24;
  --ok:#86efac; --warn:#fca5a5; }}
* {{ box-sizing:border-box; }}
html,body {{ max-width:100%; overflow-x:hidden; }}
body {{ margin:0; background:radial-gradient(circle at top left,#132033 0,#0a0d12 38%);
  color:var(--text); font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }}
main {{ width:min(1180px,calc(100% - 32px)); margin:32px auto 80px; }}
h1 {{ margin:0 0 5px; font:700 clamp(25px,4vw,40px)/1.1 system-ui,sans-serif; }}
.muted {{ color:var(--muted); overflow-wrap:anywhere; }}
.top {{ display:grid; grid-template-columns:1fr auto; gap:18px; align-items:start; }}
.status {{ border:1px solid #255b3b; color:var(--ok); background:#0d2117;
  padding:8px 12px; border-radius:999px; white-space:nowrap; }}
.summary {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:24px 0; }}
.metric,.command,.warning,.section {{ border:1px solid var(--line); background:rgba(17,23,34,.92);
  border-radius:14px; }}
.metric {{ padding:16px; }}
.metric strong {{ display:block; font-size:23px; color:var(--accent); overflow-wrap:anywhere; }}
.command {{ padding:16px; margin-bottom:14px; }}
.command-row {{ display:flex; gap:10px; align-items:center; }}
code {{ color:#d9f99d; overflow-wrap:anywhere; }}
button,input {{ font:inherit; }}
button {{ border:1px solid #3b82f6; color:#dbeafe; background:#172554; border-radius:8px;
  padding:7px 11px; cursor:pointer; }}
.warnings {{ display:grid; gap:8px; margin:14px 0 22px; }}
.warning {{ padding:12px 14px; color:var(--warn); border-color:#6b3030; background:#251315; }}
.toolbar {{ position:sticky; top:0; z-index:2; display:flex; gap:12px; align-items:center;
  padding:12px 0; background:linear-gradient(var(--bg) 72%,transparent); }}
input[type=search] {{ flex:1; min-width:120px; border:1px solid var(--line); background:#0d131d;
  color:var(--text); padding:10px 12px; border-radius:10px; }}
label {{ color:var(--muted); white-space:nowrap; }}
.sections {{ display:grid; gap:14px; }}
.section {{ overflow:hidden; }}
.section h2 {{ margin:0; padding:16px 18px 4px; font:650 19px system-ui,sans-serif; }}
.section .description {{ padding:0 18px 14px; color:var(--muted); }}
.field {{ display:grid; grid-template-columns:minmax(190px,30%) 1fr auto; gap:14px;
  align-items:start; padding:11px 18px; border-top:1px solid var(--line); }}
.field.changed {{ box-shadow:inset 3px 0 var(--changed); }}
.key {{ color:#c4b5fd; }}
.value {{ white-space:pre-wrap; overflow-wrap:anywhere; }}
.badge {{ color:var(--changed); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
details {{ margin-top:18px; }}
summary {{ cursor:pointer; color:var(--accent); }}
pre {{ max-height:650px; overflow:auto; padding:16px; border:1px solid var(--line);
  border-radius:12px; background:#080b10; color:#cbd5e1; }}
@media (max-width:720px) {{
  .top,.summary {{ grid-template-columns:1fr; }}
  .field {{ grid-template-columns:1fr; gap:5px; }}
  .command-row {{ align-items:flex-start; flex-direction:column; }}
}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div><h1>Run config preflight</h1><div class="muted" id="path"></div></div>
    <div class="status">VALID · typed RunConfig</div>
  </div>
  <div class="summary" id="summary"></div>
  <div class="command"><div class="muted">Batch command</div>
    <div class="command-row"><code id="command"></code><button id="copy">Copy</button></div>
  </div>
  <div class="warnings" id="warnings"></div>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Filter fields…">
    <label><input id="changed" type="checkbox"> changed only</label>
  </div>
  <div class="sections" id="sections"></div>
  <details><summary>Resolved JSON</summary><pre id="raw"></pre></details>
</main>
<script>
const view = {payload};
const esc = value => String(value).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const shown = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2);
document.querySelector('#path').textContent = view.config_path;
document.querySelector('#command').textContent = view.command;
document.querySelector('#raw').textContent = view.resolved_json;
document.querySelector('#summary').innerHTML = `
  <div class="metric"><span class="muted">Experiment</span><strong>${{esc(view.experiment_name)}}</strong></div>
  <div class="metric"><span class="muted">${{esc(view.selection_kind)}}</span><strong>${{esc(view.selection)}}</strong></div>
  <div class="metric"><span class="muted">Overrides</span><strong>${{view.changed_count}} / ${{view.field_count}}</strong></div>`;
document.querySelector('#warnings').innerHTML = view.warnings.map(w => `<div class="warning">${{esc(w)}}</div>`).join('');
document.querySelector('#sections').innerHTML = view.sections.map(section => `
  <section class="section" data-section="${{esc(section.name)}}">
    <h2>${{esc(section.name)}}</h2><div class="description">${{esc(section.description)}}</div>
    ${{section.fields.map(field => `<div class="field ${{field.changed ? 'changed' : ''}}"
      data-search="${{esc((section.name+' '+field.name+' '+shown(field.value)).toLowerCase())}}"
      data-changed="${{field.changed}}">
      <div class="key">${{esc(field.name)}}</div>
      <div class="value">${{esc(shown(field.value))}}</div>
      <div class="badge">${{field.changed ? 'override' : ''}}</div></div>`).join('')}}
  </section>`).join('');
function filter() {{
  const term = document.querySelector('#search').value.toLowerCase();
  const changedOnly = document.querySelector('#changed').checked;
  document.querySelectorAll('.field').forEach(row => {{
    row.hidden = !row.dataset.search.includes(term) || (changedOnly && row.dataset.changed !== 'true');
  }});
  document.querySelectorAll('.section').forEach(section => {{
    section.hidden = ![...section.querySelectorAll('.field')].some(row => !row.hidden);
  }});
}}
document.querySelector('#search').addEventListener('input', filter);
document.querySelector('#changed').addEventListener('change', filter);
document.querySelector('#copy').addEventListener('click', async event => {{
  await navigator.clipboard.writeText(view.command);
  event.target.textContent = 'Copied';
  setTimeout(() => event.target.textContent = 'Copy', 1000);
}});
</script>
</body>
</html>
"""


def _render_launcher_html(view: dict[str, Any]) -> str:
    payload = json.dumps(view, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"Run launcher · {view['experiment_name']}")
    dataset_options = "".join(
        f'<option value="{html.escape(dataset)}">'
        f"{html.escape(dataset)}"
        f"{' (3 tasks)' if dataset == 'quick_smoke' else ''}</option>"
        for dataset in view["dataset_options"]
    )
    variant_options = "".join(
        f'<option value="{html.escape(variant)}">'
        f"{html.escape(variant if variant == 'full' else f'_{variant}')}</option>"
        for variant in view["variant_options"]
    )
    implementation_options = {
        phase: "".join(
            f'<option value="{html.escape(impl)}">{html.escape(impl)}</option>'
            for impl in view["workflow_impl_options"][phase]
        )
        for phase in (Phase.FIND.value, Phase.PLAN.value, Phase.EXECUTE.value)
    }
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ color-scheme:dark; --bg:#090d13; --panel:#111824; --line:#293548;
  --text:#edf3fb; --muted:#94a3b8; --accent:#7dd3fc; --ok:#86efac;
  --warn:#fbbf24; --bad:#fca5a5; }}
* {{ box-sizing:border-box; }}
html,body {{ max-width:100%; overflow-x:hidden; }}
body {{ margin:0; background:radial-gradient(circle at top left,#15253a 0,#090d13 40%);
  color:var(--text); font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; }}
main {{ width:min(1120px,calc(100% - 28px)); max-width:calc(100% - 28px); margin:28px auto 72px; }}
h1,h2 {{ font-family:system-ui,sans-serif; }}
h1 {{ margin:0; font-size:clamp(28px,4vw,42px); }}
h2 {{ margin:0 0 14px; font-size:19px; }}
.muted {{ color:var(--muted); overflow-wrap:anywhere; }}
.top {{ display:flex; justify-content:space-between; gap:18px; align-items:start; }}
.top-actions {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
.locale-switch {{ display:flex; border:1px solid var(--line); border-radius:9px; overflow:hidden; }}
.locale-switch button {{ border:0; border-radius:0; background:#0c121c; color:var(--muted); padding:7px 10px; }}
.locale-switch button.active {{ background:#164e63; color:var(--text); }}
.status {{ padding:8px 12px; border:1px solid #2e6241; color:var(--ok);
  background:#0e2117; border-radius:999px; white-space:nowrap; }}
.precedence {{ margin:10px 0 22px; color:#c4b5fd; }}
.steps {{ min-width:0; max-width:100%; display:grid; grid-template-columns:repeat(2,minmax(0,1fr));
  gap:10px; margin:0 0 18px; }}
.step-indicator {{ border:1px solid var(--line); border-radius:12px; padding:12px 14px;
  color:var(--muted); background:#0c121c; }}
.step-indicator.active {{ border-color:#38bdf8; color:var(--text); background:#102338; }}
.step-indicator strong {{ display:block; color:var(--accent); font-family:system-ui,sans-serif; }}
.wizard-step {{ min-width:0; max-width:100%; }}
.wizard-step[hidden] {{ display:none; }}
.grid {{ min-width:0; max-width:100%; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.panel {{ width:100%; min-width:0; max-width:100%; overflow:hidden;
  border:1px solid var(--line); border-radius:14px; background:rgba(17,24,36,.94);
  padding:18px; margin-bottom:14px; }}
.wide {{ grid-column:1/-1; }}
.field {{ min-width:0; display:grid; gap:6px; margin-bottom:13px; }}
.field:last-child {{ margin-bottom:0; }}
label {{ color:var(--muted); }}
.info-wrap {{ position:relative; display:inline-flex; align-items:center; }}
.info-button {{ display:inline-grid; place-items:center; width:18px; height:18px; min-height:18px;
  padding:0; border:1px solid #64748b; border-radius:50%; background:#111827;
  color:#bae6fd; font:700 11px/1 system-ui,sans-serif; }}
.info-button:hover,.info-button:focus-visible {{ border-color:#38bdf8; background:#0c4a6e; outline:none; }}
.info-tooltip {{ position:fixed; z-index:20; display:none; width:min(340px,calc(100vw - 24px));
  padding:11px 12px; border:1px solid #475569; border-radius:10px; background:#07111f;
  color:var(--text); box-shadow:0 12px 30px rgba(0,0,0,.42); font:13px/1.5 system-ui,sans-serif; }}
.info-tooltip.visible {{ display:block; }}
.info-tooltip .tooltip-example {{ margin-top:7px; color:#bae6fd; }}
input,select,textarea,button {{ font:inherit; }}
input,select,textarea {{ width:100%; color:var(--text); background:#0b111a;
  min-width:0; max-width:100%; border:1px solid #334155; border-radius:9px; padding:10px 11px; }}
textarea {{ min-height:310px; resize:vertical; tab-size:2; }}
.row {{ min-width:0; max-width:100%; display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
.model-row {{ grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr)); }}
.hint {{ color:var(--muted); font-size:12px; }}
.checks {{ display:flex; flex-wrap:wrap; gap:16px; }}
.checks label {{ display:flex; gap:8px; align-items:center; color:var(--text); }}
.checks input {{ width:auto; }}
.source {{ color:var(--warn); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }}
.actions {{ width:100%; min-width:0; max-width:100%; position:sticky; bottom:0;
  display:flex; justify-content:space-between; gap:14px;
  align-items:center; border:1px solid #334155; border-radius:14px; padding:14px 16px;
  background:rgba(9,13,19,.96); box-shadow:0 -10px 30px rgba(0,0,0,.25); }}
button {{ border:1px solid #3b82f6; color:#dbeafe; background:#172554;
  border-radius:9px; padding:10px 14px; cursor:pointer; }}
button.primary {{ background:#075985; border-color:#38bdf8; font-weight:700; }}
button.danger {{ color:#fecaca; background:#450a0a; border-color:#ef4444; }}
button.danger:hover:not(:disabled) {{ background:#7f1d1d; }}
button:disabled {{ opacity:.5; cursor:not-allowed; }}
.message {{ min-height:20px; overflow-wrap:anywhere; }}
.message.error {{ color:var(--bad); }}
.message.success {{ color:var(--ok); }}
details {{ margin-top:12px; }}
summary {{ color:var(--accent); cursor:pointer; }}
code {{ color:#d9f99d; overflow-wrap:anywhere; }}
@media (max-width:760px) {{
  main {{ width:auto; max-width:none; margin:20px 14px 72px; }}
  .steps,.grid,.row {{ grid-template-columns:1fr; }}
  .top,.actions {{ align-items:stretch; flex-direction:column; }}
  .top-actions {{ justify-content:flex-start; }}
  .status {{ align-self:flex-start; }}
}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div><h1>AppWorld run launcher</h1><div class="muted" data-i18n="subtitle">Choose, validate, then start the batch.</div></div>
    <div class="top-actions">
      <div class="locale-switch" aria-label="Language">
        <button type="button" data-locale="zh">中文</button>
        <button type="button" data-locale="en">English</button>
      </div>
      <div class="status" id="status">READY</div>
    </div>
  </div>
  <div class="precedence"><span data-i18n="precedence_label">Initial precedence:</span> <span id="precedence"></span></div>

  <div class="steps">
    <div class="step-indicator active" data-step-indicator="1"><strong data-i18n="step_model_title">1 · Model</strong><span data-i18n="step_model_text">Complete generation settings</span></div>
    <div class="step-indicator" data-step-indicator="2"><strong data-i18n="step_run_title">2 · Run</strong><span data-i18n="step_run_text">Task, controller, skills, launch</span></div>
  </div>

  <section class="wizard-step" data-step="1">
    <div class="grid">
      <section class="panel wide">
        <h2 data-i18n="model_heading">Model configuration</h2>
        <p class="muted" data-i18n="model_intro">All seven fields of the typed ModelConfig. Preset changes reload this page; explicit launcher CLI values remain higher priority.</p>
        <div class="field"><label>Config preset</label><select id="preset"></select></div>
        <div class="field"><label>Model name <span class="source" data-source="model.name"></span></label>
          <input id="model_name" placeholder="gemini-2.5-flash"></div>
        <div class="row model-row">
          <div class="field"><label>Temperature <span class="source" data-source="model.temperature"></span></label><input id="temperature" type="number" step="0.1"></div>
          <div class="field"><label>Top P <span class="source" data-source="model.top_p"></span></label><input id="top_p" type="number" step="0.1"></div>
          <div class="field"><label>Top K <span class="source" data-source="model.top_k"></span></label><input id="top_k" type="number" min="1"></div>
        </div>
        <div class="row model-row">
          <div class="field"><label>Seed <span class="source" data-source="model.seed"></span></label><input id="seed" type="number"></div>
          <div class="field"><label>Candidate count <span class="source" data-source="model.candidate_count"></span></label><input id="candidate_count" type="number" min="1"></div>
          <div class="field">
            <label>Max output tokens <span class="source" data-source="model.max_output_tokens"></span></label>
            <input id="max_output_tokens" type="number" min="0">
            <div class="hint" data-i18n="model_hint">0 = provider default (no explicit limit)</div>
          </div>
        </div>
      </section>
    </div>
    <div class="actions">
      <div id="model_message" class="message muted" data-i18n="model_review">Review the complete model configuration.</div>
      <button id="next" class="primary" data-i18n="next">Next: Run settings</button>
    </div>
  </section>

  <section class="wizard-step" data-step="2" hidden>
  <div class="grid">
    <section class="panel wide task-panel">
      <h2 data-i18n="task_heading">Task selection</h2>
      <div class="field"><label>Experiment name <span class="source" data-source="experiment_name"></span></label>
        <input id="experiment_name"></div>
      <div class="row">
        <div class="field"><label>Dataset</label>
          <select id="dataset">{dataset_options}</select>
        </div>
        <div class="field"><label>Variant</label>
          <select id="variant">{variant_options}</select>
        </div>
        <div class="field"><label>Extra Task IDs</label><input id="extra_task_ids" placeholder="a_2,b_2"></div>
      </div>
      <div id="selection_summary" class="hint"></div>
      <div class="row">
        <div class="field"><label>Offset</label><input id="offset" type="number" min="0"></div>
        <div class="field"><label>Limit (0 = all)</label><input id="limit" type="number" min="0"></div>
        <div class="field"><label>Log root</label><input id="log_root"></div>
      </div>
      <div class="field"><label>RPC URL</label><input id="rpc_url"></div>
    </section>

    <section class="panel wide controller-panel">
      <h2 data-i18n="controller_heading">Controller & budgets</h2>
      <div class="row">
        <div class="field"><label>Finder <span class="source" data-source="impls.FIND"></span></label><select id="find_impl">{implementation_options[Phase.FIND.value]}</select></div>
        <div class="field"><label>Planner <span class="source" data-source="impls.PLAN"></span></label><select id="plan_impl">{implementation_options[Phase.PLAN.value]}</select></div>
        <div class="field"><label>Executor <span class="source" data-source="impls.EXECUTE"></span></label><select id="execute_impl">{implementation_options[Phase.EXECUTE.value]}</select></div>
      </div>
      <div class="row">
        <div class="field"><label>Task timeout (s) <span class="source" data-source="timeout_s"></span></label><input id="timeout_s" type="number" min="1"></div>
        <div class="field"><label>Executor timeout (s)</label><input id="executor_timeout_s" type="number" min="1"></div>
        <div class="field"><label>Attempt budget</label><input id="attempt_budget" type="number" min="0"></div>
      </div>
    </section>

    <section class="panel wide">
      <h2 data-i18n="skills_heading">Skills & diagnostics</h2>
      <div class="row">
        <div class="field"><label>Skill library <span class="source" data-source="skills_library"></span></label>
          <select id="skills_library"><option>off</option><option>best</option><option>q0</option><option>q1</option><option>q2</option><option>thin</option><option>native_best</option><option>native_q0</option><option>native_q1</option><option>native_q2</option></select>
        </div>
        <div class="field"><label>Skill prompt <span class="source" data-source="skills_prompt_variant"></span></label><select id="skills_prompt_variant"><option>minimal</option><option>full</option></select></div>
        <div class="field"><label>Skills root <span class="source" data-source="skills_root"></span></label><input id="skills_root"></div>
      </div>
      <div class="checks">
        <label><input id="keep_debug" type="checkbox"> Keep debug artifacts <span class="source" data-source="log.keep_debug"></span></label>
        <label><input id="executor_self_assess" type="checkbox"> Executor self-assess</label>
      </div>
      <details>
        <summary data-i18n="advanced_summary">Advanced: complete resolved RunConfig JSON</summary>
        <p class="muted" data-i18n="advanced_text">This is authoritative. Primary controls update it automatically; direct JSON edits are validated when starting.</p>
        <textarea id="config_json" spellcheck="false"></textarea>
      </details>
    </section>
  </div>

  <div class="panel"><div class="muted" data-i18n="command_preview">Command preview</div><code id="command"></code></div>
  <div class="actions">
    <div id="message" class="message muted" data-i18n="idle">No batch has been started.</div>
    <div><button id="back" data-i18n="back">Back: Model</button> <button id="validate" data-i18n="validate">Validate</button> <button id="cancel" class="danger" data-i18n="cancel" disabled>Cancel batch</button> <button id="launch" class="primary" data-i18n="launch">Start batch</button></div>
  </div>
  </section>
</main>
<div id="info_tooltip" class="info-tooltip" role="tooltip"></div>
<script>
const view = {payload};
let config = structuredClone(view.config);
let activePreset = view.selected_preset;
let locale = localStorage.getItem('appworld-launcher-locale') || (navigator.language.startsWith('zh') ? 'zh' : 'en');
let statusTimer = null;
const byId = id => document.getElementById(id);
const t = key => (view.i18n[locale] || view.i18n.en)[key] || key;
const phase = (name, fallback) => (config.impls || {{}})[name] || fallback;
const num = id => Number(byId(id).value);
const checked = id => byId(id).checked;
const skillSelection = () => {{
  const plan = phase('PLAN', 'rough');
  const execute = phase('EXECUTE', 'code_plan_execute');
  if (!plan.includes('skill') && !execute.includes('skill')) return 'off';
  if (config.skills_library === 'none') return 'thin';
  const native = plan.includes('native') || execute.includes('native');
  return native ? `native_${{config.skills_library || 'best'}}` : (config.skills_library || 'best');
}};

function installFieldHelp() {{
  for (const [id, content] of Object.entries(view.field_help)) {{
    const control = byId(id);
    if (!control) continue;
    let label = control.closest('label');
    if (!label) label = control.closest('.field')?.querySelector('label');
    if (!label && id === 'config_json') label = document.querySelector('[data-i18n="advanced_summary"]');
    if (!label || label.querySelector('.info-wrap')) continue;
    const wrap = document.createElement('span');
    wrap.className = 'info-wrap';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'info-button';
    button.textContent = '!';
    button.dataset.helpId = id;
    wrap.append(button);
    label.append(wrap);
    button.addEventListener('mouseenter', () => showHelp(button));
    button.addEventListener('mouseleave', hideHelp);
    button.addEventListener('focus', () => showHelp(button));
    button.addEventListener('blur', hideHelp);
    button.addEventListener('click', event => {{
      event.preventDefault();
      event.stopPropagation();
      showHelp(button);
    }});
  }}
}}
function showHelp(button) {{
  const content = view.field_help[button.dataset.helpId]?.[locale] || view.field_help[button.dataset.helpId]?.en;
  if (!content) return;
  const tooltip = byId('info_tooltip');
  tooltip.innerHTML = `<div>${{content[0]}}</div><div class="tooltip-example">${{content[1]}}</div>`;
  tooltip.classList.add('visible');
  const rect = button.getBoundingClientRect();
  const width = Math.min(340, window.innerWidth - 24);
  const left = Math.max(12, Math.min(window.innerWidth - width - 12, rect.left - 8));
  const topBelow = rect.bottom + 8;
  const top = topBelow + 130 < window.innerHeight ? topBelow : Math.max(8, rect.top - 130);
  tooltip.style.width = `${{width}}px`;
  tooltip.style.left = `${{left}}px`;
  tooltip.style.top = `${{top}}px`;
}}
function hideHelp() {{ byId('info_tooltip').classList.remove('visible'); }}
function applyLocale(nextLocale) {{
  locale = nextLocale;
  localStorage.setItem('appworld-launcher-locale', locale);
  document.documentElement.lang = locale === 'zh' ? 'zh-Hant' : 'en';
  document.querySelectorAll('[data-i18n]').forEach(element => {{
    element.textContent = t(element.dataset.i18n);
  }});
  document.querySelectorAll('[data-locale]').forEach(button => {{
    button.classList.toggle('active', button.dataset.locale === locale);
  }});
  installFieldHelp();
  document.querySelectorAll('.info-button').forEach(button => {{
    button.setAttribute('aria-label', `${{t('info_label')}}: ${{button.closest('label,summary')?.textContent.replace('!', '').trim() || ''}}`);
  }});
  byId('precedence').textContent = locale === 'zh'
    ? '頁面修改 > launcher CLI > selected preset > RunConfig defaults'
    : view.precedence;
  if (byId('selection_summary')) syncDatasetControls();
  hideHelp();
}}

function setValue(id, value) {{
  const el = byId(id);
  if (el.type === 'checkbox') el.checked = Boolean(value);
  else el.value = value ?? '';
}}
function loadConfig(next) {{
  config = structuredClone(next);
  setValue('model_name', config.model.name);
  setValue('temperature', config.model.temperature);
  setValue('top_p', config.model.top_p);
  setValue('top_k', config.model.top_k);
  setValue('seed', config.model.seed);
  setValue('candidate_count', config.model.candidate_count);
  setValue('max_output_tokens', config.model.max_output_tokens);
  setValue('find_impl', phase('FIND', 'community'));
  const resolvedPlan = phase('PLAN', 'rough');
  const resolvedExecute = phase('EXECUTE', 'code_plan_execute');
  setValue('plan_impl', resolvedPlan.startsWith('rough_skill') ? 'rough' : resolvedPlan);
  setValue('execute_impl', resolvedExecute.startsWith('code_plan_execute_skill') ? 'code_plan_execute' : resolvedExecute);
  setValue('timeout_s', config.timeout_s);
  setValue('executor_timeout_s', config.executor_timeout_s);
  setValue('attempt_budget', config.attempt_budget);
  setValue('skills_library', skillSelection());
  setValue('skills_prompt_variant', config.skills_prompt_variant || 'minimal');
  setValue('skills_root', config.skills_root);
  setValue('keep_debug', config.log.keep_debug);
  setValue('executor_self_assess', config.executor_self_assess);
  syncJson();
}}
function syncConfig() {{
  config.model.name = byId('model_name').value.trim();
  config.model.temperature = num('temperature');
  config.model.top_p = num('top_p');
  config.model.top_k = num('top_k');
  config.model.seed = num('seed');
  config.model.candidate_count = num('candidate_count');
  config.model.max_output_tokens = num('max_output_tokens');
  config.impls = config.impls || {{}};
  config.impls.FIND = byId('find_impl').value;
  const skillMode = byId('skills_library').value;
  const native = skillMode.startsWith('native_');
  const basePlan = byId('plan_impl').value;
  const baseExecute = byId('execute_impl').value;
  config.impls.PLAN = skillMode === 'off' || basePlan !== 'rough'
    ? basePlan
    : (native ? 'rough_skill_native' : 'rough_skill');
  config.impls.EXECUTE = skillMode === 'off'
    ? baseExecute
    : (native ? 'code_plan_execute_skill_native' : 'code_plan_execute_skill');
  config.timeout_s = num('timeout_s');
  config.executor_timeout_s = num('executor_timeout_s');
  config.attempt_budget = num('attempt_budget');
  config.skills_library = skillMode === 'thin' ? 'none' : skillMode.replace('native_', '').replace('off', 'best');
  config.skills_prompt_variant = byId('skills_prompt_variant').value;
  config.skills_root = byId('skills_root').value.trim();
  config.log.keep_debug = checked('keep_debug');
  config.executor_self_assess = checked('executor_self_assess');
  syncJson();
  updateCommand();
}}
function syncJson() {{ byId('config_json').value = JSON.stringify(config, null, 2); }}
function batch() {{
  return {{
    experiment_name: byId('experiment_name').value.trim(),
    dataset: byId('dataset').value,
    variant: byId('variant').value,
    extra_task_ids: byId('extra_task_ids').value.trim(),
    rpc_url: byId('rpc_url').value.trim(),
    log_root: byId('log_root').value.trim(),
    offset: num('offset'),
    limit: num('limit')
  }};
}}
function commandFor(b) {{
  const extras = b.extra_task_ids ? ` --extra_task_ids ${{b.extra_task_ids}}` : '';
  return `uv run scripts/run_batch.py --config <generated-config> --experiment_name ${{b.experiment_name}} --dataset ${{b.dataset}} --variant ${{b.variant}}${{extras}} --rpc_url ${{b.rpc_url}} --log_root ${{b.log_root}} --offset ${{b.offset}} --limit ${{b.limit}}`;
}}
function updateCommand() {{ byId('command').textContent = commandFor(batch()); }}
function message(text, kind='muted') {{
  byId('message').className = `message ${{kind}}`;
  byId('message').textContent = text;
}}
function showStep(step) {{
  document.querySelectorAll('.wizard-step').forEach(section => section.hidden = section.dataset.step !== String(step));
  document.querySelectorAll('.step-indicator').forEach(indicator => indicator.classList.toggle('active', indicator.dataset.stepIndicator === String(step)));
  window.scrollTo({{top:0, behavior:'smooth'}});
}}
function validateModelFields() {{
  syncConfig();
  if (!config.model.name) throw new Error(locale === 'zh' ? 'Model name 不可留空。' : 'Model name is required.');
  if (!Number.isFinite(config.model.temperature)) throw new Error(locale === 'zh' ? 'Temperature 必須是數字。' : 'Temperature must be a number.');
  if (config.model.top_p < 0 || config.model.top_p > 1) throw new Error(locale === 'zh' ? 'Top P 必須介於 0 和 1。' : 'Top P must be between 0 and 1.');
  if (config.model.top_k < 1) throw new Error(locale === 'zh' ? 'Top K 至少為 1。' : 'Top K must be at least 1.');
  if (config.model.candidate_count < 1) throw new Error(locale === 'zh' ? 'Candidate count 至少為 1。' : 'Candidate count must be at least 1.');
  if (config.model.max_output_tokens < 0) throw new Error(locale === 'zh' ? 'Max output tokens 不可小於 0。' : 'Max output tokens cannot be negative.');
}}
function payload() {{
  let parsed;
  try {{ parsed = JSON.parse(byId('config_json').value); }}
  catch (error) {{ throw new Error(`Invalid RunConfig JSON: ${{error.message}}`); }}
  return {{config:parsed, batch:batch()}};
}}
async function post(path) {{
  const response = await fetch(path, {{
    method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload())
  }});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${{response.status}}`);
  return result;
}}
function setRunControls(running) {{
  byId('launch').disabled = running;
  byId('validate').disabled = running;
  byId('cancel').disabled = !running;
}}
function stopStatusPolling() {{
  if (statusTimer !== null) clearInterval(statusTimer);
  statusTimer = null;
}}
function startStatusPolling() {{
  stopStatusPolling();
  statusTimer = setInterval(refreshRunStatus, 2000);
}}
function showTerminalStatus(current) {{
  stopStatusPolling();
  setRunControls(false);
  if (current.state === 'cancelled') {{
    byId('status').textContent = `CANCELLED · PID ${{current.pid}}`;
    message(locale === 'zh' ? 'Batch 已手動取消。' : 'Batch was cancelled.', 'error');
    return;
  }}
  byId('status').textContent = `FINISHED · EXIT ${{current.returncode}}`;
  message(
    locale === 'zh' ? `Batch 已完成，exit code ${{current.returncode}}。` : `Batch finished with exit code ${{current.returncode}}.`,
    current.returncode === 0 ? 'success' : 'error'
  );
}}
async function refreshRunStatus() {{
  try {{
    const response = await fetch('/api/status');
    if (!response.ok) return;
    const current = await response.json();
    if (current.state === 'running') {{
      setRunControls(true);
      byId('status').textContent = `RUNNING · PID ${{current.pid}}`;
      return;
    }}
    if (current.state === 'finished' || current.state === 'cancelled') {{
      showTerminalStatus(current);
    }}
  }} catch (_error) {{
    // The launcher page remains usable if a transient status request fails.
  }}
}}
byId('precedence').textContent = view.precedence;
for (const preset of view.presets) {{
  const option = document.createElement('option');
  option.value = preset.path; option.textContent = preset.name;
  option.selected = preset.path === activePreset;
  byId('preset').append(option);
}}
byId('preset').addEventListener('change', event => {{
  const preset = view.presets.find(item => item.path === event.target.value);
  if (preset) {{
    activePreset = preset.path;
    loadConfig(preset.config);
    message(locale === 'zh' ? `已載入 preset：${{preset.name}}` : `Loaded preset: ${{preset.name}}`);
  }}
}});
for (const [key, source] of Object.entries(view.cli_sources)) {{
  document.querySelectorAll(`[data-source="${{key}}"]`).forEach(el => el.textContent = source);
}}
for (const [key, value] of Object.entries(view.batch)) setValue(key, value);
loadConfig(config);
function syncDatasetControls() {{
  const fixedFull = ['quick_smoke', 'extra_only'].includes(byId('dataset').value);
  if (fixedFull) byId('variant').value = 'full';
  byId('variant').disabled = fixedFull;
  const b = batch();
  const datasetCount = view.task_set_counts[b.dataset]?.[b.variant] ?? 0;
  const afterOffset = Math.max(0, datasetCount - b.offset);
  const baseCount = b.limit > 0 ? Math.min(afterOffset, b.limit) : afterOffset;
  const extras = [...new Set(b.extra_task_ids.split(',').map(item => item.trim()).filter(Boolean))];
  byId('selection_summary').textContent = locale === 'zh'
    ? `Base ${{baseCount}} 題；輸入 ${{extras.length}} 個 Extra Task IDs（執行時會與 Base 去重）。`
    : `Base: ${{baseCount}} tasks; entered extras: ${{extras.length}} (deduplicated against the base at runtime).`;
  updateCommand();
}}
document.querySelectorAll('[data-locale]').forEach(button => {{
  button.addEventListener('click', () => applyLocale(button.dataset.locale));
}});
document.querySelectorAll('input,select').forEach(el => {{
  if (el.id !== 'preset') el.addEventListener('input', syncConfig);
}});
byId('dataset').addEventListener('change', syncDatasetControls);
byId('variant').addEventListener('change', syncDatasetControls);
byId('extra_task_ids').addEventListener('input', syncDatasetControls);
byId('offset').addEventListener('input', syncDatasetControls);
byId('limit').addEventListener('input', syncDatasetControls);
byId('config_json').addEventListener('input', updateCommand);
byId('next').addEventListener('click', () => {{
  try {{
    validateModelFields();
    byId('model_message').className = 'message success';
    byId('model_message').textContent = locale === 'zh' ? '模型設定有效。' : 'Model settings valid.';
    showStep(2);
  }} catch (error) {{
    byId('model_message').className = 'message error';
    byId('model_message').textContent = error.message;
  }}
}});
byId('back').addEventListener('click', () => showStep(1));
byId('validate').addEventListener('click', async () => {{
  try {{
    const result = await post('/api/validate');
    message(locale === 'zh' ? `設定有效：${{result.command}}` : `Valid: ${{result.command}}`, 'success');
  }}
  catch (error) {{ message(error.message, 'error'); }}
}});
byId('launch').addEventListener('click', async event => {{
  setRunControls(true);
  message(locale === 'zh' ? '正在啟動 batch…' : 'Starting batch…');
  try {{
    const result = await post('/api/launch');
    byId('status').textContent = `RUNNING · PID ${{result.pid}}`;
    message(locale === 'zh' ? `已啟動。Config：${{result.config_path}}` : `Started. Config: ${{result.config_path}}`, 'success');
    startStatusPolling();
  }} catch (error) {{
    message(error.message, 'error'); setRunControls(false);
  }}
}});
byId('cancel').addEventListener('click', async () => {{
  const confirmed = window.confirm(
    locale === 'zh'
      ? '確定要取消目前執行中的 batch？正在執行的 task 也會停止。'
      : 'Cancel the running batch? The active task will also stop.'
  );
  if (!confirmed) return;
  byId('cancel').disabled = true;
  message(locale === 'zh' ? '正在取消 batch…' : 'Cancelling batch…');
  try {{
    const result = await post('/api/cancel');
    showTerminalStatus(result);
  }} catch (error) {{
    message(error.message, 'error');
    byId('cancel').disabled = false;
  }}
}});
updateCommand();
syncDatasetControls();
applyLocale(locale);
showStep(1);
refreshRunStatus().then(() => {{
  if (!byId('cancel').disabled) startStatusPolling();
}});
</script>
</body>
</html>
"""


def _section(
    name: str,
    values: dict[str, Any],
    defaults: dict[str, Any],
    description: str | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": " ".join((description or "").split()),
        "fields": [
            {
                "name": key,
                "value": value,
                "default": defaults.get(key),
                "changed": value != defaults.get(key),
            }
            for key, value in values.items()
        ],
    }


def _preflight_warnings(config: RunConfig, config_path: Path) -> list[str]:
    warnings: list[str] = []
    if not config.impls:
        warnings.append(
            "No phase implementations are pinned; runtime registry defaults will be used."
        )
    skill_impl_selected = any("skill" in impl for impl in config.impls.values())
    if skill_impl_selected and not config.skills_root.exists():
        warnings.append(
            f"Skill implementation selected but skills_root does not exist: {config.skills_root}"
        )
    if config.executor_timeout_s > config.timeout_s:
        warnings.append("executor_timeout_s exceeds the per-task timeout_s.")
    for field_name in ("api_docs_dir", "tasks_dir"):
        value = getattr(config.paths, field_name)
        if value is not None and not value.exists():
            warnings.append(f"paths.{field_name} does not exist: {value}")
    if not config_path.exists():
        warnings.append(f"Config file no longer exists: {config_path}")
    return warnings


def _set_nested(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = data
    for key in path[:-1]:
        target = target.setdefault(key, {})
    target[path[-1]] = value


def _apply_skills_mode(data: dict[str, Any], skills: str) -> None:
    if skills == "off":
        data["skills_library"] = "best"
        data.setdefault("impls", {})[Phase.PLAN.value] = "rough"
        data["impls"][Phase.EXECUTE.value] = "code_plan_execute"
        return
    native = skills.startswith("native_")
    data["skills_library"] = (
        "none" if skills == "thin" else skills.removeprefix("native_")
    )
    data.setdefault("impls", {})[Phase.PLAN.value] = (
        "rough_skill_native" if native else "rough_skill"
    )
    data["impls"][Phase.EXECUTE.value] = (
        "code_plan_execute_skill_native" if native else "code_plan_execute_skill"
    )


__all__ = [
    "apply_cli_overrides",
    "build_batch_command",
    "build_launcher_view",
    "build_view_model",
    "discover_config_presets",
    "render_config_html",
]
