from __future__ import annotations

from pathlib import Path

from adk_appworld_agent.config_viewer import (
    apply_cli_overrides,
    build_batch_command,
    build_launcher_view,
    build_view_model,
    discover_config_presets,
    render_config_html,
)
from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.orchestration.state import Phase


def test_view_model_marks_only_overrides(tmp_path: Path):
    config = RunConfig(
        model={"temperature": 0.3},
        impls={Phase.FIND: "community", Phase.PLAN: "rough_skill"},
    )
    path = tmp_path / "experiment.json"
    path.write_text(config.to_json(), encoding="utf-8")

    view = build_view_model(
        config,
        config_path=path,
        experiment_name="smoke",
        dataset="test_normal",
        variant="2",
    )

    model = next(section for section in view["sections"] if section["name"] == "model")
    temperature = next(
        field for field in model["fields"] if field["name"] == "temperature"
    )
    seed = next(field for field in model["fields"] if field["name"] == "seed")
    assert temperature["changed"] is True
    assert seed["changed"] is False
    assert "--dataset test_normal --variant 2" in view["command"]


def test_render_is_standalone_and_escapes_embedded_payload(tmp_path: Path):
    config = RunConfig(cache={"version": "</script><script>alert(1)</script>"})
    view = build_view_model(
        config,
        config_path=tmp_path / "x.json",
        experiment_name="x",
    )

    page = render_config_html(view)

    assert "<!doctype html>" in page
    assert "</script><script>alert(1)</script>" not in page
    assert "<\\/script><script>alert(1)<\\/script>" in page


def test_batch_command_appends_extra_task_ids(tmp_path: Path):
    command = build_batch_command(
        tmp_path / "a config.json",
        experiment_name="probe",
        dataset="quick_smoke",
        extra_task_ids="a_2,b_2",
    )
    assert "--dataset quick_smoke --variant full" in command
    assert "--extra_task_ids a_2,b_2" in command
    assert "--split" not in command
    assert f"'{tmp_path / 'a config.json'}'" in command


def test_launcher_cli_overrides_preset_values():
    config, sources = apply_cli_overrides(
        RunConfig(
            model={"name": "preset-model", "temperature": 0.1},
            impls={Phase.PLAN: "rough", Phase.EXECUTE: "code_plan_execute"},
        ),
        {
            "model_name": "cli-model",
            "temperature": 0.7,
            "candidate_count": 2,
            "skills": "best",
        },
    )

    assert config.model.name == "cli-model"
    assert config.model.temperature == 0.7
    assert config.model.candidate_count == 2
    assert config.impls[Phase.PLAN] == "rough_skill"
    assert config.impls[Phase.EXECUTE] == "code_plan_execute_skill"
    assert sources["model.name"] == "cli"


def test_launcher_page_contains_editable_start_flow(tmp_path: Path):
    presets = discover_config_presets(tmp_path)
    view = build_launcher_view(
        RunConfig(),
        config_path=None,
        presets=presets,
        experiment_name="smoke",
        dataset="test_normal",
        variant="2",
        extra_task_ids="target_3",
        rpc_url="tcp://127.0.0.1:4242",
        log_root="logs",
        offset=0,
        limit=2,
    )

    page = render_config_html(view)

    assert view["batch"]["dataset"] == "test_normal"
    assert view["batch"]["variant"] == "2"
    assert view["batch"]["extra_task_ids"] == "target_3"
    assert (
        "--dataset test_normal --variant 2 --extra_task_ids target_3" in view["command"]
    )
    assert "AppWorld run launcher" in page
    assert 'data-step="1"' in page
    assert 'data-step="2" hidden' in page
    assert 'class="panel wide task-panel"' in page
    assert 'class="panel wide controller-panel"' in page
    assert '<select id="find_impl">' in page
    assert '<select id="plan_impl">' in page
    assert '<select id="execute_impl">' in page
    assert '<input id="find_impl">' not in page
    for field_id in (
        "model_name",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "candidate_count",
        "max_output_tokens",
    ):
        assert f'id="{field_id}"' in page
    assert "0 = provider default (no explicit limit)" in page
    assert "model-row" in page
    assert "overflow-x:hidden" in page
    assert 'id="next"' in page
    assert 'id="back"' in page
    assert 'id="launch"' in page
    assert 'id="cancel"' in page
    assert "/api/validate" in page
    assert "/api/launch" in page
    assert "/api/cancel" in page
    assert "refreshRunStatus()" in page
    assert 'id="selection_summary"' in page
    assert "Base: ${baseCount} tasks" in page
    assert "deduplicated against the base" in page
    assert "page edits &gt; launcher CLI" not in page  # payload is encoded as JSON
    assert 'data-locale="zh"' in page
    assert 'data-locale="en"' in page
    assert "button.className = 'info-button'" in page
    assert "installFieldHelp();" in page
    assert "更多說明" in page
    assert "More information" in page

    visible_controls = {
        "preset",
        "model_name",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "candidate_count",
        "max_output_tokens",
        "experiment_name",
        "dataset",
        "variant",
        "extra_task_ids",
        "offset",
        "limit",
        "log_root",
        "rpc_url",
        "find_impl",
        "plan_impl",
        "execute_impl",
        "timeout_s",
        "executor_timeout_s",
        "attempt_budget",
        "skills_library",
        "skills_prompt_variant",
        "skills_root",
        "keep_debug",
        "executor_self_assess",
        "config_json",
    }
    assert visible_controls <= set(view["field_help"])
    for field_id in visible_controls:
        assert view["field_help"][field_id]["en"]
        assert view["field_help"][field_id]["zh"]
