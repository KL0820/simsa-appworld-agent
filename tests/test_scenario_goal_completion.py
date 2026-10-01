from adk_appworld_agent.observability.markdown import (
    _aggregate_table_lines,
    _scenario_goal_completion,
)


def _task(task_id, passed=True):
    return {"task_id": task_id, "result": {"passed_all": passed}}


def test_single_variant_cannot_establish_scenario_success():
    tasks = [_task("example_2")]
    assert _scenario_goal_completion(tasks) == (0, 0)
    assert "N/A (requires all three variants" in "\n".join(
        _aggregate_table_lines(tasks, total_tasks=1)
    )


def test_scenario_requires_every_variant_to_pass():
    tasks = [_task(f"first_{n}") for n in (1, 2, 3)]
    tasks += [_task(f"second_{n}", passed=n != 2) for n in (1, 2, 3)]
    assert _scenario_goal_completion(tasks) == (1, 2)


def test_partial_scenario_does_not_disappear_from_the_denominator():
    tasks = [_task(f"first_{n}") for n in (1, 2, 3)] + [_task("second_2")]
    assert _scenario_goal_completion(tasks) == (0, 0)


def test_duplicate_variant_is_not_a_complete_scenario():
    assert _scenario_goal_completion([_task("first_2")] * 3) == (0, 0)


def test_pending_tasks_make_the_aggregate_sgc_unavailable():
    tasks = [_task(f"first_{n}") for n in (1, 2, 3)]
    assert "N/A (requires all three variants" in "\n".join(
        _aggregate_table_lines(tasks, total_tasks=6)
    )
