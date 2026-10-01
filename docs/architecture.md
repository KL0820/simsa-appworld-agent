# Architecture

## Online execution

SIMSA has four logical roles: planning, retrieval, execution and progress control.
Execution contains a code planner and a code executor; this is not simply a
four-model-call pipeline.

| Role | Code | Responsibility |
|---|---|---|
| Planning | `subagents/planner/rough_planner/` | Create the initial milestone list |
| Retrieval | `subagents/finder/` | Narrow API candidates and expand dependencies |
| Execution | `subagents/executor/code_plan_execute/` | Build a code plan, execute in the sandbox, return typed results |
| Progress control | `subagents/continuer/continuation/` | Propose retry, forward revision, advancement, submission or abort |

Paths above are relative to `src/adk_appworld_agent/`.

`AppWorldAgent` is an ADK Custom Agent. Its `Orchestrator` provides deterministic
state storage, worker invocation, completion checks and ledger operations; it
is not another LLM decision maker. The historical `PLAN` phase invokes the
initial planner on cycle 0 and the progress-control agent on subsequent cycles.
The thesis's `NEXT` action corresponds to the implementation's `ADVANCE`.

### Boundaries that matter

- Workers receive typed inputs and return validated envelopes. They do not
  directly commit controller state.
- Each milestone produces named values for downstream milestones.
- Progress decisions pass through framework guards: valid submission,
  bounded plan revisions, stable milestone identities and cycle limits.
- Model completion, accepted submission and evaluator success are distinct.
- Provider failures, program errors and lack of task progress require different
  recovery paths. Retries are bounded; a failure is not silently converted into success.

### Tradeoffs

Role separation makes components easier to replace, inspect and compare, but
adds model calls and data handoff. A retrieval miss can affect execution; a
successful API call can still fail the user's task. End-to-end evaluation is
therefore necessary alongside component tests.

## Offline skill learning

`src/mind_skill/loop.py` coordinates the learning loop:

1. Extract a component's context and behavior from a successful training trajectory.
2. Induce a reusable skill.
3. Deduce/reconstruct behavior using that skill.
4. Evaluate outcome, reconstruction and skill quality.
5. Use textual feedback to refine the induction prompt and retain the best candidate.
6. Consolidate overlapping skills and refine names/descriptions for retrieval.

Planner deduction and full sandbox outcome evaluation are not interchangeable.
The method adapts MIND-Skill; it is prompt/skill optimization, not model fine-tuning.

At runtime, the reference configuration selects up to three skills for each of
the three skill-enabled components at task scope. Selected text is injected into
thin instructions. Embedding selection and native ADK skill loading are alternate
experiment paths. Native names may receive in-memory aliases for ADK compatibility;
the released skill files are unchanged.

## Reading and testing

Start with `_run_one_cycle`, then its phase helpers and `progress_policy.py`.
Use `tests/test_cycle_loop.py`, `tests/test_completion_gate.py` and
`tests/test_scenario_goal_completion.py` to inspect state and evaluation contracts.
The analysis viewers and some research routines are larger; they are secondary
to the runtime and learning-loop entry points.
