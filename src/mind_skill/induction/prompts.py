"""Initial induction prompts P_I^(0), one per skilled component (B-A, faithful to
MIND-Skill Fig 7a — arXiv:2605.08670).

These are the STARTING prompts; TextGrad refines each per (component, trajectory) during
the closed loop (Q≈2). Component-specific ONLY in: the Role sentence, the "Given:" line,
and rule 1's category list. Rules 2-3 + the schema-output line are identical across
components. Specific rules (list-vs-dict handling, distractor-robustness, ...) are NOT
written here — they are discovered by TextGrad from deduction failures (Fig 7b).

Each *_induction_agent is a gemini-2.5-flash LlmAgent with output_schema=InducedSkill
(see schemas.py); ADK appends the JSON-schema instruction, so the "Output:" line below is
reinforcement, not the format authority.
"""

from __future__ import annotations

CODE_EXECUTOR_INDUCTION_PROMPT = """\
You are an expert at extracting reusable procedural strategies from solved tasks. Given the \
milestone plan and the offered APIs, the EXECUTOR of a multi-subagent AppWorld agent writes \
the Python that performs each milestone. Extract a SKILL describing the procedure pattern — \
the structural "how-to" NOT obvious from the instruction alone.

Given: TASK_INSTRUCTION + EXECUTOR_TRAJECTORY (per milestone: the plan it was handed, the \
APIs offered, the Python it ran, what it returned, turns/repairs).

Rules for a good skill:
 1. Describe ONLY solving strategy and structural patterns: authentication/credential flow, \
pagination/iteration, multi-step data retrieval, data transformation, output construction, \
reading prior-milestone variables, the task-completion / submission contract.
 2. Do NOT include task-specific info: no specific API names, field names, entity names, \
thresholds. Test: if someone can guess the original task from your skill alone, too specific.
 3. Focus on NON-OBVIOUS structural knowledge.

Output: emit the skill as the structured fields of the output schema (name / description / \
overview / when_to_apply / procedure / key_patterns / common_pitfalls). Do NOT write markdown.
"""

CODE_PLANNER_INDUCTION_PROMPT = """\
You are an expert at extracting reusable procedural strategies from solved tasks. Given one \
milestone and the offered APIs, the CODE-PLANNER of a multi-subagent AppWorld agent writes an \
ordered plan of concrete steps the executor turns into Python. Extract a SKILL describing the \
procedure pattern NOT obvious from the instruction alone.

Given: TASK_INSTRUCTION + CODE_PLANNER_TRAJECTORY (per milestone: the milestone intent, the \
APIs offered, the step plan it produced).

Rules for a good skill:
 1. Describe ONLY solving strategy and structural patterns: selecting the API that matches the \
milestone action, sequencing calls (obtain a credential/key, then use it), what to extract \
from each response, pagination/iteration, reading prior-milestone variables, the output the \
executor must produce.
 2. Do NOT include task-specific info: no specific API names, field names, entity names, \
thresholds. Test: if someone can guess the original task from your skill alone, too specific.
 3. Focus on NON-OBVIOUS structural knowledge.

Output: emit the skill as the structured fields of the output schema (name / description / \
overview / when_to_apply / procedure / key_patterns / common_pitfalls). Do NOT write markdown.
"""

ROUGH_PLANNER_INDUCTION_PROMPT = """\
You are an expert at extracting reusable procedural strategies from solved tasks. The PLANNER \
of a multi-subagent AppWorld agent decomposes a task into an ordered list of milestones, each \
later retrieved-for and executed independently. Extract a SKILL describing the decomposition \
pattern NOT obvious from the instruction alone.

Given: TASK_INSTRUCTION + PLANNER_OUTPUT (the milestone decomposition it produced, with rationale).

Rules for a good skill:
 1. Describe ONLY decomposition strategy and structural patterns: resolving identities/credentials \
before acting on them, reading a value before using it downstream, ordering milestones by data \
dependency, separating a read step from its action step, placing a constraint before the \
iteration it bounds.
 2. Do NOT include task-specific info: no specific app names, entity names, field names, \
thresholds. Test: if someone can guess the original task from your skill alone, too specific.
 3. Focus on NON-OBVIOUS structural knowledge.

Output: emit the skill as the structured fields of the output schema (name / description / \
overview / when_to_apply / procedure / key_patterns / common_pitfalls). Do NOT write markdown.
"""

# component key (matches make_induction_slices.py slice keys) -> initial induction prompt
INDUCTION_PROMPTS: dict[str, str] = {
    "code_executor": CODE_EXECUTOR_INDUCTION_PROMPT,
    "code_planner": CODE_PLANNER_INDUCTION_PROMPT,
    "rough_planner": ROUGH_PLANNER_INDUCTION_PROMPT,
}
