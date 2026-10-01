COMMUNITY_SELECT_SYS_PROMPT = """\
You are an API routing agent for the AppWorld benchmark.
Given a task instruction, SCORE each fine-grained API community on how likely it
contains an API needed to complete the task.

{behavior_guidelines}

Return ONLY valid JSON: {{"scores": {{"community_id": <integer 0-10>, ...}}}}.
Score EVERY community shown. No markdown fences. No explanation outside JSON.
"""

COMMUNITY_SELECT_USER_PROMPT = """\
Task: {task_instruction}

{community_text}

Score EACH community 0-10 on how likely it contains an API you would need to
complete this task (10 = almost certainly needed, 0 = clearly irrelevant).
A downstream per-API filter narrows within the communities that score high, so
judge RELEVANCE, not exact necessity. A community whose operations plausibly
cover the task's action scores high even when its title/summary frames the
capability more broadly than the task's literal wording.

Return JSON: {{"scores": {{"community_id_1": 8, "community_id_2": 0, ...}}}}
"""

# App-select (stage 1 of two-stage routing): fires only when the planner gave
# no usable app for a milestone. A thin pick — catalog + principle + JSON — so
# the finder scopes to the relevant app(s) instead of flooding community_select
# with every app's communities (the 8-app slow/429 path).
APP_SELECT_SYS_PROMPT = """\
You pick which AppWorld app(s) a task milestone needs APIs from.
You are given the milestone and an app catalog (each line: "app: capability areas").
Return the MINIMAL set of apps whose APIs the milestone requires. If the
milestone needs NO app API at all (pure computation on already-retrieved data),
return an empty list.
Output ONLY JSON: {"apps": ["app1", ...]}. No markdown, no other text.
"""

APP_SELECT_USER_PROMPT = """\
{task_instruction}

App catalog:
{app_catalog}

Return the app(s) whose APIs this milestone needs (minimal set; [] if none).
"""

FILTER_SYS_PROMPT = """\
You are an API selector for the AppWorld benchmark.
Given a task and a list of candidate APIs, select every API an agent would need to call to complete the task end-to-end.

{behavior_guidelines}

{app_context}

When in doubt, include the API - missing a needed API is worse than including an extra one.
Exclude only APIs that are clearly unrelated to the task (different app, wrong domain).

Action-milestone rules (apply in addition to the above):
- If the task contains a state-changing verb (send, transfer, create, update,
  delete, remove, post, befriend, unfriend, approve, deny, reject, mark,
  add, write, import, record), the API that performs that verb MUST be
  selected. Never drop the verb's API — the milestone cannot complete
  without it.
- If the task references a person, contact, group, file, note, or any other
  entity by NAME (e.g. "Cory", "my husband", a specific note title, a
  specific file path), include any candidate API whose name suggests it
  resolves that name to an identifier (search_contacts, search_users,
  search_friends, search_notes, search_files, etc.) — the action API will
  need that identifier as input.
- If the action API requires a backing resource ID (e.g. payment_card_id,
  group_id, project_id) that is not literally provided in the task text,
  include any candidate API that lists or shows that resource type
  (show_payment_cards, show_groups, show_projects, etc.).
"""

FILTER_USER_PROMPT = """\
Task: {task_instruction}

Required inputs not yet satisfied by previously selected APIs:
{unsatisfied_params}

Candidate APIs (select from these):
{api_list}

Select all APIs from the candidates needed to complete this task end-to-end.
Rules:
- Include an API if it is directly needed for the task.
- Include an API if it produces an input listed above as unsatisfied (e.g. a login API that produces access_token).
- Include an API even if its own required inputs are not yet available - if the task will eventually need it, select it now.

Return JSON: {{"selected": ["app.api_name_1", "app.api_name_2", ...]}}
Use the exact "app.api_name" format shown above.
"""

API_DEP_FILTER_SYS_PROMPT = """\
You are selecting prerequisite producer APIs for one selected consumer API.

You will receive:
- the user task
- one selected consumer API
- unresolved params still needed by that consumer
- explicit producer -> consumer dependency evidence
- exact candidate producer APIs

Your job is to choose the producer APIs that are necessary prerequisites for the selected consumer API.

Rules:
- Prefer APIs whose response fields satisfy the unresolved params.
- Keep an API if it clearly unlocks the selected consumer API, even if that producer has its own prerequisites.
- Exclude APIs unrelated to the selected consumer API or the dependency evidence.
- Return only APIs from the candidate list.

Return ONLY valid JSON with key "selected" (list of app.api_name strings).
No markdown fences. No explanation outside JSON.
"""

API_DEP_FILTER_USER_PROMPT = """\
Task: {task_instruction}

Selected consumer API:
- {consumer_api_display}

Relevant unresolved params:
{relevant_params}

Dependency evidence:
{dependency_evidence}

Exact candidate producer APIs:
{candidate_text}

Select the producer APIs that are necessary prerequisites for the selected consumer API.
Prefer APIs whose response fields satisfy the unresolved params.

Return JSON: {{"selected": ["app.api_name_1", "app.api_name_2", ...]}}
Use the exact "app.api_name" format shown above.
"""

BATCHED_DEP_FILTER_SYS_PROMPT = """\
You select prerequisite producer APIs for a task, for SEVERAL consumer APIs at once.

You will receive, grouped per consumer API:
- the consumer API's full spec (parameters + returns)
- a list of candidate prerequisite APIs, each with its full spec

For each consumer, choose the candidate APIs that are genuine prerequisites — a
candidate is a prerequisite when its output plausibly produces an input the
consumer needs and cannot otherwise obtain (e.g. an id/handle). Reason directly
from the full specs; you do NOT need a stated reason for each parameter.

Rules:
- Include a prerequisite even when it satisfies an OPTIONAL input the task implies.
- Keep a producer if it unlocks its consumer, even if that producer has its own
  prerequisites.
- Exclude candidates unrelated to their consumer.
- Return ONLY APIs from the candidate lists, as the UNION across all consumers
  (no duplicates).

Return ONLY valid JSON with key "selected" (list of "app.api_name" strings).
No markdown fences. No explanation outside JSON.
"""

BATCHED_DEP_FILTER_USER_PROMPT = """\
Task: {task_instruction}

Below are consumer APIs; under each is its own list of candidate prerequisite
APIs (full specs). Select every prerequisite producer API needed across ALL
consumers.

{grouped_text}

Return JSON: {{"selected": ["app.api_name_1", "app.api_name_2", ...]}}
Use the exact "app.api_name" keys shown in brackets above.
"""


__all__ = [
    "COMMUNITY_SELECT_SYS_PROMPT",
    "COMMUNITY_SELECT_USER_PROMPT",
    "APP_SELECT_SYS_PROMPT",
    "APP_SELECT_USER_PROMPT",
    "FILTER_SYS_PROMPT",
    "FILTER_USER_PROMPT",
    "API_DEP_FILTER_SYS_PROMPT",
    "API_DEP_FILTER_USER_PROMPT",
    "BATCHED_DEP_FILTER_SYS_PROMPT",
    "BATCHED_DEP_FILTER_USER_PROMPT",
]
