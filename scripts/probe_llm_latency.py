"""Probe Gemini latency before launching a smoke / batch.

Sends N (default 3) one-shot calls with an EXECUTE-shaped prompt, measures
per-call wall time, and prints a recommended `EXECUTOR_CODE_EXECUTE_TIMEOUT_S`
value sized for the observed latency.

Why this script exists: the executor's per-attempt timeout is a hard cap.
When Gemini latency drifts above ~1/5 of the cap, attempts start failing
with TimeoutError and the run looks like a code regression even when it
isn't. Run this script before each smoke session and export the
recommended env var.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/probe_llm_latency.py

    # then copy the printed `export ...` line into your shell.

Output is plain text; the recommended env var line is the last line so
shell substitution works:

    eval "$(PYTHONPATH=src .venv/bin/python scripts/probe_llm_latency.py | tail -1)"

Cost: ~3 calls × ~3K prompt tokens each = ~$0.001 on gemini-2.5-flash.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
import uuid

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_appworld_agent.orchestration.run_config import model_config_from_env
from adk_appworld_agent.repo_paths import repo_env_path

# Realistic EXECUTE-phase prompt size (~2.5K tokens).
_PROMPT_BODY = """You are the AppWorld code executor. Call execute_python EXACTLY ONCE
with a multi-line Python program that solves the milestone below.

Milestone: print the integer 42. The single tool call should be a print statement.

Available APIs (illustrative — many fields below are intentionally verbose so the
prompt size matches production EXECUTE prompts):
- venmo.search_friends(query, user_email, page_index, page_limit) — paginated,
  returns first_name, last_name, email, registered_at, friends_since per item.
- phone.search_contacts(query, relationship, page_index, page_limit) — paginated,
  returns first_name, last_name, phone_number, email, relationship, address per item.
- spotify.show_current_song() — returns song_id, title, album_title, artists list,
  duration_ms, is_playing, queue_position, queue_length.
- spotify.show_downloaded_songs(query, page_index, page_limit) — paginated, returns
  song_id, title, album_title, artists list, downloaded_at.
- splitwise.show_friends(page_index, page_limit) — paginated, returns user_id, name,
  email, balance per item; cents-precision.
- amazon.show_orders(status, min_created_at, max_created_at, page_index, page_limit)
  — paginated, returns order_id, status, items, total_cents, shipping_address.
- gmail.show_emails(query, label, min_received_at, max_received_at, page_index,
  page_limit) — paginated, returns email_id, sender, subject, body_preview, received_at.
- todoist.show_projects(page_index, page_limit) — returns project_id, name,
  is_archived, item_count per item.
- todoist.show_tasks_in_project(project_id, is_completed, page_index, page_limit) —
  paginated, returns task_id, content, due_at, is_completed, comment_count per item.
- file_system.list(path) — returns name, type ("file"|"folder"), size_bytes,
  modified_at per entry; non-recursive.
- file_system.read(path) — returns full file contents as a single string.
- simple_note.show_notes(query, tag, page_index, page_limit) — paginated, returns
  note_id, title, body, tags, created_at, modified_at per item.

Run the milestone now by writing the smallest possible Python that satisfies the goal.
"""


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = pct / 100.0 * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def _recommended_timeout_s(
    walls: list[float], *, safety_factor: float, floor: float
) -> float:
    if not walls:
        return floor
    p95 = _percentile(walls, 95)
    return max(p95 * safety_factor, floor)


async def _one_probe(
    model_name: str, generate_config: types.GenerateContentConfig
) -> float:
    agent = LlmAgent(
        name="latency_probe",
        model=model_name,
        description="Disposable latency probe; not used in production runs.",
        instruction="Reply with the single word OK.",
        generate_content_config=generate_config,
    )
    session_service = InMemorySessionService()
    app_name = "latency_probe"
    user_id = "probe"
    session_id = f"probe_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    runner = Runner(agent=agent, session_service=session_service, app_name=app_name)
    started = time.monotonic()
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=_PROMPT_BODY)]),
    ):
        pass
    return time.monotonic() - started


async def _run_probes(n: int, model_cfg) -> list[float]:
    config = types.GenerateContentConfig(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k,
        candidate_count=model_cfg.candidate_count,
        seed=model_cfg.seed,
        max_output_tokens=model_cfg.max_output_tokens or None,
    )
    walls: list[float] = []
    for i in range(n):
        wall = await _one_probe(model_cfg.name, config)
        walls.append(wall)
        print(f"probe {i + 1}/{n}: {wall:.2f}s", file=sys.stderr, flush=True)
    return walls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3, help="number of probe calls")
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=5.0,
        help="multiplier applied to observed p95 latency to set the recommended "
        "timeout. Default 5x leaves headroom for tail-latency spikes during the "
        "actual run.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=90.0,
        help="minimum recommended timeout. Default matches the production "
        "default; the script will never recommend a value below this.",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=repo_env_path())
    model_cfg = model_config_from_env()
    print(f"model={model_cfg.name} probes={args.n}", file=sys.stderr)

    walls = asyncio.run(_run_probes(args.n, model_cfg))
    if not walls:
        print("no probe data collected", file=sys.stderr)
        return 1

    median = statistics.median(walls)
    p95 = _percentile(walls, 95)
    recommended = _recommended_timeout_s(
        walls, safety_factor=args.safety_factor, floor=args.floor
    )

    print(
        f"latency median={median:.2f}s p95={p95:.2f}s "
        f"safety_factor={args.safety_factor:g}x floor={args.floor:g}s",
        file=sys.stderr,
    )

    # Last line on stdout is the export — easy to eval-substitute.
    print(f"export EXECUTOR_CODE_EXECUTE_TIMEOUT_S={recommended:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
