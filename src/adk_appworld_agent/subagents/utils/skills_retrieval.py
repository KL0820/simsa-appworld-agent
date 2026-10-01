"""Held-out skill retrieval (MIND-Skill eval, spec §12 / paper B.3).

On train tasks the skill library is indexed by task_id; on HELD-OUT tasks the
runner retrieves top-K skills per component instead: an LLM call sees the task
instruction plus every library skill's (name, description) — the designated
retrieval keys — and picks the K most relevant. The chosen skills are injected
for the WHOLE task (composite-per-task) and the choice is persisted for
reproducibility.

Retrieval model = gemini-2.5-flash (self-generation alignment), temp 0/seed 123.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

from adk_appworld_agent.gemini_thinking import thinking_config_from_env
from adk_appworld_agent.orchestration.run_config import ModelConfig

DEFAULT_K = 3
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

# Loaded SentenceTransformer models, keyed by model name. The all-mpnet model is
# ~419M; cache it so a per-task resolve does not reload weights every call.
_EMBED_MODELS: dict = {}

RETRIEVAL_SYSTEM = """\
You select reusable procedural skills for an agent about to solve a task.
You are given the task instruction and a catalog of skills, each with a name
and a one-sentence description of when it applies.
Pick the {k} skills whose descriptions best match what this task will require.
Output: JSON with `skill_names` — exactly {k} names copied verbatim from the
catalog, most relevant first. No other text.
"""


class SkillSelection(BaseModel):
    skill_names: list[str] = Field(min_length=1)


def _frontmatter_field(text: str, field: str) -> str:
    for line in text.splitlines()[:8]:
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return ""


def load_library(library_dir: Path) -> list[dict]:
    """[{name, description, path, text}] for every SKILL.md under the library."""
    entries = []
    for skill_path in sorted(library_dir.glob("*/*/SKILL.md")):
        text = skill_path.read_text(encoding="utf-8")
        entries.append(
            {
                "name": _frontmatter_field(text, "name"),
                "description": _frontmatter_field(text, "description"),
                "path": str(skill_path),
                "text": text,
            }
        )
    return entries


def _build_retrieval_agent(model_cfg: ModelConfig, k: int) -> LlmAgent:
    return LlmAgent(
        name="skill_retrieval",
        model=model_cfg.name,
        description="Selects the top-K relevant skills for a task by description.",
        instruction=RETRIEVAL_SYSTEM.format(k=k),
        output_schema=SkillSelection,
        generate_content_config=types.GenerateContentConfig(
            temperature=model_cfg.temperature,
            top_p=model_cfg.top_p,
            top_k=model_cfg.top_k,
            candidate_count=1,
            seed=model_cfg.seed,
            thinking_config=thinking_config_from_env(),
        ),
    )


async def _select(
    agent: LlmAgent, instruction: str, entries: list[dict], k: int
) -> list[str]:
    catalog = "\n".join(f"- {e['name']}: {e['description']}" for e in entries)
    prompt = f"TASK_INSTRUCTION:\n{instruction}\n\nSKILL CATALOG:\n{catalog}"
    session_service = InMemorySessionService()
    session_id = f"retrieve_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name="skill_retrieval", user_id="retrieval", session_id=session_id
    )
    runner = Runner(
        agent=agent, session_service=session_service, app_name="skill_retrieval"
    )
    text = ""
    async for event in runner.run_async(
        user_id="retrieval",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
        if event.content and event.content.parts:
            chunk = "\n".join(
                p.text for p in event.content.parts if getattr(p, "text", None)
            )
            if chunk:
                text = chunk
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rstrip("`\n ")
    names = SkillSelection.model_validate(json.loads(cleaned)).skill_names
    by_name = {e["name"]: e for e in entries}
    return [n for n in names if n in by_name][:k]


def _embed_text(entry: dict) -> str:
    """The text embedded for one skill: the same (name, description) retrieval
    keys the LLM selector reads — so both rankers see identical signal."""
    name = entry.get("name", "") or ""
    description = entry.get("description", "") or ""
    return f"{name}: {description}".strip() if description else name


def _get_embed_model(model_name: str):
    model = _EMBED_MODELS.get(model_name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _EMBED_MODELS[model_name] = model
    return model


def select_by_embedding(
    instruction: str,
    entries: list[dict],
    k: int,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[str]:
    """Top-K skill names by cosine similarity of each skill's (name: description)
    against the task instruction. Local, deterministic — no provider call, no
    429. Ranking is over the SAME catalog text the LLM selector reads."""
    model = _get_embed_model(model_name)
    catalog = model.encode(
        [_embed_text(e) for e in entries],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query = model.encode(
        [instruction], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    scores = catalog @ query  # normalized embeddings -> dot product == cosine
    order = sorted(range(len(entries)), key=lambda i: float(scores[i]), reverse=True)
    return [entries[i]["name"] for i in order[:k]]


async def retrieve_skills_async(
    *,
    instruction: str,
    library_dir: Path,
    model_cfg: ModelConfig,
    k: int = DEFAULT_K,
    selector: str = "llm",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    """Top-K library entries for a task instruction.

    Rate-limit errors (429 / RESOURCE_EXHAUSTED) get the shared provider
    ladder (waits granted as grace) — quota interference must
    not change which skills get retrieved. After the budget the error
    raises and the task fails cleanly (rerun later). Other provider transients
    use the same waiting schedule but preserve their generic failure contract.
    """
    entries = load_library(library_dir)
    if not entries:
        return []
    if len(entries) <= k:
        return entries
    by_name = {e["name"]: e for e in entries}

    if selector == "embedding":
        # Local, deterministic ranker — no provider call, so it bypasses the
        # rate-limit ladder below entirely.
        names = select_by_embedding(instruction, entries, k, embedding_model)
        return [by_name[n] for n in names]

    from adk_appworld_agent.orchestration.active_config import active_config
    from adk_appworld_agent.orchestration.rate_limit_grace import grant_rate_limit_grace
    from adk_appworld_agent.subagents.utils.retry import (
        RateLimitExhausted,
        is_rate_limit,
        is_retryable,
    )

    agent = _build_retrieval_agent(model_cfg, k)
    attempt = 0
    provider_delays = active_config().retry.provider_backoff_delays
    while True:
        try:
            names = await _select(agent, instruction, entries, k)
            break
        except Exception as exc:
            if is_rate_limit(str(exc)):
                if attempt >= len(provider_delays):
                    raise RateLimitExhausted(str(exc)) from exc
                delay = provider_delays[attempt]
                sys.stderr.write(
                    f"[skill_retrieval] rate-limited ({type(exc).__name__}), "
                    f"waiting {delay:.0f}s "
                    f"({attempt + 1}/{len(provider_delays)})\n"
                )
                sys.stderr.flush()
                grant_rate_limit_grace(delay)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            if attempt >= len(provider_delays) or not is_retryable(str(exc)):
                raise
            delay = provider_delays[attempt]
            sys.stderr.write(
                f"[skill_retrieval] transient ({type(exc).__name__}), "
                f"retry {attempt + 1}/{len(provider_delays)} in {delay:.0f}s\n"
            )
            sys.stderr.flush()
            await asyncio.sleep(delay)
            attempt += 1
    return [by_name[n] for n in names]


def retrieve_skills(
    *,
    instruction: str,
    library_dir: Path,
    model_cfg: ModelConfig,
    k: int = DEFAULT_K,
    selector: str = "llm",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict]:
    """Sync wrapper (tests / offline manifest building)."""
    return asyncio.run(
        retrieve_skills_async(
            instruction=instruction,
            library_dir=library_dir,
            model_cfg=model_cfg,
            k=k,
            selector=selector,
            embedding_model=embedding_model,
        )
    )


__all__ = [
    "DEFAULT_K",
    "DEFAULT_EMBEDDING_MODEL",
    "RETRIEVAL_SYSTEM",
    "load_library",
    "select_by_embedding",
    "retrieve_skills",
    "retrieve_skills_async",
]
