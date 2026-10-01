"""Embedding-based held-out skill selection (skills_selector="embedding").

The ranking math + routing are tested with a FAKE model (deterministic, no
419M load); one guarded test exercises the real all-mpnet model for a semantic
sanity check and skips if the model/dep is unavailable.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.utils import skills_retrieval
from adk_appworld_agent.subagents.utils.skills_retrieval import (
    retrieve_skills,
    select_by_embedding,
)
from adk_appworld_agent.subagents.utils.skills_source import SkillsSource


class _FakeModel:
    """Returns a fixed vector per exact input string (default = zero vector)."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        out = []
        for t in texts:
            arr = np.array(self._vectors.get(t, [0.0, 0.0, 0.0]), dtype=float)
            if normalize_embeddings:
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
            out.append(arr)
        return np.array(out)


def _entry(name: str, description: str) -> dict:
    return {"name": name, "description": description, "path": "", "text": f"#{name}"}


def _install_fake(monkeypatch, vectors: dict[str, list[float]]) -> None:
    model = _FakeModel(vectors)
    monkeypatch.setattr(skills_retrieval, "_get_embed_model", lambda _name: model)


def test_embedding_ranks_by_cosine_and_respects_k(monkeypatch):
    entries = [
        _entry("alpha", "do A"),
        _entry("beta", "do B"),
        _entry("gamma", "do C"),
    ]
    # query aligned with beta, then gamma, then alpha
    _install_fake(
        monkeypatch,
        {
            "alpha: do A": [1.0, 0.0, 0.0],
            "beta: do B": [0.0, 1.0, 0.0],
            "gamma: do C": [0.0, 0.6, 0.4],
            "QUERY": [0.0, 1.0, 0.0],
        },
    )
    names = select_by_embedding("QUERY", entries, k=2)
    assert names == ["beta", "gamma"]  # top-2 by cosine, beta closest


def test_embedding_is_deterministic(monkeypatch):
    entries = [_entry("a", "x"), _entry("b", "y"), _entry("c", "z")]
    vectors = {
        "a: x": [1.0, 0.0, 0.0],
        "b: y": [0.0, 1.0, 0.0],
        "c: z": [0.0, 0.0, 1.0],
        "Q": [0.9, 0.1, 0.0],
    }
    _install_fake(monkeypatch, vectors)
    first = select_by_embedding("Q", entries, k=3)
    _install_fake(monkeypatch, vectors)
    second = select_by_embedding("Q", entries, k=3)
    assert first == second == ["a", "b", "c"]


def _write_skill(library_dir: Path, src: str, name: str, description: str) -> None:
    skill_dir = library_dir / src / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"name: {name}\ndescription: {description}\n\n# {name}\nbody\n",
        encoding="utf-8",
    )


def test_retrieve_embedding_path_skips_llm(monkeypatch, tmp_path):
    library_dir = tmp_path / "code_executor" / "best"
    _write_skill(library_dir, "t1", "read-then-filter", "read records then filter")
    _write_skill(library_dir, "t2", "send-money", "transfer money between accounts")
    _write_skill(library_dir, "t3", "make-playlist", "create a music playlist")
    _write_skill(library_dir, "t4", "post-review", "post a review with a rating")

    _install_fake(
        monkeypatch,
        {
            "read-then-filter: read records then filter": [1.0, 0.0, 0.0],
            "send-money: transfer money between accounts": [0.0, 1.0, 0.0],
            "make-playlist: create a music playlist": [0.0, 0.0, 1.0],
            "post-review: post a review with a rating": [0.0, 0.0, 0.5],
            "create a spotify playlist of songs": [0.0, 0.0, 1.0],
        },
    )

    # If the embedding branch is correct, the LLM agent is never built.
    def _boom(*_a, **_k):
        raise AssertionError("LLM retrieval agent must not be built for embedding")

    monkeypatch.setattr(skills_retrieval, "_build_retrieval_agent", _boom)

    chosen = retrieve_skills(
        instruction="create a spotify playlist of songs",
        library_dir=library_dir,
        model_cfg=ModelConfig(),
        k=2,
        selector="embedding",
    )
    names = [c["name"] for c in chosen]
    assert names[0] == "make-playlist"
    assert len(names) == 2


def test_skills_source_embedding_resolves_heldout(monkeypatch, tmp_path):
    library_dir = tmp_path / "code_executor" / "best"
    _write_skill(library_dir, "t1", "read-filter", "read then filter records")
    _write_skill(library_dir, "t2", "send-money", "transfer money")
    _write_skill(library_dir, "t3", "make-playlist", "create a playlist")
    _write_skill(library_dir, "t4", "post-review", "post a rating review")

    _install_fake(
        monkeypatch,
        {
            "read-filter: read then filter records": [1.0, 0.0, 0.0],
            "send-money: transfer money": [0.0, 1.0, 0.0],
            "make-playlist: create a playlist": [0.0, 0.0, 1.0],
            "post-review: post a rating review": [0.0, 0.0, 0.2],
            "build me a playlist": [0.0, 0.0, 1.0],
        },
    )
    source = SkillsSource(
        component="code_executor",
        root=tmp_path,
        library="best",
        model_cfg=ModelConfig(),
        k=1,
        selector="embedding",
    )
    # held-out task_id (no per-task dir) -> retrieval path
    resolved = asyncio.run(source.resolve("held_out_99", "build me a playlist"))
    assert resolved.source == "retrieval"
    assert resolved.names == ["make-playlist"]


def test_catalog_at_or_below_k_returns_all_regardless_of_selector(
    monkeypatch, tmp_path
):
    library_dir = tmp_path / "code_executor" / "best"
    _write_skill(library_dir, "t1", "only-one", "the only skill")
    # embedding model must never be touched on the short-circuit
    monkeypatch.setattr(
        skills_retrieval,
        "_get_embed_model",
        lambda _n: (_ for _ in ()).throw(
            AssertionError("model loaded on short-circuit")
        ),
    )
    chosen = retrieve_skills(
        instruction="anything",
        library_dir=library_dir,
        model_cfg=ModelConfig(),
        k=3,
        selector="embedding",
    )
    assert [c["name"] for c in chosen] == ["only-one"]


def test_real_model_semantic_sanity():
    """Guarded: uses the real all-mpnet model; skips if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        pytest.skip("sentence-transformers not installed")
    entries = [
        _entry("transfer-money", "send money from one account to another"),
        _entry("create-playlist", "create a music playlist and add songs"),
        _entry("read-notes", "read and search personal notes"),
    ]
    try:
        names = select_by_embedding(
            "make a spotify playlist out of my favourite songs", entries, k=1
        )
    except Exception as exc:  # model weights missing / offline
        pytest.skip(f"embedding model unavailable: {exc}")
    assert names == ["create-playlist"]
