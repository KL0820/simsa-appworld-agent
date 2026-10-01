"""Orchestration tests for the two-stage curation pipeline (stubbed LLM/embedder).

No network/LLM/deduction: a StubEmbedder makes clustering deterministic (one-hot
per GROUP=XX marker), and stub resolver/verifier/sharpener drive the branches.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np

from mind_skill.curation.curate import (
    Sharpen,
    SharpenResolution,
    sharpen_to_convergence,
    similarity_clusters,
    stage2_sharpen_library,
)
from mind_skill.curation.dedup import (
    load_skill_files,
    render_skill_md,
    uniquify_component,
)


def _grp(text: str) -> str:
    m = re.search(r"GROUP=([A-Z]+)", text)
    return m.group(1) if m else "NONE"


class StubEmbedder:
    """One-hot per GROUP marker -> cosine 1.0 within group, 0.0 across."""

    def encode(self, texts, normalize_embeddings=True):
        idx: dict[str, int] = {}
        for t in texts:
            idx.setdefault(_grp(t), len(idx))
        dim = max(len(idx), 1)
        out = []
        for t in texts:
            v = np.zeros(dim)
            v[idx[_grp(t)]] = 1.0
            out.append(v)
        return np.array(out)


def _write(lib, task_id, name, desc, body):
    p = lib / task_id / name / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_skill_md(name, desc, body), encoding="utf-8")


def _fixture(tmp_path):
    L = tmp_path / "best"
    _write(L, "t1", "alpha", "GROUP=AB do x", "body alpha GROUP=AB")
    _write(L, "t2", "bravo", "GROUP=AB do x2", "body bravo GROUP=AB")
    _write(L, "t3", "charlie", "GROUP=CD do y", "body charlie GROUP=CD")
    _write(L, "t4", "delta", "GROUP=CD do y2", "body delta GROUP=CD")
    _write(L, "t5", "echo", "GROUP=EF do z", "body echo GROUP=EF")
    _write(L, "t6", "foxtrot", "GROUP=EF do z2", "body foxtrot GROUP=EF")
    _write(L, "t7", "golf", "GROUP=G alone", "body golf GROUP=G")
    return L


def test_similarity_clusters(tmp_path):
    skills = load_skill_files(_fixture(tmp_path))
    clusters = similarity_clusters(
        skills, kind="body", threshold=0.99, embedder=StubEmbedder()
    )
    got = sorted(sorted(s.task_id for s in c) for c in clusters)
    assert got == [["t1", "t2"], ["t3", "t4"], ["t5", "t6"]]  # golf(t7) alone


def test_stage2_sharpen(tmp_path):
    lib = _fixture(tmp_path)

    async def sharpener(skills, avoid):
        g = _grp(skills[0].description)
        return SharpenResolution(
            reason="separated by distinctive trigger",
            renames=[
                Sharpen(
                    new_name=f"{s.name}-{g.lower()}",
                    new_description=f"distinct for {s.task_id}",
                )
                for s in skills
            ],
        )

    actions = asyncio.run(
        stage2_sharpen_library(
            lib, threshold=0.99, sharpener=sharpener, embedder=StubEmbedder()
        )
    )
    names = {sf.name for sf in load_skill_files(lib)}
    # AB, CD, EF each a confusable PAIR by name+desc (GROUP in desc) -> 6 skills
    # sharpened in one pass; golf alone is never a candidate.
    assert len(actions) == 6
    assert "golf" in names  # alone -> untouched
    assert sum(1 for n in names if n.endswith(("-ab", "-cd", "-ef"))) == 6


def test_uniquify_component_renames_never_merges(tmp_path):
    """Induction post-assembly hygiene is deterministic rename-only (no LLM, no merge):
    same-name skills from different tasks become name, name-2, ...; every skill + body
    is preserved. (Semantic merging is the curation stage's job, not induction's.)"""
    comp = tmp_path / "code_executor"
    for task, name, body in [
        ("t1", "foo", "B1"),
        ("t2", "foo", "B2"),
        ("t3", "bar", "B3"),
    ]:
        p = comp / "best" / task / name / "SKILL.md"
        p.parent.mkdir(parents=True)
        p.write_text(render_skill_md(name, "d", body), encoding="utf-8")

    out = uniquify_component(comp)
    sk = load_skill_files(comp / "best")
    names = sorted(sf.name for sf in sk)
    assert len(sk) == 3  # nothing merged away
    assert names == ["bar", "foo", "foo-2"]  # deterministic rename
    assert len(set(names)) == 3  # unique
    assert sorted(sf.body for sf in sk) == ["B1", "B2", "B3"]  # bodies preserved
    assert out["libraries"]["best"] == [{"task_id": "t2", "from": "foo", "to": "foo-2"}]


def test_sharpen_to_convergence_settles_clique(tmp_path):
    """3 skills all in GROUP=AB => every pair has cosine 1.0 (a clique). The stub
    relabels but KEEPS the marker, so similarity never drops -- convergence must come
    from the settle-set (each task-id pair sharpened at most once). Mirrors Stage-1's
    independence-graph termination."""
    L = tmp_path / "best"
    _write(L, "t1", "alpha", "GROUP=AB do x", "BODY-A")
    _write(L, "t2", "bravo", "GROUP=AB do y", "BODY-B")
    _write(L, "t3", "charlie", "GROUP=AB do z", "BODY-C")
    bodies0 = sorted(sf.body for sf in load_skill_files(L))

    calls = {"n": 0}

    async def sharpener(skills, avoid):
        calls["n"] += 1
        return SharpenResolution(
            reason="kept similar (worst case)",
            renames=[
                Sharpen(
                    new_name=f"{s.task_id}-s{calls['n']}",
                    new_description=f"GROUP=AB v{calls['n']} {s.task_id}",
                )
                for s in skills
            ],
        )

    out = asyncio.run(
        sharpen_to_convergence(
            L, threshold=0.99, sharpener=sharpener, embedder=StubEmbedder()
        )
    )
    assert out["converged"] is True
    assert out["settled"] == 3  # all 3 clique pairs, once each
    assert sum(r["sharpened_count"] for r in out["rounds"]) == 3
    assert (
        out["rounds"][-1]["sharpened_count"] == 0
    )  # final round is the empty (converged) one
    sk = load_skill_files(L)
    assert len(sk) == 3  # count preserved
    assert sorted(sf.body for sf in sk) == bodies0  # bodies verbatim
    import json

    settled = json.loads((L / "sharpened_pairs.json").read_text())
    assert sorted(map(sorted, settled)) == [["t1", "t2"], ["t1", "t3"], ["t2", "t3"]]


def test_sharpen_to_convergence_separates(tmp_path):
    """One confusable pair (GROUP=AB) + one lonely skill. The stub gives each member a
    DISTINCT new marker, so after one sharpen their cosine drops to 0 and they leave
    the candidate set naturally -- converges in 2 rounds (1 sharpen + 1 empty)."""
    L = tmp_path / "best"
    _write(L, "t1", "alpha", "GROUP=AB do x", "BODY-A")
    _write(L, "t2", "bravo", "GROUP=AB do y", "BODY-B")
    _write(L, "t3", "charlie", "GROUP=Q lonely", "BODY-C")

    async def sharpener(skills, avoid):
        return SharpenResolution(
            reason="separated",
            renames=[
                Sharpen(
                    new_name=f"{s.task_id}-sep",
                    new_description=f"GROUP=SEP{chr(65 + i)} {s.task_id}",
                )
                for i, s in enumerate(skills)
            ],
        )

    out = asyncio.run(
        sharpen_to_convergence(
            L, threshold=0.99, sharpener=sharpener, embedder=StubEmbedder()
        )
    )
    assert out["converged"] is True
    assert out["settled"] == 1  # only the AB pair
    assert sum(r["sharpened_count"] for r in out["rounds"]) == 1
    names = {sf.name for sf in load_skill_files(L)}
    assert "charlie" in names  # lonely never a candidate
    assert {"t1-sep", "t2-sep"} <= names
