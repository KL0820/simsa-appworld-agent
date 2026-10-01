"""Engine tests for the independence-graph multi-round merge driver (stubbed).

Covers the pieces that make multi-round + (_3) incremental curation correct and
cheap, all without network/LLM/deduction:
  - merge_nodes inherits the union of both parents' independence edges (A⊥C and
    A+B->AB  =>  AB⊥C), so a composite never re-merges what a part rejected.
  - reconcile drops edges whose endpoints changed (cross-session staleness).
  - the round loop blocks inherited-independent pairs and converges.
  - ExecutorMergeVerifier reuses its eval cache (no re-deduction) and enforces the
    requirement-count-no-decrease criterion, updating the live baseline on accept.
  - Coverage unions source tasks so a merge is verified against grandparent tasks.

A StubVecEmbedder reads "VEC=a;b;c" from each skill's text and returns the (unit)
vector, so pairwise cosines are exact and controllable.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np

from mind_skill.curation.curate import (
    Coverage,
    ExecutorMergeVerifier,
    IndependenceGraph,
    _md_sha,
    curate_merge_to_convergence,
)
from mind_skill.curation.dedup import (
    CollisionResolution,
    SkillFile,
    load_skill_files,
    render_skill_md,
)


def _sf(task_id: str, name: str, body: str = "") -> SkillFile:
    from pathlib import Path

    return SkillFile(
        task_id=task_id, name=name, description="d", body=body, path=Path("/x")
    )


class StubVecEmbedder:
    """Parse 'VEC=a;b;c' from each text -> unit vector (exact cosines)."""

    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        dim = 0
        parsed = []
        for t in texts:
            m = re.search(r"VEC=([0-9.;\-]+)", t)
            v = [float(x) for x in m.group(1).split(";")] if m else [0.0]
            parsed.append(v)
            dim = max(dim, len(v))
        for v in parsed:
            v = v + [0.0] * (dim - len(v))
            arr = np.array(v, dtype=float)
            n = np.linalg.norm(arr)
            vecs.append(arr / n if n else arr)
        return np.array(vecs)


# ── unit: graph inheritance / reconcile ────────────────────────────────────────
def test_graph_merge_inherits_edges():
    g = IndependenceGraph()
    a, b, c = _sf("t1", "alpha"), _sf("t2", "beta"), _sf("t3", "gamma")
    g.add_edge(a, c, kind="distinct", reason="diff", round_no=1)  # A ⊥ C
    assert g.blocked(a, c)
    ab = _sf("t1", "ab", body="merged")  # A + B -> AB
    g.merge_nodes(a, b, ab)
    assert g.blocked(ab, c)  # inherited: AB ⊥ C
    assert not g.blocked(a, c)  # A is gone as a node


def test_graph_reconcile_drops_stale():
    g = IndependenceGraph()
    a, c = _sf("t1", "alpha", body="x"), _sf("t3", "gamma", body="y")
    g.add_edge(a, c, kind="distinct", reason="diff", round_no=1)
    # a's content changed -> its edge is stale and must be dropped
    a2 = _sf("t1", "alpha", body="CHANGED")
    g.reconcile([a2, c])
    assert not g.edges


def test_coverage_union():
    cov = Coverage()
    a, b = _sf("t1", "alpha"), _sf("t2", "beta")
    assert cov.tasks_of(a) == {"t1"}
    ab = _sf("t1", "ab")
    cov.set_merged(ab, cov.tasks_of(a) | cov.tasks_of(b))
    assert cov.tasks_of(ab) == {"t1", "t2"}  # grandparent tasks preserved


# ── integration: inheritance blocks a pair across rounds, loop converges ───────
def _write(lib, task_id, name, body):
    p = lib / task_id / name / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_skill_md(name, "d", body), encoding="utf-8")


def test_engine_inheritance_blocks_across_rounds(tmp_path):
    lib = tmp_path / "best"
    # cos(alpha,gamma)=0.9 (distinct), cos(alpha,beta)=0.85 (merge), cos(beta,gamma)=0.765 (<0.8)
    _write(lib, "t1", "alpha", "VEC=1;0;0")
    _write(lib, "t3", "gamma", "VEC=0.9;0.43589;0")
    _write(lib, "t2", "beta", "VEC=0.85;0;0.52678")

    async def resolver(name, skills, avoid):
        names = {s.name for s in skills}
        if names == {"alpha", "beta"}:
            return CollisionResolution(
                verdict="merge",
                reason="same",
                merged_name="ab",
                merged_description="d",
                merged_body="VEC=0.9;0.43589;0\nmerged",  # = gamma direction
            )
        return CollisionResolution(verdict="distinct", reason="different")

    async def verifier(merged_md, source_task_ids):
        return True, {"accept": True, "source_task_ids": source_task_ids}

    out = asyncio.run(
        curate_merge_to_convergence(
            lib,
            threshold=0.8,
            resolver=resolver,
            verifier=verifier,
            embedder=StubVecEmbedder(),
            state_dir=tmp_path / "state",
            max_rounds=5,
        )
    )
    names = {sf.name for sf in load_skill_files(lib)}
    assert names == {"ab", "gamma"}  # alpha+beta merged; gamma kept
    assert out["converged"]
    r1, r2 = out["rounds"][0], out["rounds"][1]
    assert r1["merged_count"] == 1 and len(r1["distinct"]) == 1
    # round 2: the only candidate (ab, gamma) is inherited-blocked, not re-judged
    assert r2["merged_count"] == 0
    assert any(set(p[:2]) == {"t1:ab", "t3:gamma"} for p in r2["skipped_blocked"])
    # coverage unioned the merge's source tasks (grandparent preserved)
    import json

    cov = json.loads((tmp_path / "state" / "provenance.json").read_text("utf-8"))
    assert set(cov["t1:ab"]) == {"t1", "t2"}


# ── verifier: eval-cache reuse + count criterion + baseline update ─────────────
def _verifier(tmp_path):
    return ExecutorMergeVerifier(
        runs_dir=tmp_path,
        model_cfg=None,
        rpc_url="x",
        work_dir=tmp_path / "w",
        baseline_cache_path=tmp_path / "base.json",
        merge_eval_cache_path=tmp_path / "eval.json",
        audit_path=tmp_path / "audit.json",
    )


def test_verifier_cache_hit_skips_deduction(tmp_path):
    v = _verifier(tmp_path)
    v.base = {"t1": [2, 4], "t2": [3, 3]}
    md = render_skill_md("m", "d", "body")
    sha = _md_sha(md)
    v.eval = {
        f"{sha}:t1": {
            "merged_passed": 3,
            "total": 4,
            "cleared": False,
            "failed_requirements": [],
        },
        f"{sha}:t2": {
            "merged_passed": 3,
            "total": 3,
            "cleared": True,
            "failed_requirements": [],
        },
    }

    async def boom(*a, **k):
        raise AssertionError("deduction must not run on a cache hit")

    v._deduct_one = boom
    accept, detail = asyncio.run(v.verify(md, ["t1", "t2"]))
    assert accept  # 3>=2 and 3>=3 -> no regression
    assert v.base["t1"] == [3, 4] and v.base["t2"] == [
        3,
        3,
    ]  # baseline updated to merged


def test_verifier_count_criterion_rejects_regression(tmp_path):
    v = _verifier(tmp_path)
    v.base = {"t1": [4, 4], "t2": [3, 3]}
    md = render_skill_md("m", "d", "body")
    sha = _md_sha(md)
    v.eval = {  # t1 drops 4 -> 2 (regression) even though t2 holds
        f"{sha}:t1": {
            "merged_passed": 2,
            "total": 4,
            "cleared": False,
            "failed_requirements": [],
        },
        f"{sha}:t2": {
            "merged_passed": 3,
            "total": 3,
            "cleared": True,
            "failed_requirements": [],
        },
    }

    async def boom(*a, **k):
        raise AssertionError("cache hit")

    v._deduct_one = boom
    accept, detail = asyncio.run(v.verify(md, ["t1", "t2"]))
    assert not accept
    assert v.base["t1"] == [4, 4]  # baseline NOT updated on reject
    assert [pt["regressed"] for pt in detail["per_task"]] == [True, False]
