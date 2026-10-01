"""Two-stage skill-library curation (builds on dedup.py).

Motivation (project_skill_curation_rationale): the deployment retrieval selector
operates under "hidden-body asymmetry" (SkillRouter) — it only sees each skill's
name+description, never the body. Training every parameterization (_1/_2/_3) of a
scenario induces near-duplicate skills (~83% high body-similarity for _1/_2). So a
combined library must (1) remove true duplicates and (2) make survivors' name+desc
mutually distinctive. Two stages, each using the RIGHT signal:

  Stage 1 (merge, signal = BODY): cluster skills by full-body embedding similarity
    -> judge reads bodies, decides merge (same skill) vs distinct -> a MERGE is
    accepted only if a `verifier` (deduction, no outcome regression) confirms it
    didn't drop a load-bearing detail. Distinct clusters are left for Stage 2.

  Stage 2 (sharpen, signal = NAME+DESC): on the post-merge library, cluster
    survivors by name+description similarity (incl. exact-name) -> rewrite their
    names+descriptions to be mutually distinctive so the selector can route. Body
    is UNCHANGED, so no deduction needed.

Reuses dedup.py for SkillFile/parse/render/load + the merge/distinct judge +
apply_resolution. The embedding clusterer (similarity trigger) and the
deduction-verify gate are the new pieces here.

Provenance (AGENTS.md): operates ONLY on already-induced skill content as library
hygiene; the merge/rename judgments are the model's (on bodies), the merge gate is
the real deduction outcome. Nothing task-specific hand-written.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from mind_skill.curation.dedup import (
    CollisionResolution,
    Resolver,
    SkillFile,
    _llm_resolver,
    _unique,
    apply_resolution,
    load_skill_files,
    render_skill_md,
)

DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
# Both curation stages use the deployed 0.80 candidate cutoff.
# The judge and per-source deduction gate decide whether a merge is accepted.
DEFAULT_BODY_THRESHOLD = 0.80
# Stage 2 sharpens name+desc (what the selector sees). It uses the SAME 0.80 cosine
# candidate cutoff as Stage-1 merge, on the SAME PAIRWISE highest-first basis (a pair
# is a sharpen candidate iff its name+desc cosine >= 0.80) -- so the two stages are
# one curation loop with one calibrated cutoff, differing only in the operation
# (merge a redundant pair vs. relabel a confusable pair) and the embedded text
# (body vs. name+desc). Pairwise (not connected-component) avoids transitive chaining
# collapsing the library into one blob -- same rationale as `pairwise_candidates`.
DEFAULT_NAMEDESC_THRESHOLD = 0.80


# ── embedding + clustering (the similarity trigger) ────────────────────────────
_MODEL_CACHE: dict[str, object] = {}


def _embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _skill_text(sf: SkillFile, kind: Literal["body", "namedesc"]) -> str:
    head = f"{sf.name}. {sf.description}"
    return f"{head}\n{sf.body}" if kind == "body" else head


def similarity_clusters(
    skills: list[SkillFile],
    *,
    kind: Literal["body", "namedesc"],
    threshold: float,
    embedder=None,
) -> list[list[SkillFile]]:
    """Group skills into clusters where each cluster is a connected component of the
    "pairwise cosine >= threshold" graph and has >1 member. `kind` picks the embedded
    text: 'body' (name+desc+body, for Stage-1 merge) or 'namedesc' (Stage-2 sharpen).
    Exact-name pairs are always linked (cosine of identical text is 1.0 anyway)."""
    n = len(skills)
    if n < 2:
        return []
    model = embedder or _embedder()
    vecs = model.encode(
        [_skill_text(s, kind) for s in skills], normalize_embeddings=True
    )
    # union-find over pairs above threshold
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            if float(vecs[i] @ vecs[j]) >= threshold:
                union(i, j)
    groups: dict[int, list[SkillFile]] = {}
    for i, s in enumerate(skills):
        groups.setdefault(find(i), []).append(s)
    return [g for g in groups.values() if len(g) > 1]


# ── pairwise candidates (avoids connected-component chaining) ──────────────────
def pairwise_candidates(
    skills: list[SkillFile],
    *,
    kind: Literal["body", "namedesc"],
    threshold: float,
    embedder=None,
) -> list[tuple[SkillFile, SkillFile, float]]:
    """Every PAIR (not transitive cluster) with cosine >= threshold, sorted by
    similarity desc. Pairwise (vs connected-component) so structurally-similar
    skills never CHAIN into one giant blob — each candidate is judged on just its
    two members' content."""
    n = len(skills)
    if n < 2:
        return []
    model = embedder or _embedder()
    vecs = model.encode(
        [_skill_text(s, kind) for s in skills], normalize_embeddings=True
    )
    out: list[tuple[SkillFile, SkillFile, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(vecs[i] @ vecs[j])
            if sim >= threshold:
                out.append((skills[i], skills[j], sim))
    out.sort(key=lambda t: -t[2])
    return out


# ── pairwise merge judge (similar content, possibly different names) ────────────
CURATE_MERGE_SYSTEM = """\
You are curating a procedural-skill library. You are given TWO skills with similar
content (they may have DIFFERENT names). Each skill's (name, description) is the
retrieval key a task is matched against. Decide, by reading their FULL bodies:

- "merge": one skill's useful information is REDUNDANT with / SUBSUMED by the other
  -- they teach the SAME procedure for the SAME situation, and their differences
  are only wording / emphasis / examples, OR one is essentially a subset of the
  other. Keeping both wastes a retrieval slot on a duplicate. Produce ONE focused
  skill whose body UNIONS the genuinely-useful knowledge of both -- do NOT drop a
  load-bearing detail (a pitfall, an ordering invariant, an extra phase) that only
  one of them had.

- "distinct": each skill provides DIFFERENT useful information -- a different
  trigger condition, a different operation, or a load-bearing detail the other
  lacks -- so a single skill covering both would either LOSE that information or
  become a vague catch-all. Keep BOTH. (Naming is a separate later step; you do not
  rename here.)

Decision test: would merging LOSE useful information that ONE skill had and the
other did NOT? If yes -> distinct. If one focused skill can faithfully hold
everything useful from both -> merge.

Rules: merged_name lowercase kebab-case. For "merge" fill merged_name /
merged_description / merged_body (markdown body, NO frontmatter). For "distinct"
leave the merge fields empty and renames empty.
Output: the structured object only.
"""


def build_curate_judge(*, journal_path: Path | None = None):
    from google.adk.agents import LlmAgent

    from mind_skill.runtime import (
        make_journal_callbacks,
        meta_generate_config,
        meta_model_config,
    )

    callbacks: dict = {}
    if journal_path is not None:
        before, after = make_journal_callbacks(journal_path, role="curate_merge_judge")
        callbacks = {"before_model_callback": before, "after_model_callback": after}
    return LlmAgent(
        name="skill_curate_merge_judge",
        model=meta_model_config().name,
        description="Decides merge-vs-distinct for two similar-content skills.",
        instruction=CURATE_MERGE_SYSTEM,
        output_schema=CollisionResolution,
        generate_content_config=meta_generate_config(),
        **callbacks,
    )


# ── independence graph + multi-round convergence engine ────────────────────────
# Design (project_skill_curation_rationale, user 2026-06-29): curation is graph
# contraction. NODES = current skills; EDGES = "must stay independent (do NOT
# merge)", added on BOTH a judge `distinct` verdict AND a verify failure. Each
# round: candidate pairs = (cosine >= threshold) AND (no independence edge),
# processed sim-desc, consume-once. A passing merge CONTRACTS its two nodes into
# one that INHERITS the union of both parents' edges -- invariant `A⊥C => (A∪B)⊥C`
# (a merge is a union AB⊇A, so A's distinction from C survives in AB). Edges only
# grow (and are inherited on merge) => candidates shrink monotonically => the
# round loop converges. Settled pairs (distinct / verify-fail) are NEVER re-judged
# or re-verified unless a skill's content changes. The graph + coverage + caches
# persist so a later (_3) increment only judges pairs involving new/changed skills.


def _content_sha(sf: SkillFile) -> str:
    """Stable 16-hex content fingerprint of a skill (name+desc+body). Identity for
    independence edges: a merge changes content -> changes sha -> a stale edge from
    a prior session no longer matches and is dropped on reconcile."""
    txt = render_skill_md(sf.name, sf.description, sf.body)
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


def _node_key(sf: SkillFile) -> str:
    return f"{sf.task_id}:{sf.name}"


def _md_sha(merged_md: str) -> str:
    return hashlib.sha256(merged_md.encode("utf-8")).hexdigest()[:16]


class IndependenceGraph:
    """Symmetric "must-stay-independent" relation over current skills.

    Edge key = frozenset({node_key_a, node_key_b}); value carries kind
    (distinct|verify_fail), reason, round, the two endpoints' content shas (for
    cross-session reconcile), and whether it was inherited via a merge."""

    def __init__(self) -> None:
        self.edges: dict[frozenset, dict] = {}

    def blocked(self, a: SkillFile, b: SkillFile) -> bool:
        return frozenset({_node_key(a), _node_key(b)}) in self.edges

    def add_edge(
        self, a: SkillFile, b: SkillFile, *, kind: str, reason: str, round_no: int
    ) -> None:
        ka, kb = _node_key(a), _node_key(b)
        self.edges[frozenset({ka, kb})] = {
            "kind": kind,
            "reason": reason,
            "round": round_no,
            "sha": {ka: _content_sha(a), kb: _content_sha(b)},
            "inherited": False,
        }

    def merge_nodes(self, a: SkillFile, b: SkillFile, merged: SkillFile) -> None:
        """Contract a,b -> merged: every edge touching a or b is redirected to
        merged (independence inherited as a union). The a-b pair itself cannot have
        an edge (you never merge a blocked pair), so no self-loop arises."""
        ka, kb, km = _node_key(a), _node_key(b), _node_key(merged)
        msha = _content_sha(merged)
        parents = {ka, kb}
        new_edges: dict[frozenset, dict] = {}
        for es, meta in self.edges.items():
            redirected = {km if x in parents else x for x in es}
            if len(redirected) < 2:  # both endpoints were the parents -> collapses
                continue
            nsha = {}
            for x in es:
                tgt = km if x in parents else x
                nsha[tgt] = msha if x in parents else meta["sha"].get(x)
            new_edges[frozenset(redirected)] = {
                **meta,
                "sha": nsha,
                "inherited": meta.get("inherited", False) or bool(parents & set(es)),
            }
        self.edges = new_edges

    def reconcile(self, skills: list[SkillFile]) -> None:
        """Drop edges whose endpoints no longer match a current skill by key+sha
        (stale from a prior session where a skill was edited/removed outside merge)."""
        cur = {_node_key(s): _content_sha(s) for s in skills}
        self.edges = {
            es: meta
            for es, meta in self.edges.items()
            if all(k in cur and cur[k] == meta["sha"].get(k) for k in es)
        }

    def as_list(self) -> list[dict]:
        return [{"nodes": sorted(es), **meta} for es, meta in self.edges.items()]

    def persist(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.as_list(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "IndependenceGraph":
        g = cls()
        p = Path(path)
        if p.exists():
            for e in json.loads(p.read_text(encoding="utf-8")):
                ns = e["nodes"]
                g.edges[frozenset(ns)] = {
                    "kind": e.get("kind", "distinct"),
                    "reason": e.get("reason", ""),
                    "round": e.get("round", 0),
                    "sha": e.get("sha", {}),
                    "inherited": e.get("inherited", False),
                }
        return g


class Coverage:
    """Per-skill set of source task_ids it covers (= tasks its merge must not
    regress). Original skill covers {its task_id}; a merge covers the union of its
    parents'. Persisted (provenance.json) so verification always checks ALL source
    tasks of a merged skill, not just its on-disk home task_id."""

    def __init__(self, mapping: dict[str, list[str]] | None = None) -> None:
        self.map: dict[str, set[str]] = {k: set(v) for k, v in (mapping or {}).items()}

    def tasks_of(self, sf: SkillFile) -> set[str]:
        return self.map.get(_node_key(sf), {sf.task_id})

    def set_merged(self, merged: SkillFile, tasks: set[str]) -> None:
        self.map[_node_key(merged)] = set(tasks)

    def sync(self, skills: list[SkillFile]) -> None:
        """Ensure every current skill has an entry (default = own task_id) and prune
        entries for skills no longer present."""
        keys = {_node_key(s) for s in skills}
        for s in skills:
            self.map.setdefault(_node_key(s), {s.task_id})
        self.map = {k: v for k, v in self.map.items() if k in keys}

    def persist(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {k: sorted(v) for k, v in self.map.items()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Coverage":
        p = Path(path)
        return cls(json.loads(p.read_text(encoding="utf-8")) if p.exists() else {})


# verifier protocol for the engine: (merged_md, source_task_ids) -> (accept, detail)
CachingVerifier = Callable[[str, list[str]], Awaitable[tuple[bool, dict]]]


async def _merge_one_round(
    library_dir: Path,
    *,
    threshold: float,
    resolver: Resolver,
    verifier: CachingVerifier | None,
    embedder,
    graph: IndependenceGraph,
    coverage: Coverage,
    round_no: int,
) -> dict:
    """One PAIRWISE, consume-once, graph-gated, verify-gated merge round."""
    skills = load_skill_files(library_dir)
    graph.reconcile(skills)
    coverage.sync(skills)
    cands = pairwise_candidates(
        skills, kind="body", threshold=threshold, embedder=embedder
    )
    consumed: set[str] = set()
    log = {
        "round": round_no,
        "n_start": len(skills),
        "merged": [],
        "rejected": [],
        "distinct": [],
        "skipped_consumed": [],
        "skipped_blocked": [],
        "skipped_infra": [],
    }
    for a, b, sim in cands:
        ka, kb = _node_key(a), _node_key(b)
        if ka in consumed or kb in consumed:
            log["skipped_consumed"].append([ka, kb, round(sim, 3)])
            continue
        if graph.blocked(a, b):
            log["skipped_blocked"].append([ka, kb, round(sim, 3)])
            continue
        all_names = {s.name for s in load_skill_files(library_dir)}
        res = await resolver(a.name, [a, b], sorted(all_names - {a.name, b.name}))
        entry = {"pair": [ka, kb], "sim": round(sim, 3), "reason": res.reason}
        if res.verdict != "merge":
            graph.add_edge(a, b, kind="distinct", reason=res.reason, round_no=round_no)
            log["distinct"].append(entry)
            continue
        merged_md = render_skill_md(
            res.merged_name, res.merged_description, res.merged_body
        )
        source_tasks = sorted(coverage.tasks_of(a) | coverage.tasks_of(b))
        if verifier is None:
            accept, detail = True, {}
        else:
            try:
                accept, detail = await verifier(merged_md, source_tasks)
            except asyncio.TimeoutError:
                # deduction hung past the timeout + retries -> infra, not a quality
                # verdict. Skip (no edge, no merge); a later round retries.
                log["skipped_infra"].append({**entry, "reason": "deduction timeout"})
                continue
            except RuntimeError as exc:
                # Provider-quota infra abort (e.g. rough_planner's
                # RoughOutcomeRateLimited: 429 grind past its rollout budget).
                # Same treatment as a timeout: no edge, no merge, retry later.
                if type(exc).__name__ != "RoughOutcomeRateLimited":
                    raise
                log["skipped_infra"].append({**entry, "reason": f"rate-limited: {exc}"})
                continue
        if not accept:
            graph.add_edge(
                a, b, kind="verify_fail", reason=res.reason, round_no=round_no
            )
            log["rejected"].append(
                {**entry, "merged_name": res.merged_name, "verify": detail}
            )
            continue
        actions = apply_resolution([a, b], res, all_names - {a.name, b.name})
        to = actions[0]["to"]  # {task_id, name}
        merged_sf = next(
            s
            for s in load_skill_files(library_dir)
            if s.task_id == to["task_id"] and s.name == to["name"]
        )
        graph.merge_nodes(a, b, merged_sf)
        coverage.set_merged(merged_sf, set(source_tasks))
        consumed.add(ka)
        consumed.add(kb)
        log["merged"].append(
            {
                **entry,
                "merged_name": to["name"],
                "source_tasks": source_tasks,
                "verify": detail,
            }
        )
    log["n_end"] = len(load_skill_files(library_dir))
    log["merged_count"] = len(log["merged"])
    return log


async def curate_merge_to_convergence(
    library_dir: Path,
    *,
    threshold: float = DEFAULT_BODY_THRESHOLD,
    resolver: Resolver | None = None,
    verifier: CachingVerifier | None = None,
    embedder=None,
    state_dir: Path | None = None,
    snapshot_dir: Path | None = None,
    start_round: int = 1,
    max_rounds: int = 10,
) -> dict:
    """Stage-1 merge run to convergence (graph contraction, multi-round).

    Mutates library_dir. Persists state under `state_dir` (default = library_dir):
    independence.json (the must-stay-independent edges, inherited on merge) and
    provenance.json (coverage). If `snapshot_dir` is given, the library is copied to
    `snapshot_dir/round<r>` after each round (staged storage to browse every stage).
    Stops when a round merges 0 (converged) or after max_rounds. `start_round`>1
    CONTINUES a prior run (rounds.json is loaded and appended, round labels/snapshots
    keep counting), reusing the persisted graph/coverage so settled pairs stay frozen
    -- the same mechanism a later (_3) increment uses."""
    import shutil

    if resolver is None:
        resolver = _llm_resolver(build_curate_judge())
    state = Path(state_dir or library_dir)
    state.mkdir(parents=True, exist_ok=True)
    graph = IndependenceGraph.load(state / "independence.json")
    coverage = Coverage.load(state / "provenance.json")
    rounds: list[dict] = (
        json.loads((state / "rounds.json").read_text("utf-8"))
        if (state / "rounds.json").exists()
        else []
    )
    converged = False
    for r in range(start_round, max_rounds + 1):
        rl = await _merge_one_round(
            library_dir,
            threshold=threshold,
            resolver=resolver,
            verifier=verifier,
            embedder=embedder,
            graph=graph,
            coverage=coverage,
            round_no=r,
        )
        rounds.append(rl)
        graph.persist(state / "independence.json")
        coverage.persist(state / "provenance.json")
        (state / "rounds.json").write_text(
            json.dumps(rounds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if snapshot_dir is not None:
            snap = Path(snapshot_dir) / f"round{r}"
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(library_dir, snap)
        if rl["merged_count"] == 0:
            converged = True
            break
    return {
        "library": str(library_dir),
        "rounds": rounds,
        "converged": converged,
        "n_final": len(load_skill_files(library_dir)),
        "edges": len(graph.edges),
    }


# deduction protocol: (merged_md, task_id, work_dir) -> {merged_passed,total,cleared,failed_requirements}
DeductOne = Callable[[str, str, Path], Awaitable[dict]]


def make_executor_deduct_one(*, runs_dir, model_cfg, rpc_url: str) -> DeductOne:
    """code_executor outcome: teacher-force the plan, run the executor's code in the
    sandbox, score requirements."""
    from mind_skill.deduction.code_executor import load_gold_task, run_deduction

    async def deduct_one(merged_md: str, task_id: str, work_dir: Path) -> dict:
        gold = load_gold_task(runs_dir, task_id)
        res = await run_deduction(
            gold,
            skill_texts=[merged_md],
            model_cfg=model_cfg,
            rpc_url=rpc_url,
            work_dir=work_dir,
            executor_instruction=None,
        )
        return {
            "merged_passed": res.passed,
            "total": res.total,
            "cleared": bool(res.cleared),
            "failed_requirements": res.failed_requirements,
        }

    return deduct_one


def make_planner_deduct_one(
    component: str, *, runs_dir, model_cfg, rpc_url: str, skills_root
) -> DeductOne:
    """code_planner / rough_planner OUTCOME (Path B, deployment-matched): the live
    thin+skill planner generates the plan, the real downstream (executor with its
    OWN per-task skill from `skills_root`) runs it through the sandbox + evaluator.
    Outcome (passed/total), NOT recon -- a merged planner skill is kept iff the
    plans it yields still succeed (per user 2026-06-29). NOTE rough_planner runs the
    FULL controller (continuation on), single-sample => higher variance."""
    import importlib

    mod = importlib.import_module(f"mind_skill.deduction.{component}")

    async def deduct_one(merged_md: str, task_id: str, work_dir: Path) -> dict:
        gold = mod.load_gold(runs_dir, task_id)
        _planner, ded = await mod.run_outcome(
            gold,
            skill_texts=[merged_md],
            model_cfg=model_cfg,
            rpc_url=rpc_url,
            work_dir=work_dir,
            runs_dir=runs_dir,
            skills_root=skills_root,
        )
        return {
            "merged_passed": ded.passed,
            "total": ded.total,
            "cleared": bool(ded.cleared),
            "failed_requirements": ded.failed_requirements,
        }

    return deduct_one


class MergeVerifier:
    """Outcome-based, requirement-count-no-decrease merge gate with incremental
    persistence (so a kill / criterion change never forces re-running deduction).
    Deduction is INJECTED (`deduct_one`) so the same gate serves any layer
    (executor / code_planner / rough_planner).

    `verify(merged_md, source_task_ids)`: for EACH source task, run the merged
    skill once (or reuse a cached result keyed by the merged skill's content sha +
    task) and accept iff `merged_passed >= baseline_passed` for every task --
    baseline = the CURRENT best for that task in the library (updated when a merge
    is accepted, so each step is no-regression relative to the live library, and a
    merged skill is re-checked against ALL the tasks it covers including
    'grandparent' tasks of earlier merges).

    State files (under cache dir):
      - baseline_cache.json: task_id -> [passed, total]  (current best; updated)
      - merge_eval_cache.json: f"{merged_sha}:{task_id}" -> deduction result
      - merge_audit.json: append-only per-merge audit
    All persisted immediately after each deduction so partial progress survives.
    """

    def __init__(
        self,
        *,
        deduct_one: DeductOne,
        work_dir: Path,
        baseline_cache_path: Path,
        merge_eval_cache_path: Path,
        audit_path: Path | None = None,
        baseline_log_dir: Path | None = None,
        baseline_skill_md: Callable[[str], str | None] | None = None,
        verify_timeout_s: float | None = None,
        verify_retries: int = 1,
    ) -> None:
        self._deduct_one = deduct_one
        # original-skill markdown for a task -> lazily compute its baseline when not
        # in the cache nor the training log (e.g. _2 tasks with no outcome log).
        self._baseline_skill_md = baseline_skill_md
        # per-deduction wall-clock cut + retry: a hung gemini stream / sandbox call
        # would otherwise hang the whole run forever (no timeout on this path).
        # On final timeout _eval_task raises asyncio.TimeoutError -> the engine logs
        # the merge as skipped_infra (no permanent edge; a later round retries).
        self.verify_timeout_s = verify_timeout_s
        self.verify_retries = verify_retries
        self.work_dir = Path(work_dir)
        self.baseline_cache_path = Path(baseline_cache_path)
        self.merge_eval_cache_path = Path(merge_eval_cache_path)
        self.audit_path = Path(audit_path) if audit_path else None
        self.baseline_log_dir = Path(baseline_log_dir) if baseline_log_dir else None
        self.base: dict[str, list[int]] = (
            {
                k: list(v)
                for k, v in json.loads(
                    self.baseline_cache_path.read_text("utf-8")
                ).items()
            }
            if self.baseline_cache_path.exists()
            else {}
        )
        self.eval: dict[str, dict] = (
            json.loads(self.merge_eval_cache_path.read_text("utf-8"))
            if self.merge_eval_cache_path.exists()
            else {}
        )
        self.audit: list[dict] = (
            json.loads(self.audit_path.read_text("utf-8"))
            if self.audit_path and self.audit_path.exists()
            else []
        )

    def _persist_base(self) -> None:
        self.baseline_cache_path.write_text(
            json.dumps(self.base, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _persist_eval(self) -> None:
        self.merge_eval_cache_path.write_text(
            json.dumps(self.eval, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _persist_audit(self) -> None:
        if self.audit_path:
            self.audit_path.write_text(
                json.dumps(self.audit, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    async def _baseline(self, task_id: str) -> tuple[int, int]:
        if task_id in self.base:
            return tuple(self.base[task_id])
        # training log (read passed/total of the best-q deduction)
        if self.baseline_log_dir is not None:
            rp = self.baseline_log_dir / task_id / "result.json"
            if rp.exists():
                r = json.loads(rp.read_text(encoding="utf-8"))
                dp = (
                    self.baseline_log_dir
                    / task_id
                    / f"q{r['best_q']}"
                    / "deduction.json"
                )
                if dp.exists():
                    dd = json.loads(dp.read_text(encoding="utf-8"))
                    pair = [int(dd.get("passed", 0)), int(dd.get("total", 0))]
                    self.base[task_id] = pair
                    self._persist_base()
                    return tuple(pair)
        # compute from the ORIGINAL skill (lazy; cached in the eval store by sha:task)
        if self._baseline_skill_md is not None:
            md = self._baseline_skill_md(task_id)
            if md:
                data = await self._eval_task(md, task_id)
                pair = [data["merged_passed"], data["total"]]
                self.base[task_id] = pair
                self._persist_base()
                return tuple(pair)
        raise KeyError(
            f"no baseline for task {task_id} (seed baseline_cache, pass baseline_log_dir, "
            "or baseline_skill_md)"
        )

    async def _eval_task(self, merged_md: str, task_id: str) -> dict:
        ekey = f"{_md_sha(merged_md)}:{task_id}"
        if ekey in self.eval:
            return self.eval[ekey]
        base = self.work_dir / f"merged_{_md_sha(merged_md)}_{task_id}"
        if not self.verify_timeout_s:
            data = await self._deduct_one(merged_md, task_id, base)
        else:
            last: Exception | None = None
            for attempt in range(self.verify_retries + 1):
                wd = base if attempt == 0 else Path(f"{base}_retry{attempt}")
                try:
                    data = await asyncio.wait_for(
                        self._deduct_one(merged_md, task_id, wd),
                        timeout=self.verify_timeout_s,
                    )
                    break
                except asyncio.TimeoutError as exc:  # hung stream/call -> cut + retry
                    last = exc
            else:
                raise last  # exhausted retries -> propagate (engine logs skipped_infra)
        self.eval[ekey] = data
        self._persist_eval()
        return data

    async def verify(
        self, merged_md: str, source_task_ids: list[str]
    ) -> tuple[bool, dict]:
        per_task = []
        ok = True
        for task_id in source_task_ids:
            base_passed, total = await self._baseline(task_id)
            data = await self._eval_task(merged_md, task_id)
            regressed = data["merged_passed"] < base_passed
            per_task.append(
                {
                    "task_id": task_id,
                    "baseline_passed": base_passed,
                    "merged_passed": data["merged_passed"],
                    "total": data["total"],
                    "regressed": bool(regressed),
                    "merged_cleared": data["cleared"],
                    "failed_requirements": data["failed_requirements"],
                }
            )
            if regressed:
                ok = False
        detail = {"accept": ok, "merged_sha": _md_sha(merged_md), "per_task": per_task}
        self.audit.append(detail)
        self._persist_audit()
        if ok:  # the merged skill is now the current best for every task it covers
            for pt in per_task:
                self.base[pt["task_id"]] = [pt["merged_passed"], pt["total"]]
            self._persist_base()
        return ok, detail


class ExecutorMergeVerifier(MergeVerifier):
    """`MergeVerifier` wired to the code_executor outcome deduction (back-compat
    constructor for the code_executor curation runner)."""

    def __init__(
        self,
        *,
        runs_dir,
        model_cfg,
        rpc_url: str,
        work_dir: Path,
        baseline_cache_path: Path,
        merge_eval_cache_path: Path,
        audit_path: Path | None = None,
        baseline_log_dir: Path | None = None,
    ) -> None:
        super().__init__(
            deduct_one=make_executor_deduct_one(
                runs_dir=runs_dir, model_cfg=model_cfg, rpc_url=rpc_url
            ),
            work_dir=work_dir,
            baseline_cache_path=baseline_cache_path,
            merge_eval_cache_path=merge_eval_cache_path,
            audit_path=audit_path,
            baseline_log_dir=baseline_log_dir,
        )


# ── Stage 2: name+desc-similarity sharpen (rewrite for selector distinctiveness) ─
class Sharpen(BaseModel):
    """One rewritten skill, positionally matched to its input."""

    new_name: str = Field(
        description="unique lowercase kebab-case name (a-z, 0-9, hyphens)"
    )
    new_description: str = Field(
        description="one-line description sharpened so a retriever seeing ONLY "
        "name+description can tell this skill apart from its look-alike siblings: "
        "encode the distinctive trigger/applicability (when to use this vs the others)"
    )


class SharpenResolution(BaseModel):
    """Stage-2 output for one confusably-similar-named cluster: rename+sharpen ALL
    (no merge — these survived Stage 1 as genuinely distinct)."""

    reason: str = Field(
        description="one sentence: the distinctive axis used to separate them"
    )
    renames: list[Sharpen] = Field(
        description="exactly one Sharpen per input skill, SAME ORDER as given"
    )


SHARPEN_SYSTEM = """\
You are improving the RETRIEVAL KEYS of an agent skill library. The deployment \
selector picks skills by reading ONLY their name + one-line description (it never \
sees the body — "hidden-body asymmetry"). You are given two or more DISTINCT \
skills (already confirmed not duplicates) whose names/descriptions are \
confusingly similar, so the selector cannot reliably tell them apart.

Read their full bodies and rewrite EACH skill's name + one-line description so \
that, looking at name+description ALONE, the selector can route the right task to \
the right skill. Make the names mutually NON-overlapping and the descriptions \
encode each skill's DISTINCTIVE trigger/applicability (when to use THIS one and \
NOT its siblings). Do NOT change behavior — you are only relabeling.

Rules:
- All names lowercase kebab-case, unique, and not equal to any name in AVOID_NAMES.
- Return exactly one rename per input skill, in the SAME ORDER (Skill 1, Skill 2, ...).
- Bodies are NOT modified.
Output: the structured object only.
"""


def build_sharpen_judge(*, journal_path: Path | None = None):
    from google.adk.agents import LlmAgent

    from mind_skill.runtime import (
        make_journal_callbacks,
        meta_generate_config,
        meta_model_config,
    )

    callbacks: dict = {}
    if journal_path is not None:
        before, after = make_journal_callbacks(journal_path, role="sharpen_judge")
        callbacks = {"before_model_callback": before, "after_model_callback": after}
    return LlmAgent(
        name="skill_sharpen_judge",
        model=meta_model_config().name,
        description="Rewrites confusably-similar skill names+descriptions to be distinctive.",
        instruction=SHARPEN_SYSTEM,
        output_schema=SharpenResolution,
        generate_content_config=meta_generate_config(),
        **callbacks,
    )


def build_sharpen_input(skills: list[SkillFile], avoid_names: list[str]) -> str:
    blocks = [
        f"=== Skill {i} (from task {s.task_id}) ===\nname: {s.name}\n"
        f"description: {s.description}\n\n{s.body}"
        for i, s in enumerate(skills, 1)
    ]
    avoid = ", ".join(sorted(avoid_names)) or "(none)"
    return f"AVOID_NAMES (do not reuse): {avoid}\n\n" + "\n\n".join(blocks)


Sharpener = Callable[[list[SkillFile], list[str]], Awaitable[SharpenResolution]]


def _llm_sharpener(judge) -> Sharpener:
    from mind_skill.runtime import run_meta_agent_validated

    async def sharpen(skills, avoid_names):
        return await run_meta_agent_validated(
            judge, build_sharpen_input(skills, avoid_names), SharpenResolution
        )

    return sharpen


def _apply_sharpen(
    skills: list[SkillFile], res: SharpenResolution, taken: set[str]
) -> list[dict]:
    """Rewrite frontmatter name+description (body verbatim) for a sharpened cluster."""
    import shutil

    if len(res.renames) != len(skills):
        raise ValueError(f"sharpen needs {len(skills)} renames, got {len(res.renames)}")
    for sf in skills:
        shutil.rmtree(sf.skill_dir)
    actions = []
    for sf, rn in zip(skills, res.renames):
        final = _unique(rn.new_name, taken)
        taken.add(final)
        dest = sf.skill_dir.parent / final / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            render_skill_md(final, rn.new_description, sf.body), encoding="utf-8"
        )
        actions.append(
            {"from": {"task_id": sf.task_id, "name": sf.name}, "to_name": final}
        )
    return actions


# Stage-2 is the same convergence loop as Stage-1, with the operation swapped from
# "merge a redundant pair" to "relabel a confusable pair". A pair, once sharpened, is
# recorded in a settle-set keyed by the two skills' STABLE home task_ids (a rename
# changes a skill's name but not its home task, and each task owns at most one library
# skill -> the key survives the rewrite). The settle-set is the Stage-2 analogue of
# Stage-1's IndependenceGraph: it only grows, over the finite set of task-id pairs, so
# the loop is guaranteed to converge (a round either sharpens a new pair or stops).
def _load_settled(path: Path) -> set[frozenset]:
    p = Path(path)
    if not p.exists():
        return set()
    return {frozenset(pair) for pair in json.loads(p.read_text(encoding="utf-8"))}


def _persist_settled(settled: set[frozenset], path: Path) -> None:
    Path(path).write_text(
        json.dumps(
            sorted(sorted(pair) for pair in settled), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


async def _sharpen_one_round(
    library_dir: Path,
    *,
    threshold: float,
    sharpener: Sharpener,
    embedder,
    settled: set[frozenset],
    round_no: int,
) -> dict:
    """One PAIRWISE, consume-once, settle-gated name+desc sharpen round -- the Stage-2
    analogue of `_merge_one_round`. Candidates are the unsettled name+desc pairs with
    cosine >= threshold, processed highest-first; each rewrites BOTH members' name +
    one-line description to be mutually distinctive (bodies verbatim) and records the
    pair (by home task_ids) as settled so it never re-fires."""
    skills = load_skill_files(library_dir)
    cands = pairwise_candidates(
        skills, kind="namedesc", threshold=threshold, embedder=embedder
    )
    consumed: set[str] = set()
    log: dict = {
        "round": round_no,
        "n_start": len(skills),
        "sharpened": [],
        "skipped_consumed": [],
        "skipped_settled": [],
    }
    for a, b, sim in cands:
        pair = frozenset({a.task_id, b.task_id})
        if a.task_id in consumed or b.task_id in consumed:
            log["skipped_consumed"].append([sorted(pair), round(sim, 3)])
            continue
        if pair in settled:
            log["skipped_settled"].append([sorted(pair), round(sim, 3)])
            continue
        all_names = {s.name for s in load_skill_files(library_dir)}
        res = await sharpener([a, b], sorted(all_names - {a.name, b.name}))
        actions = _apply_sharpen([a, b], res, set(all_names - {a.name, b.name}))
        settled.add(pair)
        consumed.add(a.task_id)
        consumed.add(b.task_id)
        log["sharpened"].append(
            {
                "pair": sorted(pair),
                "sim": round(sim, 3),
                "reason": res.reason,
                "renames": actions,
            }
        )
    log["n_end"] = len(load_skill_files(library_dir))
    log["sharpened_count"] = len(log["sharpened"])
    return log


async def stage2_sharpen_library(
    library_dir: Path,
    *,
    threshold: float = DEFAULT_NAMEDESC_THRESHOLD,
    sharpener: Sharpener | None = None,
    embedder=None,
) -> list[dict]:
    """ONE pairwise name+desc sharpen pass; returns the flat list of rename actions.
    For the full Stage-2 run use `sharpen_to_convergence`. Pairwise (not
    connected-component) so a chain of weak look-alikes never collapses the whole
    library into one cluster -- the same anti-chaining rationale as `pairwise_candidates`."""
    if sharpener is None:
        sharpener = _llm_sharpener(build_sharpen_judge())
    rl = await _sharpen_one_round(
        library_dir,
        threshold=threshold,
        sharpener=sharpener,
        embedder=embedder,
        settled=set(),
        round_no=1,
    )
    return [a for s in rl["sharpened"] for a in s["renames"]]


async def sharpen_to_convergence(
    library_dir: Path,
    *,
    threshold: float = DEFAULT_NAMEDESC_THRESHOLD,
    sharpener: Sharpener | None = None,
    embedder=None,
    state_dir: Path | None = None,
    snapshot_dir: Path | None = None,
    start_round: int = 1,
    max_rounds: int = 10,
) -> dict:
    """Stage-2 sharpen run to convergence -- structurally mirrors
    `curate_merge_to_convergence`. Each round computes pairwise name+desc candidates
    (cosine >= threshold, minus already-sharpened pairs), processes them consume-once
    highest-first, and rewrites each pair's name+description for selector
    distinctiveness (bodies + skill count never change -- only retrieval keys).

    Mutates library_dir. Persists `sharpened_pairs.json` (the settle-set, the Stage-2
    analogue of independence.json) and `sharpen_rounds.json` under `state_dir`
    (default = library_dir); snapshots `snapshot_dir/round<r>` per round if given.
    Stops when a round sharpens 0 (converged) or after max_rounds; `start_round`>1
    continues a prior run, reusing the persisted settle-set so settled pairs stay
    frozen (same resume mechanism as Stage-1)."""
    import shutil

    if sharpener is None:
        sharpener = _llm_sharpener(build_sharpen_judge())
    state = Path(state_dir or library_dir)
    state.mkdir(parents=True, exist_ok=True)
    settled = _load_settled(state / "sharpened_pairs.json")
    rounds: list[dict] = (
        json.loads((state / "sharpen_rounds.json").read_text("utf-8"))
        if (state / "sharpen_rounds.json").exists()
        else []
    )
    converged = False
    for r in range(start_round, max_rounds + 1):
        rl = await _sharpen_one_round(
            library_dir,
            threshold=threshold,
            sharpener=sharpener,
            embedder=embedder,
            settled=settled,
            round_no=r,
        )
        rounds.append(rl)
        _persist_settled(settled, state / "sharpened_pairs.json")
        (state / "sharpen_rounds.json").write_text(
            json.dumps(rounds, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if snapshot_dir is not None:
            snap = Path(snapshot_dir) / f"round{r}"
            if snap.exists():
                shutil.rmtree(snap)
            shutil.copytree(library_dir, snap)
        if rl["sharpened_count"] == 0:
            converged = True
            break
    return {
        "library": str(library_dir),
        "rounds": rounds,
        "converged": converged,
        "n_final": len(load_skill_files(library_dir)),
        "settled": len(settled),
    }


__all__ = [
    "CachingVerifier",
    "Coverage",
    "DeductOne",
    "ExecutorMergeVerifier",
    "IndependenceGraph",
    "MergeVerifier",
    "make_executor_deduct_one",
    "make_planner_deduct_one",
    "Sharpen",
    "SharpenResolution",
    "Sharpener",
    "build_sharpen_judge",
    "build_sharpen_input",
    "curate_merge_to_convergence",
    "pairwise_candidates",
    "similarity_clusters",
    "stage2_sharpen_library",
    "sharpen_to_convergence",
]
