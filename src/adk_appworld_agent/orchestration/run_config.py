"""Typed runtime configuration — the SINGLE source of truth for every knob.

One `RunConfig` carries every runtime setting (model, retry ladders,
retrieval, skills, prompts, paths, cache, debug). Scripts load it from ONE
place — an experiment config file (`load_run_config(path)`) — and inject it
through `build_appworld_controller_agent`; subagents read fields off the
config object, never `os.environ` directly. Every run dumps the resolved
config (`dump_resolved_config`) into its artifacts so any run's exact
settings are always recoverable from disk (no un-logged env vars).

Migration note: the `*_from_env` helpers below are backward-compat shims for
call sites not yet ported to read `RunConfig`. They are being removed as each
subsystem migrates; new code MUST read the typed config, not env.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from adk_appworld_agent.orchestration.state import Phase

DEFAULT_SKILLS_ROOT = Path("data/mind_skill/skills/release/thesis_final")


class ModelConfig(BaseModel):
    """LLM-call defaults applied to every subagent that builds an inner LlmAgent.

    Subagents that need slightly different generation parameters can still
    derive their own per-call config; this is the project-wide baseline.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 123
    candidate_count: int = 1
    max_output_tokens: int = 0  # 0 keeps provider default


class RetryConfig(BaseModel):
    """All retry/backoff policy: the shared provider ladder + executor stall
    handling + the finder's own transient backoff + the LLM HTTP timeout.
    """

    model_config = ConfigDict(extra="forbid")

    max_executor_retries: int = 1
    # Shared provider-error ladder for 429 and retryable 5xx failures.
    # Classification remains separate so exhaustion still produces the correct
    # typed outcome (rate-limited vs generic provider failure).
    provider_backoff_delays: tuple[float, ...] = (4, 8, 16, 32, 64)
    # Executor stall/transient backoff (was code_plan_execute._TRANSIENT_BACKOFF_S).
    executor_stall_backoff: tuple[float, ...] = (16, 32, 64, 128)
    # Finder's own exponential backoff for non-rate-limit transients
    # (was FINDER_RETRY_* env). jitter default 0.25 matches production.
    finder_max_retries: int = 5
    finder_base_delay_s: float = 1.0
    finder_max_delay_s: float = 60.0
    finder_jitter_ratio: float = 0.25
    # UNIFIED gemini-call max-wait (seconds). ONE knob governs BOTH the finder
    # Vertex-call deadline AND the executor Stage-2 per-event stall timer —
    # finder 429/504 and executor EXECUTOR_LLM_STALLED are the same underlying
    # gemini-2.5-flash streaming hang (python-genai #1893 / openclaw #80349).
    # 150 keeps it above the ~125s thinking-hang so a legitimate long-thinking
    # call finishes instead of being cut. Healthy calls finish far under this, so
    # raising it costs nothing on the normal path — it only lets hung/long calls
    # wait longer before being treated as stalled.
    llm_call_timeout_s: float = 150.0


class RetrievalConfig(BaseModel):
    """Finder retrieval knobs: which dependency graph, and how the dependency
    expansion behaves. `dependency_index_file` is a filename under the finder
    `data/` dir (e.g. `api_dependency_index_llm_only.json` for the pure-LLM
    graph) or an absolute path — replaces the per-worktree `API_DEP_INDEX` env.
    """

    model_config = ConfigDict(extra="forbid")

    dependency_index_file: str = "api_dependency_index.json"
    dependency_expansion: bool = True
    batched_dependency: bool = True
    forward_dependency_expansion: bool = False
    embedding: bool = False
    embedding_model: str = "all-mpnet-base-v2"
    # community_select: LLM scores each community 0-10; keep those >= this
    # threshold (graded scoring replaces binary pick — a high-relevance
    # community no longer gets dropped just because a binary judgement was
    # conservative). Lenient default; lower to widen, raise to tighten.
    community_select_min_score: int = 5


class PathsConfig(BaseModel):
    """Data-location overrides (were APPWORLD_* / EXECUTOR_LOG_PATH env).
    None = use the in-code default for that path.
    """

    model_config = ConfigDict(extra="forbid")

    api_docs_dir: Path | None = None
    tasks_dir: Path | None = None
    executor_log_path: Path | None = None


class CacheConfig(BaseModel):
    """Subagent replay-cache control (was SUBAGENT_CACHE_* env). dir=None
    disables cache. `read` lists subagent cache-keys to serve from cache.
    """

    model_config = ConfigDict(extra="forbid")

    dir: Path | None = None
    read: list[str] = Field(default_factory=list)
    version: str | None = None


class DebugConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rough_planner_debug: bool = False


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_root: Path = Path("logs")
    keep_debug: bool = False


class RunConfig(BaseModel):
    """Top-level config bundle — the single source of truth, passed to every
    subagent and the orchestrator.

    `impls` selects which subagent implementation backs each phase, e.g.
    `{Phase.FIND: "community", Phase.PLAN: "rough", Phase.EXECUTE: "stub"}`.
    `timeout_s` is the per-task hard timeout; `executor_timeout_s` bounds a
    single executor code run (was EXECUTOR_TIMEOUT_S).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model: ModelConfig = Field(default_factory=ModelConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    impls: dict[Phase, str] = Field(default_factory=dict)
    timeout_s: float = 600.0
    executor_timeout_s: float = 240.0
    # (executor Stage-2 stall timer now follows the unified
    # retry.llm_call_timeout_s — no separate executor knob.)
    # Run each Stage-2 executor LLM attempt on its OWN fresh event loop in a
    # short-lived worker thread. genai binds its aiohttp connector to the
    # running loop; a gemini-2.5-flash socket stall (python-genai #1893) plus
    # the unawaited AsyncClient.aclose() on GC (#1709) leave a half-dead
    # connection that, on the SHARED per-task loop, makes every subsequent
    # attempt see 0 events (the observed 5x events_seen=0 cascade on 9016950 /
    # d18139b / 042a9fc). A throwaway loop per attempt means the poisoned
    # connection dies with that loop and never reaches the next attempt/phase.
    # A fresh LOOP per attempt (not a fresh process) is enough: the sandbox tool
    # offloads to its own zerorpc thread regardless, and the sandbox trace is
    # file-based, so a worker thread is safe. Default ON; ablatable.
    executor_isolate_llm_loop: bool = True
    # Executor self-assess (observe-then-conclude): after a clean execute, the
    # executor runs ONE grounded review of (milestone intent + produced value +
    # code + real api_trace returns) and emits a GROUNDED done verdict, instead
    # of trusting the summary it wrote BEFORE seeing the result. Default ON
    # (2026-06-22): the convergence redesign makes this the executor→continuation
    # diagnosis channel (continuation §3g HONORS self_assess.ok=False as a
    # high-priority override of phase-level success). Without it the continuation
    # re-reads api_trace blind and misdiagnoses (e.g. 2c544f9: read a real
    # "insufficient balance" return as a "hallucination"). The prior net-negative
    # verifier (5 gain / 7 loss) over-flagged correct answers; this one is
    # grounded in api_trace AND the §3b/§3d/pagination false-positive carve-outs
    # in §3g stop the override from blocking correct mutations/under-fetches.
    # Costs +1 grounded-review LLM call per clean executor attempt. Ablatable.
    executor_self_assess: bool = True
    # On an executor RETRY, render the prior attempt's REAL api returns (actual
    # field names / values / sample rows from api_trace.result_items) into the
    # prior-attempts block, so the code-writer aligns to observed data instead
    # of re-writing blind. Default on (harmless: 0 extra LLM calls, only renders
    # when prior returns exist). Toggle for ablation — was the unnamed "Option A".
    executor_retry_sees_prior_returns: bool = True
    # When True, the code_planner/code-writer for a milestone ALSO sees the REAL
    # api returns observed by EARLIER, already-completed milestones — not just
    # this milestone's own retries and the lossy committed-variable summary.
    # Closes the intermediate-visibility gap: a later step that needs a raw
    # field an earlier step observed but did NOT commit into a named variable
    # (9016950: last_name lives in M1's search_contacts return, but M1 committed
    # only email+phone, so M3's code_planner saw only the committed dict and
    # fabricated the surname 'Kathryn Kathryn'). Reuses the existing
    # _render_prior_returns machinery — 0 extra LLM calls, bounded rows, capped
    # to the most recent N prior milestones. Default OFF so baseline == prior
    # behaviour; ablatable A/B via env EXECUTOR_SEES_PRIOR_MS_RETURNS=1.
    executor_sees_prior_milestone_returns: bool = False
    # Max prior milestones whose observed returns are surfaced (token bound).
    executor_prior_milestone_returns_cap: int = 3
    # ── Economic stop on no structural progress (§4.2) ───────────────────
    # When the executor produces structurally-identical work (same candidate
    # APIs + same APIs actually called + EXACT same committed value) this many
    # times in a row on one milestone, RunController takes the economic-stop
    # terminal — ADVANCE to a later milestone (reallocate budget) or a clean
    # typed FAILED-with-best-so-far on the last milestone — instead of looping to
    # max_cycles and burning into the provider rate-limit storm. Never framed as
    # "task impossible": the task is solvable, the per-milestone budget is just
    # spent. `give_up_enabled` gates the whole path (kept as the ablation-flag
    # name; ablatable vs the prior loop-to-wall behaviour). This identical-streak
    # window is the tighter "structurally stuck" trigger; the looser re-route-
    # immune bound is `attempt_budget` (sweepable: window 4-5).
    give_up_enabled: bool = True
    no_progress_fingerprint_window: int = 4
    # Continuation recovery must not GROW the plan. When True, a
    # `revised_milestones` whose length EXCEEDS the writable region
    # (len > writable_len) is REJECTED — that is the loop-unrolling pathology
    # (325d6ec exploded 1->15 by appending one milestone per iteration). Only
    # NET growth is blocked: in-place refine (len < writable_len, tail preserved)
    # AND a full 1:1 tail replace (len == writable_len, incl. a single-milestone
    # 1-item revise) are ALLOWED — that is the legitimate recovery channel (the
    # continuation re-wording the active milestone so the finder surfaces the
    # right APIs). A loop must be expressed as a single "repeatedly do X until Y"
    # milestone, never unrolled. Ablatable A/B. NOTE: this does NOT fix the
    # 59fae45 full-tail-replace-drops-a-step footgun (len == writable that drops
    # content) — that is a separate, content-level concern.
    continuation_block_plan_growth: bool = True
    # §4.3 bounded prerequisite insertion. With block_plan_growth on, a RETRY
    # revise that is EXACTLY +1 longer than the writable region — [new_prereq,
    # active, …tail] — is PERMITTED (not rejected as growth) up to this many
    # insertions per stable milestone id: the active milestone is preserved
    # (shifted by one, keeping its id) and the new upstream step runs first.
    # This is the legitimate "missing prerequisite" recovery (6f4b9a5: insert a
    # SEARCH before a READ). >+1 growth is still rejected (loop-unrolling). The
    # per-id cap stops it degenerating into a slow one-per-cycle unroll. 0
    # disables insertion entirely (pure in-place refine; ablatable A/B).
    prereq_insertion_cap: int = 2
    # §4.4 bounded FORWARD growth on ADVANCE. With block_plan_growth on, the
    # growth guard rejects ANY plan-lengthening — which also kills the legitimate
    # "the initial plan under-decomposed; ADVANCE and ADD the missing next step"
    # recovery (a30375d read-note→extract-quote, 0a9d82a read-notes→compute-streak,
    # b9c5c9a read-csv→update+write, b6d1104 read-note→compute+append all submit a
    # stale/empty result because the act/compute milestone could never be added).
    # This cap permits an ADVANCE to grow the plan, counted per TASK
    # (state.forward_growth_count): a 1→2/1→3 sequential decomposition completes,
    # while a genuine loop (advance-add one iteration per cycle) hits the cap and
    # economic-stops — so the 1→15 unroll the guard exists to stop stays bounded.
    # DOWNSTREAM forward-add only (distinct from prereq_insertion_cap's upstream
    # insert). 0 disables (pure no-growth; restores the strict guard for A/B).
    continuation_forward_growth_cap: int = 3
    # ── 429 localization probe (diagnostic, default off) ──
    # When True, the FIRST time a finder call hits a 429 it fires a 2-way probe
    # with the BYTE-IDENTICAL request: arm A = same lru_cached client, arm B =
    # fresh in-process genai.Client. A=B 429 → not client-object state
    # (account/project level); A 429 / B ok → per-client-object state. Read-only:
    # never mutates the result or retry metrics.
    finder_429_probe_enabled: bool = False
    # ── Fresh httpx pool on a finder 429 retry — CONFIRMED NO-OP, default off ──
    # Swapping the genai.Client (fresh httpx pool) on a 429 retry is a no-op.
    # ★ DISPROVEN (2026-06-25): the old belief "a fresh PROCESS clears the 429"
    # is FALSE. A controlled test ran each finder call in its OWN fresh
    # subprocess (the former finder_subprocess_isolation, now removed) and
    # d194965 STILL hit consecutive 429s. So neither a fresh client NOR a fresh
    # process escapes this finder 429; the working mitigations are the call
    # deadline (retry.llm_call_timeout_s) + the retry ladder + grace, and
    # rerun-later. Kept only as an ablation toggle; leave False.
    finder_fresh_pool_on_429: bool = False
    # ── Re-route-immune attempt budget (§4.1, clean fix 2026-06-22) ──
    # The terminal transition fires on EITHER N identical fingerprints
    # (no_progress_fingerprint_window) OR this many TOTAL EXECUTE attempts on one
    # milestone regardless of fingerprint. The revise loop changes the
    # fingerprint every cycle (re-route → new candidate_apis), so the identical
    # streak never reaches the window and the milestone would revise to
    # MAX_CYCLES (2c544f9: 14 revise cycles / 123 LLM calls, never bounded →
    # burned the project token quota → 429 cascade). Counting TOTAL attempts —
    # keyed by stable milestone id so it survives re-routes — caps the loop:
    # at the budget the controller takes a terminal transition (ADVANCE
    # best-so-far if not last, else economic stop), on BOTH the success-but-empty
    # revise loop and the repeated-failure loop. USER-SET 6. 0 disables (ablatable).
    attempt_budget: int = 6
    # ── Per-TASK total-work backstop (§4.1b, decompose-immune) ──
    # The per-milestone attempt_budget above is keyed by milestone id, so a
    # continuation that responds to a stuck step by re-DECOMPOSING (split / insert
    # prerequisite) mints new ids → each gets a fresh per-id budget → no single id
    # ever hits the cap → only the loose MAX_CYCLES backstop catches it (3d9a636:
    # plan grew 4→6, 96 LLM calls / 691k tok before MAX_CYCLES). This counts TOTAL
    # EXECUTE attempts across ALL milestones (immune to how the plan is reshaped);
    # at the budget the controller takes the economic-stop terminal (ADVANCE
    # best-so-far if not last, else economic stop), tighter than MAX_CYCLES.
    # Generous vs legitimate use (PASS tasks use ~1-3 total) but well below the
    # ~15 a decompose-escape churns to. 0 disables (ablatable).
    task_attempt_budget: int = 12
    # ── MIND-Skill prompt variants + deterministic skill injection ──
    # "full" = current fat prompt (default, byte-identical to pre-flag
    # behavior). "minimal" = the layer's frozen thin deduction prompt
    # (each agent package's prompts/*.md); the *_skills lists are SKILL.md
    # texts deterministically injected into the minimal prompt's SKILLS slot
    # (training/eval injection path — not native load_skill).
    executor_prompt_variant: str = "full"
    code_planner_prompt_variant: str = "full"
    rough_planner_prompt_variant: str = "full"
    # Prompt variant for the deterministic skill-injection path (impls
    # rough_skill / code_plan_execute_skill). "minimal" (default) keeps the
    # frozen Fig-8 thin prompt; "full" injects the resolved skills on top of
    # the constraint-preserving full prompt (decoupling ablation).
    skills_prompt_variant: str = "minimal"
    executor_skills: list[str] = Field(default_factory=list)
    code_planner_skills: list[str] = Field(default_factory=list)
    rough_planner_skills: list[str] = Field(default_factory=list)
    # Native-ADK SkillToolset condition (model-pull): when a layer's dir is
    # set, the builder mounts SkillToolset over every <task>/<name>/SKILL.md
    # under it and the model loads skills itself via load_skill. Mutually
    # exclusive with the *_skills deterministic injection above.
    executor_skills_dir: Path | None = None
    code_planner_skills_dir: Path | None = None
    rough_planner_skills_dir: Path | None = None
    # Skill-impl configuration (the registry impls rough_skill /
    # code_plan_execute_skill read these): which library to use and the
    # held-out retrieval K (spec §12).
    skills_root: Path = DEFAULT_SKILLS_ROOT
    skills_library: str = "best"
    skills_k: int = 3
    # Held-out skill ranker (train tasks stay pinned-per-task regardless of this):
    #   "llm"       (default) — a gemini call picks top-K over (name, description)
    #   "embedding" — local SentenceTransformer cosine over the same catalog
    #                 (deterministic, no provider call / no 429).
    # Ablation comparator so the skill-retrieval layer can be measured in
    # isolation. Distinct from the (dead) finder-side retrieval.embedding flag.
    skills_selector: str = "llm"
    skills_embedding_model: str = "all-mpnet-base-v2"
    # Held-out skill-selection GRANULARITY (executor stage only; rough_planner
    # runs before milestones exist, so it stays per-task):
    #   False (default) — resolve once per task over the whole instruction,
    #                     reused for every milestone.
    #   True  — re-resolve per active milestone, querying the milestone intent
    #           on top of the task instruction (precision over the per-task
    #           union). NOTE: with the LLM selector this multiplies retrieval
    #           calls per task. Train tasks stay pinned-per-task regardless.
    skills_per_milestone: bool = False

    # ── Single-source load/dump ──────────────────────────────────────────
    def to_json(self) -> str:
        """Serialize the full resolved config (the exact settings of a run)."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        """Load a RunConfig from one experiment config file (JSON).

        This is THE entry point for experiment settings — one file per
        experiment, validated against the typed schema (unknown keys raise).
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(text)


def load_run_config(path: str | Path) -> RunConfig:
    """Module-level alias for `RunConfig.from_file` (import-friendly)."""
    return RunConfig.from_file(path)


def dump_resolved_config(config: RunConfig, artifacts_dir: str | Path) -> Path:
    """Write the resolved RunConfig into a run's artifacts directory.

    Guarantees every run records its exact settings on disk, so what a run
    used is always recoverable (no un-logged env vars). Returns the path.
    """
    out = Path(artifacts_dir) / "resolved_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(config.to_json() + "\n", encoding="utf-8")
    return out


def model_config_from_env() -> ModelConfig:
    """DEPRECATED backward-compat shim: build a `ModelConfig` from `MODEL_*`
    env vars. Being removed as call sites migrate to read `RunConfig.model`.
    """

    raw_max = os.getenv("MODEL_MAX_OUTPUT_TOKENS")
    return ModelConfig(
        name=os.getenv("MODEL_NAME", "gemini-2.5-flash"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0")),
        top_p=float(os.getenv("MODEL_TOP_P", "1")),
        top_k=int(os.getenv("MODEL_TOP_K", "1")),
        seed=int(os.getenv("MODEL_SEED", "123")),
        candidate_count=int(os.getenv("MODEL_CANDIDATE_COUNT", "1")),
        max_output_tokens=int(raw_max) if raw_max else 0,
    )


def executor_timeout_from_env(default_s: float = 240.0) -> float:
    """DEPRECATED backward-compat shim: read `EXECUTOR_TIMEOUT_S` env.
    Being removed in favor of `RunConfig.executor_timeout_s`.
    """
    raw = os.getenv("EXECUTOR_TIMEOUT_S")
    if raw is None:
        return default_s
    try:
        return float(raw)
    except ValueError:
        return default_s


__all__ = [
    "CacheConfig",
    "DebugConfig",
    "LogConfig",
    "ModelConfig",
    "PathsConfig",
    "RetrievalConfig",
    "RetryConfig",
    "RunConfig",
    "dump_resolved_config",
    "executor_timeout_from_env",
    "load_run_config",
    "model_config_from_env",
]
