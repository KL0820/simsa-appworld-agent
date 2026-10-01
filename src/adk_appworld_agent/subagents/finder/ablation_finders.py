"""Retrieval-ablation finder variants for the 2x2 community/dependency study.

Experiment-only subagents. Each flips one or both of the two retrieval switches
on CommunityFinderSubagent (community_layer, dependency_expansion) — see
routing.search_apis_by_community_routing. The production finder (Variant A) is
community_finder.CommunityFinderSubagent with both switches True; nothing here
touches it. Safe to delete once the ablation is concluded.

Variant map (community_layer, dependency_expansion):
  A        (True,  True ) -> community_finder (not here)
  B        (False, True ) -> AppScopeFinderSubagent        impl "app_scope"
  B2       (False, False) -> AppScopeNoDepFinderSubagent   impl "app_scope_nodep"
  A-no-dep (True,  False) -> CommunityNoDepFinderSubagent  impl "community_nodep"
"""

from __future__ import annotations

from typing import ClassVar

from adk_appworld_agent.orchestration.cache.spec import CacheSpec
from adk_appworld_agent.orchestration.run_config import (
    RunConfig,
    model_config_from_env,
)
from adk_appworld_agent.subagents.finder.community_finder import (
    CommunityFinderSubagent,
    _community_finder_key,
)


def _ablation_cache_spec(subagent_name: str) -> CacheSpec:
    # Each variant gets its OWN cache namespace so it can never read Variant A's
    # (or another variant's) cached finder output. key_fn is shared — the cache
    # key is (task, milestone, planned_apps); the distinct subagent_name keeps
    # the variants' caches apart.
    return CacheSpec(
        subagent_name=subagent_name,
        version="v1",
        key_fn=_community_finder_key,
    )


class AppScopeFinderSubagent(CommunityFinderSubagent):
    """Variant B: no community layer, dependency expansion kept."""

    cache_spec_: ClassVar[CacheSpec | None] = _ablation_cache_spec("app_scope_finder")
    community_layer: bool = False
    dependency_expansion: bool = True


class AppScopeNoDepFinderSubagent(CommunityFinderSubagent):
    """Variant B2: pure app->api (no community, no dependency expansion)."""

    cache_spec_: ClassVar[CacheSpec | None] = _ablation_cache_spec(
        "app_scope_nodep_finder"
    )
    community_layer: bool = False
    dependency_expansion: bool = False


class CommunityNoDepFinderSubagent(CommunityFinderSubagent):
    """Variant A-no-dep: community layer kept, dependency expansion removed."""

    cache_spec_: ClassVar[CacheSpec | None] = _ablation_cache_spec(
        "community_nodep_finder"
    )
    community_layer: bool = True
    dependency_expansion: bool = False


class GlobalSeedFilterFinderSubagent(CommunityFinderSubagent):
    """No-narrowing baseline: seed_filter selects from the ENTIRE 454-op
    user-facing API surface (planned_apps ignored), no community layer, no
    dependency expansion. Measures whether selection degrades at full scale and
    how much retrieval narrowing buys in tokens."""

    cache_spec_: ClassVar[CacheSpec | None] = _ablation_cache_spec(
        "global_seed_filter_finder"
    )
    community_layer: bool = False
    dependency_expansion: bool = False
    global_pool: bool = True


class CommunityNoGuideFinderSubagent(CommunityFinderSubagent):
    """Production community finder (community layer + dep ON) but with the
    train-induced behavior_guidelines + app_context blanked out. Isolates how
    much of the finder's selection accuracy rests on that semi-manually induced
    guidance vs the model + official API specs alone."""

    cache_spec_: ClassVar[CacheSpec | None] = _ablation_cache_spec(
        "community_noguide_finder"
    )
    community_layer: bool = True
    dependency_expansion: bool = True
    use_guidance: bool = False


class CommunityForwardFinderSubagent(CommunityFinderSubagent):
    """Production community finder (community + dep + guidance, full-spec) PLUS
    forward dependency expansion: the dep loop also surfaces consumer communities
    of selected producers ("create X -> act on X" next-step APIs). Measures
    whether forward expansion lifts recall of next-step action APIs (e.g.
    add_song_to_playlist) over the forward-OFF production baseline."""

    cache_spec_: ClassVar[CacheSpec | None] = _ablation_cache_spec(
        "community_forward_finder"
    )
    forward_dependency_expansion: bool = True


def build_app_scope_finder_subagent(
    name: str = "finder_subagent_app_scope",
    *,
    run_config: RunConfig | None = None,
) -> AppScopeFinderSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    return AppScopeFinderSubagent(
        name=name,
        description="Ablation B: app-scope finder (no community layer, dependency expansion kept).",
        model_cfg=model_cfg,
    )


def build_app_scope_nodep_finder_subagent(
    name: str = "finder_subagent_app_scope_nodep",
    *,
    run_config: RunConfig | None = None,
) -> AppScopeNoDepFinderSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    return AppScopeNoDepFinderSubagent(
        name=name,
        description="Ablation B2: pure app->api finder (no community, no dependency expansion).",
        model_cfg=model_cfg,
    )


def build_community_nodep_finder_subagent(
    name: str = "finder_subagent_community_nodep",
    *,
    run_config: RunConfig | None = None,
) -> CommunityNoDepFinderSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    return CommunityNoDepFinderSubagent(
        name=name,
        description="Ablation A-no-dep: community finder without dependency expansion.",
        model_cfg=model_cfg,
    )


def build_global_seed_filter_finder_subagent(
    name: str = "finder_subagent_global_seed_filter",
    *,
    run_config: RunConfig | None = None,
) -> GlobalSeedFilterFinderSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    return GlobalSeedFilterFinderSubagent(
        name=name,
        description="No-narrowing baseline: seed_filter over the whole 454-op API surface.",
        model_cfg=model_cfg,
    )


def build_community_noguide_finder_subagent(
    name: str = "finder_subagent_community_noguide",
    *,
    run_config: RunConfig | None = None,
) -> CommunityNoGuideFinderSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    return CommunityNoGuideFinderSubagent(
        name=name,
        description="Guidance-ablation: community finder with behavior_guidelines + app_context blanked.",
        model_cfg=model_cfg,
    )


def build_community_forward_finder_subagent(
    name: str = "finder_subagent_community_forward",
    *,
    run_config: RunConfig | None = None,
) -> CommunityForwardFinderSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    return CommunityForwardFinderSubagent(
        name=name,
        description="Production community finder + forward dependency expansion.",
        model_cfg=model_cfg,
    )


__all__ = [
    "AppScopeFinderSubagent",
    "AppScopeNoDepFinderSubagent",
    "CommunityNoDepFinderSubagent",
    "GlobalSeedFilterFinderSubagent",
    "CommunityNoGuideFinderSubagent",
    "CommunityForwardFinderSubagent",
    "build_app_scope_finder_subagent",
    "build_app_scope_nodep_finder_subagent",
    "build_community_nodep_finder_subagent",
    "build_global_seed_filter_finder_subagent",
    "build_community_noguide_finder_subagent",
    "build_community_forward_finder_subagent",
]
