from __future__ import annotations

import pytest

from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.finder.ablation_finders import (
    AppScopeFinderSubagent,
    AppScopeNoDepFinderSubagent,
    CommunityForwardFinderSubagent,
    CommunityNoDepFinderSubagent,
    CommunityNoGuideFinderSubagent,
    GlobalSeedFilterFinderSubagent,
)
from adk_appworld_agent.subagents.finder.community_finder import CommunityFinderSubagent
from adk_appworld_agent.subagents.registry import (
    available_subagent_impls,
    build_subagent,
)

# (impl key, class, community_layer, dependency_expansion, cache namespace)
_VARIANTS = [
    ("community", CommunityFinderSubagent, True, True, "community_finder"),
    ("app_scope", AppScopeFinderSubagent, False, True, "app_scope_finder"),
    (
        "app_scope_nodep",
        AppScopeNoDepFinderSubagent,
        False,
        False,
        "app_scope_nodep_finder",
    ),
    (
        "community_nodep",
        CommunityNoDepFinderSubagent,
        True,
        False,
        "community_nodep_finder",
    ),
    (
        "global_seed_filter",
        GlobalSeedFilterFinderSubagent,
        False,
        False,
        "global_seed_filter_finder",
    ),
    (
        "community_noguide",
        CommunityNoGuideFinderSubagent,
        True,
        True,
        "community_noguide_finder",
    ),
    (
        "community_forward",
        CommunityForwardFinderSubagent,
        True,
        True,
        "community_forward_finder",
    ),
]


@pytest.mark.parametrize(
    "impl,cls,community_layer,dependency_expansion,cache_name", _VARIANTS
)
def test_registry_builds_finder_variant_with_correct_flags(
    impl, cls, community_layer, dependency_expansion, cache_name
):
    sub = build_subagent(Phase.FIND, impl)
    assert isinstance(sub, cls)
    assert sub.community_layer is community_layer
    assert sub.dependency_expansion is dependency_expansion
    assert sub.phase_ is Phase.FIND
    assert sub.cache_spec_ is not None
    assert sub.cache_spec_.subagent_name == cache_name


def test_find_registry_exposes_ablation_impls():
    impls = available_subagent_impls(Phase.FIND)
    for key in (
        "community",
        "app_scope",
        "app_scope_nodep",
        "community_nodep",
        "global_seed_filter",
    ):
        assert key in impls


def test_global_seed_filter_flag_isolated_to_its_variant():
    # global_pool must be True ONLY for global_seed_filter; every other variant
    # (incl. production) keeps it False so their pools stay app/community-scoped.
    gsf = build_subagent(Phase.FIND, "global_seed_filter")
    assert gsf.global_pool is True
    assert gsf.community_layer is False
    assert gsf.dependency_expansion is False
    for impl in ("community", "app_scope", "app_scope_nodep", "community_nodep"):
        assert build_subagent(Phase.FIND, impl).global_pool is False


def test_community_noguide_blanks_guidance_only_for_its_variant():
    # use_guidance must be False ONLY for community_noguide (which otherwise
    # matches the production finder: community + dep on); every other variant
    # keeps guidance on.
    ng = build_subagent(Phase.FIND, "community_noguide")
    assert ng.use_guidance is False
    assert ng.community_layer is True
    assert ng.dependency_expansion is True
    for impl in ("community", "app_scope", "global_seed_filter"):
        assert build_subagent(Phase.FIND, impl).use_guidance is True


def test_community_forward_flag_isolated_to_its_variant():
    # forward_dependency_expansion True ONLY for community_forward; it otherwise
    # matches the production finder (community + dep + guidance ON).
    fw = build_subagent(Phase.FIND, "community_forward")
    assert fw.forward_dependency_expansion is True
    assert fw.community_layer is True
    assert fw.dependency_expansion is True
    assert fw.use_guidance is True
    for impl in ("community", "app_scope", "community_noguide", "global_seed_filter"):
        assert build_subagent(Phase.FIND, impl).forward_dependency_expansion is False


def test_ablation_variants_have_distinct_cache_namespaces():
    names = {
        build_subagent(Phase.FIND, impl).cache_spec_.subagent_name
        for impl, *_ in _VARIANTS
    }
    # Each variant must read from its own cache namespace — never another's.
    assert len(names) == len(_VARIANTS)


def test_ablation_variants_are_community_finder_subclasses():
    # They inherit run_subagent + the WorkerOutputEnvelope payload schema from
    # the production finder, so only the retrieval mechanism differs.
    for impl in (
        "app_scope",
        "app_scope_nodep",
        "community_nodep",
        "global_seed_filter",
    ):
        assert isinstance(build_subagent(Phase.FIND, impl), CommunityFinderSubagent)


def test_production_community_finder_defaults_unchanged():
    # Guard: Variant A (the production finder) must keep both switches True so
    # its behaviour is byte-for-byte the pre-ablation pipeline.
    sub = build_subagent(Phase.FIND, "community")
    assert sub.community_layer is True
    assert sub.dependency_expansion is True
