"""Build subagent cache from log artifacts.

Walks a `logs/<experiment>/<run>` directory, reads each task's
`artifacts/subagent_io.jsonl` (the machine-readable sidecar produced by
`SubagentIoJsonlSink`), and writes per-subagent JSONL files in the
`subagent_cache.v1` format that `JsonlSubagentCacheStore` consumes at
runtime.

Runtime is read-only: scripts/run_task.py never writes cache. To refresh
the cache after a prompt or schema change, regenerate logs (run a batch
without `SUBAGENT_CACHE_*` env), then run this script.

Usage:
    uv run scripts/build_cache.py \
        --from logs/my_exp/<run> \
        --to data/subagent_cache/v1 \
        --subagents rough_planner,community_finder

After:
    SUBAGENT_CACHE_DIR=data/subagent_cache/v1 \
    SUBAGENT_CACHE_READ=rough_planner,community_finder \
        uv run scripts/run_task.py --task_id <id> --experiment_name <exp>

The `--subagents` filter selects which subagent_names to extract. Pass
`all` to extract every subagent that has a registered cache_spec.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.cache.spec import CacheSpec
from adk_appworld_agent.orchestration.cache.store import (
    CACHE_FORMAT,
    JsonlSubagentCacheStore,
)
from adk_appworld_agent.subagents.finder.community_finder import (
    build_community_finder_subagent,
)
from adk_appworld_agent.subagents.planner import build_rough_planner_subagent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a subagent_cache.v1 directory from run logs.",
    )
    parser.add_argument(
        "--from",
        dest="src",
        required=True,
        help="Run directory under logs/, e.g. logs/my_exp/20260427_143000.",
    )
    parser.add_argument(
        "--to",
        dest="dst",
        required=True,
        help="Output directory; one <subagent_name>.jsonl per cached subagent.",
    )
    parser.add_argument(
        "--subagents",
        type=str,
        default="all",
        help=(
            "Comma-separated subagent_names to extract. 'all' (default) "
            "extracts every subagent the registry knows a cache_spec for."
        ),
    )
    return parser.parse_args()


def _resolve_specs(filter_str: str) -> dict[str, CacheSpec]:
    """Map envelope.subagent_name → CacheSpec for every registered subagent.

    The lookup key is the *instance* name (what the runtime stamps into
    `envelope.subagent_name`, e.g. "rough_planner_subagent",
    "finder_subagent_community"), not the spec.subagent_name shorthand
    used in cache filenames.

    User-facing `--subagents` filtering still uses `spec.subagent_name`
    (the cache-key shorthand, e.g. `rough_planner`, `community_finder`).
    """

    factories = [
        build_rough_planner_subagent,
        build_community_finder_subagent,
    ]
    by_instance_name: dict[str, CacheSpec] = {}
    by_spec_name: dict[str, str] = {}  # spec.subagent_name -> instance.name
    for factory in factories:
        try:
            instance = factory()
        except Exception as exc:
            print(f"warn: skipped factory {factory.__name__}: {exc}")
            continue
        spec = instance.cache_spec
        if spec is None:
            continue
        by_instance_name[instance.name] = spec
        by_spec_name[spec.subagent_name] = instance.name

    if filter_str.strip().lower() == "all":
        return by_instance_name

    requested = {name.strip() for name in filter_str.split(",") if name.strip()}
    missing = requested - by_spec_name.keys()
    if missing:
        print(f"warn: no cache_spec for subagent_names: {sorted(missing)}")
    return {
        instance_name: by_instance_name[instance_name]
        for spec_name, instance_name in by_spec_name.items()
        if spec_name in requested
    }


def _iter_subagent_io_jsonl(src: Path) -> Iterable[tuple[Path, dict]]:
    """Yield (jsonl_path, record_dict) for every record under `src`."""

    for jsonl_path in sorted(src.glob("**/artifacts/subagent_io.jsonl")):
        for line_no, line in enumerate(
            jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warn: {jsonl_path}:{line_no} invalid JSON: {exc}")
                continue
            yield jsonl_path, record


def _envelope_from_record(record: dict) -> SubagentEnvelope | None:
    raw = record.get("envelope")
    if not isinstance(raw, dict):
        return None
    try:
        return SubagentEnvelope.model_validate(raw)
    except Exception as exc:
        print(f"warn: envelope failed validation: {exc}")
        return None


def _subagent_input_from_record(
    record: dict, envelope: SubagentEnvelope
) -> SubagentInput | None:
    """Reconstruct the SubagentInput used at the original call site.

    The IO record stores `worker_input` / `subagent_input` either as a dict
    or as `subagent_input_text` (json string). Try each variant.
    """

    io_record = (
        record.get("io_record") if isinstance(record.get("io_record"), dict) else {}
    )
    input_block = (
        io_record.get("input") if isinstance(io_record.get("input"), dict) else {}
    )
    candidate = input_block.get("subagent_input")
    if isinstance(candidate, dict):
        try:
            return SubagentInput.model_validate(candidate)
        except Exception:
            pass
    text = input_block.get("subagent_input_text")
    if isinstance(text, str) and text.strip():
        try:
            return SubagentInput.model_validate_json(text)
        except Exception:
            pass
    return None


def _build_cache(src: Path, dst: Path, specs: dict[str, CacheSpec]) -> dict[str, int]:
    """Walk logs and emit cache records keyed by `spec.subagent_name`.

    `specs` maps envelope.subagent_name (instance name) → CacheSpec; the
    emitted JSONL filenames use `spec.subagent_name` (the short cache key
    the runtime store expects, e.g. `rough_planner.jsonl`).
    """

    store = JsonlSubagentCacheStore(dst)
    counts: dict[str, int] = {spec.subagent_name: 0 for spec in specs.values()}
    skipped = 0
    seen_keys: dict[str, set[str]] = {
        spec.subagent_name: set() for spec in specs.values()
    }

    for jsonl_path, record in _iter_subagent_io_jsonl(src):
        envelope = _envelope_from_record(record)
        if envelope is None:
            skipped += 1
            continue
        if envelope.status != SubagentStatus.SUCCEEDED:
            continue
        spec = specs.get(envelope.subagent_name)
        if spec is None:
            continue
        if not spec.replayable(envelope):
            continue
        subagent_input = _subagent_input_from_record(record, envelope)
        if subagent_input is None:
            print(
                f"warn: {jsonl_path}: cannot reconstruct SubagentInput for "
                f"{envelope.subagent_name}; record skipped"
            )
            skipped += 1
            continue
        key = spec.key_fn(subagent_input)
        dedup_token = f"{spec.version}\x1f{key}"
        if dedup_token in seen_keys[spec.subagent_name]:
            continue
        seen_keys[spec.subagent_name].add(dedup_token)
        store.put(
            spec.subagent_name,
            key,
            spec.version,
            envelope,
            meta={"source": str(jsonl_path)},
        )
        counts[spec.subagent_name] += 1

    if skipped:
        print(f"warn: {skipped} records skipped (invalid envelope or missing input)")
    return counts


def main() -> int:
    args = _parse_args()
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"ERROR: source not found: {src}")
        return 2
    dst.mkdir(parents=True, exist_ok=True)

    specs = _resolve_specs(args.subagents)
    if not specs:
        print("ERROR: no matching subagents resolved; nothing to do.")
        return 2

    print(f"Source : {src}")
    print(f"Output : {dst}")
    print(f"Subagents: {sorted(specs.keys())}")
    print(f"Cache format: {CACHE_FORMAT}")

    counts = _build_cache(src, dst, specs)
    for name, count in counts.items():
        path = dst / f"{name}.jsonl"
        print(f"  {name}: {count} records -> {path}")

    if not any(counts.values()):
        print("warn: no cache records were emitted")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
