from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Protocol

from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope

CACHE_FORMAT = "subagent_cache.v1"


class SubagentCacheStore(Protocol):
    def get(
        self, subagent_name: str, key: str, version: str
    ) -> tuple[SubagentEnvelope, dict] | None: ...

    def put(
        self,
        subagent_name: str,
        key: str,
        version: str,
        envelope: SubagentEnvelope,
        *,
        meta: dict,
    ) -> None: ...


class JsonlSubagentCacheStore:
    """One JSONL file per subagent_name under a namespace directory.

    File layout: ``<root>/<subagent_name>.jsonl``. Each line is a record keyed by
    (subagent_name, version, key). Later records win on duplicate keys; readers
    keep the last matching record.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._cache: dict[str, dict[str, dict]] = {}
        self._loaded: set[str] = set()

    def _file_for(self, subagent_name: str) -> Path:
        return self.root / f"{subagent_name}.jsonl"

    def _load_subagent(self, subagent_name: str) -> None:
        if subagent_name in self._loaded:
            return
        self._loaded.add(subagent_name)
        path = self._file_for(subagent_name)
        records: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if record.get("cache_format") != CACHE_FORMAT:
                    continue
                if record.get("subagent_name") != subagent_name:
                    continue
                key = record.get("key")
                version = record.get("version")
                if isinstance(key, str) and isinstance(version, str):
                    records[f"{version}\x1f{key}"] = record
        self._cache[subagent_name] = records

    def get(
        self, subagent_name: str, key: str, version: str
    ) -> tuple[SubagentEnvelope, dict] | None:
        self._load_subagent(subagent_name)
        record = self._cache[subagent_name].get(f"{version}\x1f{key}")
        if record is None:
            return None
        try:
            envelope = SubagentEnvelope.model_validate(record["envelope"])
        except Exception:
            return None
        meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
        return envelope, dict(meta)

    def put(
        self,
        subagent_name: str,
        key: str,
        version: str,
        envelope: SubagentEnvelope,
        *,
        meta: dict,
    ) -> None:
        self._load_subagent(subagent_name)
        record = {
            "cache_format": CACHE_FORMAT,
            "subagent_name": subagent_name,
            "version": version,
            "key": key,
            "envelope": envelope.model_dump(mode="json"),
            "meta": {**meta, "saved_at": time.time()},
        }
        self._cache[subagent_name][f"{version}\x1f{key}"] = record
        self._flush(subagent_name)

    def _flush(self, subagent_name: str) -> None:
        path = self._file_for(subagent_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(
            self._cache[subagent_name].values(),
            key=lambda r: (str(r.get("version", "")), str(r.get("key", ""))),
        )
        body = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        )
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{subagent_name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
