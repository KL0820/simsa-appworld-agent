"""Tracked AppWorld task-set definitions used by launchers and batch runs."""

from __future__ import annotations

from pathlib import Path

TASK_SETS_DIR = Path(__file__).resolve().parents[2] / "configs" / "task_sets"
STANDARD_DATASETS = ("train", "dev", "test_normal", "test_challenge")
DATASET_CHOICES = ("quick_smoke", *STANDARD_DATASETS, "extra_only")
VARIANT_CHOICES = ("full", "1", "2", "3")


def task_set_path(dataset: str) -> Path:
    """Return the single tracked source file for a selectable dataset."""
    return TASK_SETS_DIR / f"{dataset}.txt"


def task_set_counts() -> dict[str, dict[str, int]]:
    """Return full and suffix-derived counts for every selectable dataset."""
    counts: dict[str, dict[str, int]] = {"extra_only": {"full": 0}}
    for dataset in ("quick_smoke", *STANDARD_DATASETS):
        task_ids = _read_task_ids(task_set_path(dataset))
        dataset_counts = {"full": len(task_ids)}
        if dataset != "quick_smoke":
            dataset_counts.update(
                {
                    variant: sum(
                        task_id.endswith(f"_{variant}") for task_id in task_ids
                    )
                    for variant in VARIANT_CHOICES[1:]
                }
            )
        counts[dataset] = dataset_counts
    return counts


def _read_task_ids(path: Path) -> list[str]:
    return [
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


__all__ = [
    "DATASET_CHOICES",
    "STANDARD_DATASETS",
    "TASK_SETS_DIR",
    "VARIANT_CHOICES",
    "task_set_counts",
    "task_set_path",
]
