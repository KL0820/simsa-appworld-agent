"""Canonical repository paths for the offline MIND-Skill pipeline.

Single source of truth for where data lives, so runners/curation/deploy never
hardcode a directory. Layout (data/mind_skill/):
  source_trajectories/   induction inputs (trajectory.json + induction_slices.json)
  training_runs/<comp>/<run_tag>/   per-run induction records (q0/q1/q2 + result)
  skills/
    induced/   raw induction libraries (<comp>/<run_tag>/{q0,q1,q2,best})
    refined/   curation output (<comp>/{stage1_consolidated,
               stage2_name_description_refined, summary.txt, actions.json})
    release/   final deployed library (thesis_final/<comp>/best + summary.txt)
    skill_libraries/   per-task best libraries (current induced source, pre-rename)
  gold_trajectories/   oracle-generated gold (deduction ground truth)
  _archive/   preserved-but-not-in-clean-structure (engine internals, scratch)
"""

from __future__ import annotations

from pathlib import Path

MIND_SKILL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MIND_SKILL_ROOT.parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "mind_skill"

# induction inputs + ground truth
SOURCE_TRAJECTORIES_DIR = DATA_ROOT / "source_trajectories" / "train"
# Historical harness terminology; both induction and deduction read the same
# canonical 90-task source corpus. The old gold_trajectories folder is archival.
GOLD_TRAJECTORIES_DIR = SOURCE_TRAJECTORIES_DIR
TRAINING_RUNS_DIR = DATA_ROOT / "training_runs"

# skill libraries (the three buckets + the current per-task source)
SKILLS_DIR = DATA_ROOT / "skills"
INDUCED_DIR = SKILLS_DIR / "induced"
REFINED_DIR = SKILLS_DIR / "refined"
RELEASE_DIR = SKILLS_DIR / "release"
SKILL_LIBRARIES_DIR = SKILLS_DIR / "skill_libraries"

# preservation + generated scratch
ARCHIVE_DIR = DATA_ROOT / "_archive"
GENERATED_RUNS_DIR = MIND_SKILL_ROOT / "runs"  # oracle scratch, gitignored

# canonical curation deliverable subdir names (under REFINED_DIR/<comp>/)
STAGE1_CONSOLIDATED = "stage1_consolidated"
STAGE2_REFINED = "stage2_name_description_refined"

__all__ = [
    "ARCHIVE_DIR",
    "DATA_ROOT",
    "GENERATED_RUNS_DIR",
    "GOLD_TRAJECTORIES_DIR",
    "INDUCED_DIR",
    "MIND_SKILL_ROOT",
    "PROJECT_ROOT",
    "REFINED_DIR",
    "RELEASE_DIR",
    "SKILLS_DIR",
    "SKILL_LIBRARIES_DIR",
    "SOURCE_TRAJECTORIES_DIR",
    "STAGE1_CONSOLIDATED",
    "STAGE2_REFINED",
    "TRAINING_RUNS_DIR",
]
