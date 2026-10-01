"""Three losses + lexicographic ordering (spec §8/§10).

Judges return QUALITY scores (high = good, the calibrated direction for LLMs —
paper D.3); losses convert via loss = upper_bound − score, so every loss is
"lower = better".

  outcome loss  [0,1]  — failed/total from the real evaluator (no LLM)
  recon  loss   [0,10] — 10 − recon judge alignment_score
  rubric loss   [0,10] — GT-independence-GATED: a skill below the GT-independence
                          gate (≥7, round1 spec A3) is scored by that axis alone
                          (so leakage dominates); above the gate, 10 − mean(5 axes).

Lexicographic comparison: outcome > recon > rubric (spec §10).
"""

from __future__ import annotations

from dataclasses import dataclass

from mind_skill.judges.recon import ReconJudgment
from mind_skill.judges.rubric import RubricJudgment

GT_INDEPENDENCE_GATE = 7  # round1 spec A3: pass requires gt_independence >= 7


# ── Loss triple ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, order=True)
class LossTriple:
    """Ordered (outcome, recon, rubric); dataclass order=True IS the
    lexicographic comparison the closed loop uses for best-so-far."""

    outcome: float
    recon: float
    rubric: float

    def as_dict(self) -> dict:
        return {"outcome": self.outcome, "recon": self.recon, "rubric": self.rubric}


def outcome_loss(*, passed: int, failed: int, total: int) -> float:
    """Failure degree from the real evaluator. 0 = all tests passed."""
    if total <= 0:
        return 1.0
    return failed / total


def recon_loss(judgment: ReconJudgment) -> float:
    return 10.0 - float(judgment.alignment_score)


def rubric_loss(judgment: RubricJudgment) -> float:
    if judgment.gt_independence < GT_INDEPENDENCE_GATE:
        return 10.0 - float(judgment.gt_independence)
    axes = [
        judgment.gt_independence,
        judgment.actionability,
        judgment.transferability,
        judgment.completeness,
        judgment.conciseness,
    ]
    return 10.0 - sum(axes) / len(axes)


__all__ = [
    "GT_INDEPENDENCE_GATE",
    "LossTriple",
    "outcome_loss",
    "recon_loss",
    "rubric_loss",
]
