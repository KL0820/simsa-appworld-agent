"""TextGrad update cycle (Fig 11/12): gradient diagnoses, optimizer applies.

gradient.py  — GRADIENT_SYSTEM; sees P_I / skill / τ̂ / loss feedback, never τ
optimizer.py — OPTIMIZER_SYSTEM; sees only P_I + gradient text, <IMPROVED_VARIABLE> out
"""

from mind_skill.textgrad.gradient import (
    GRADIENT_SYSTEM,
    build_gradient_agent,
    build_gradient_input,
    compute_gradient,
)
from mind_skill.textgrad.optimizer import (
    OPTIMIZER_SYSTEM,
    build_optimizer_agent,
    build_optimizer_input,
    extract_improved_prompt,
    optimize_prompt,
)

__all__ = [
    "GRADIENT_SYSTEM",
    "OPTIMIZER_SYSTEM",
    "build_gradient_agent",
    "build_gradient_input",
    "build_optimizer_agent",
    "build_optimizer_input",
    "compute_gradient",
    "extract_improved_prompt",
    "optimize_prompt",
]
