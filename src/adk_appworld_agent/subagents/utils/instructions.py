"""Instruction provider helpers shared by every agent.

ADK templates plain-string instructions: any `{identifier}` is resolved against
session state and raises KeyError when absent. Injected SKILL.md texts are
model-generated and may legitimately contain braces, so every skill-capable
instruction goes through the no-templating callable below. `.text` carries the
raw prompt for the canonical io block.
"""

from __future__ import annotations


def static_instruction_provider(text: str):
    def _provider(_ctx=None) -> str:
        return text

    _provider.text = text
    return _provider


def instruction_text(instruction) -> str:
    """Raw prompt text of a string or provider instruction (for logs)."""
    return getattr(instruction, "text", instruction) if instruction is not None else ""


__all__ = ["instruction_text", "static_instruction_provider"]
