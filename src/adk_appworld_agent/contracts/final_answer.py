from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FinalAnswerOutput(BaseModel):
    """AppWorld final answer extractor output.

    Schema and prompt design ported from cuga-agent's FinalAnswerAgent
    (final_answer_agent.py + system_appworld.jinja2). The extractor reads
    the task instruction plus the last executor output and produces a
    bare answer string the AppWorld evaluator can compare directly.

    final_answer_type drives downstream submission formatting:
    - "int" / "float": numeric query (e.g., "How many", "How much")
    - "str": text query, action confirmation, or error passthrough
    """

    thoughts: list[str] = Field(default_factory=list)
    final_answer: str
    final_answer_type: Literal["str", "int", "float"]


__all__ = ["FinalAnswerOutput"]
