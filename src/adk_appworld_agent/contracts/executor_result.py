from __future__ import annotations

import json

from pydantic import BaseModel, Field, field_validator


class MemoryVariable(BaseModel):
    name: str
    value_json: str
    description: str = ""
    type_name: str = ""
    count_items: int = 1
    created_at: str = ""
    source_milestone_id: str = ""

    @field_validator("name")
    @classmethod
    def _name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("memory variable name must not be blank")
        return value

    @field_validator("value_json")
    @classmethod
    def _value_json_must_decode(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("memory variable value_json must not be blank")
        try:
            json.loads(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("memory variable value_json must be valid JSON") from exc
        return value


class ExecutorResult(BaseModel):
    answer: str = ""
    milestone_done: bool = False
    summary: str = ""
    variables: list[MemoryVariable] = Field(default_factory=list)


__all__ = ["MemoryVariable", "ExecutorResult"]
