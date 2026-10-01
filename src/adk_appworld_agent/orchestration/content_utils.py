from __future__ import annotations

from google.genai import types


def content_from_text(text: str, *, role: str) -> types.Content:
    return types.Content(role=role, parts=[types.Part(text=text)])


def content_to_text(content: types.Content | None) -> str:
    if content is None or not content.parts:
        return ""

    chunks: list[str] = []
    for part in content.parts:
        if getattr(part, "text", None):
            chunks.append(part.text)
    return "\n".join(chunks).strip()
