from __future__ import annotations

import pytest

from adk_appworld_agent.subagents.executor.guardrails import (
    GeneratedCodeValidationError,
    validate_generated_code,
)


def test_out_of_candidate_call_rejection_enumerates_valid_apis():
    """B: when generated code calls an API outside candidate_apis (e.g. the
    continuation-hallucinated plural `add_songs_to_playlist` while the real
    `add_song_to_playlist` is in candidates — 634f342_2), the rejection must
    enumerate the available APIs so the executor's repair can self-correct to a
    real one instead of looping on the wrong name."""
    candidates = [
        {
            "app": "spotify",
            "name": "add_song_to_playlist",
            "parameters": [{"name": "playlist_id"}, {"name": "song_id"}],
        },
        {"app": "spotify", "name": "show_playlist", "parameters": []},
    ]
    code = "apis.spotify.add_songs_to_playlist(playlist_id=1, song_id=2)"  # hallucinated plural

    with pytest.raises(GeneratedCodeValidationError) as exc:
        validate_generated_code(
            code, candidate_apis=candidates, allowed_profile_keys=set()
        )

    msg = str(exc.value)
    assert "add_songs_to_playlist" in msg  # names the offending call
    # steers to the actual candidates
    assert "apis.spotify.add_song_to_playlist" in msg
    assert "apis.spotify.show_playlist" in msg


def test_in_candidate_call_passes():
    candidates = [
        {
            "app": "spotify",
            "name": "add_song_to_playlist",
            "parameters": [{"name": "playlist_id"}, {"name": "song_id"}],
        }
    ]
    # valid call with explicit kwargs from the spec — must not raise
    validate_generated_code(
        "apis.spotify.add_song_to_playlist(playlist_id=1, song_id=2)",
        candidate_apis=candidates,
        allowed_profile_keys=set(),
    )
