"""Strict input validation and a small, explicitly fictional dataset."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["client_id", "session", "note"]


def validate_notes(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Input contains no notes.")
    result = frame[REQUIRED_COLUMNS].copy()
    for column in ("client_id", "note"):
        if result[column].isna().any():
            raise ValueError(f"{column} must contain non-empty text.")
        if not result[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain text.")
        result[column] = result[column].str.strip()
        if result[column].eq("").any():
            raise ValueError(f"{column} must contain non-empty text.")
    sessions = pd.to_numeric(result["session"], errors="coerce")
    if (sessions.isna() | (sessions < 1) | (sessions % 1 != 0)).any():
        raise ValueError("session must be a positive integer.")
    result["session"] = sessions.astype("int64")
    if result.duplicated(["client_id", "session"]).any():
        raise ValueError("Each client_id/session pair must be unique.")
    return result.sort_values(["client_id", "session"]).reset_index(drop=True)


def load_notes(path: Path) -> pd.DataFrame:
    # Preserve IDs like 001 and literal text like NA rather than treating it as missing.
    frame = pd.read_csv(path, dtype={"client_id": str, "note": str}, keep_default_na=False)
    return validate_notes(frame)


def synthetic_notes() -> pd.DataFrame:
    notes = {
        "fictional-A": [
            "I feel anxious about work. I am not sleeping well and feel overwhelmed.",
            "I still worry about work, but the breathing exercise helped me pause.",
            "I felt proud after speaking up. I enjoyed a calm walk with a friend.",
            "This week was difficult again. I felt frustrated and tired after an argument.",
            "I asked for support and felt heard. I am hopeful about practicing new skills.",
        ],
        "fictional-B": [
            "I feel sad and lonely since moving. I miss my old friends.",
            "I was nervous at a community event, but I stayed and met someone kind.",
            "I felt angry when my plans fell through. I do not know what to try next.",
            "I called a friend and enjoyed our conversation. I feel less alone today.",
            "I feel both excited and afraid about the coming week. I plan to stay connected.",
        ],
    }
    return validate_notes(pd.DataFrame([
        {"client_id": client, "session": session, "note": note}
        for client, entries in notes.items()
        for session, note in enumerate(entries, 1)
    ]))
