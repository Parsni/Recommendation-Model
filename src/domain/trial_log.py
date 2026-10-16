"""Append-only personal trial logger for the prompt -> AI model domain.

The DA data-collection protocol (data/prompt_rec/README_DATA_COLLECTION.md) needs
one row per (prompt, model, rating). This writes those rows to NEW files in a
folder the caller chooses; it never touches the source CSVs.

Files produced (same headers the engine's column mapping expects):
  my_trials.csv  prompt_id,model_id,quality_score,timestamp
  my_prompts.csv prompt_id,task_category,language,prompt_text
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TRIALS_FILE = "my_trials.csv"
PROMPTS_FILE = "my_prompts.csv"
TRIAL_COLUMNS = ["prompt_id", "model_id", "quality_score", "timestamp"]
PROMPT_COLUMNS = ["prompt_id", "task_category", "language", "prompt_text"]


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing the '{col}' column")
    return df


def next_prompt_id(folder: Path, prefix: str = "MY") -> str:
    """Highest id ever issued, across both files, so deleting a prompts row can
    never make a new prompt reuse an id that trials already reference."""
    folder = Path(folder)
    used = set(_read(folder / PROMPTS_FILE, PROMPT_COLUMNS)["prompt_id"].astype(str))
    used |= set(_read(folder / TRIALS_FILE, TRIAL_COLUMNS)["prompt_id"].astype(str))
    numbers = [int(p[len(prefix):]) for p in used if p.startswith(prefix)
               and p[len(prefix):].isdigit()]
    return f"{prefix}{max(numbers, default=0) + 1:03d}"


def append_trial(folder: Path, prompt_text: str, task_category: str, model_id: str,
                 rating: int, language: str = "en",
                 date: str | None = None) -> dict:
    """Log one rated output. Reusing a prompt id happens when the prompt text
    already exists in my_prompts.csv, so multiple models on the same prompt form
    one basket instead of duplicate prompts."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not str(prompt_text).strip():
        raise ValueError("prompt_text must not be empty")
    if not 1 <= int(rating) <= 5:
        raise ValueError("rating must be between 1 and 5")
    prompts = _read(folder / PROMPTS_FILE, PROMPT_COLUMNS)
    same = prompts[prompts["prompt_text"].astype(str).str.strip() == str(prompt_text).strip()]
    prompt_id = str(same.iloc[0]["prompt_id"]) if len(same) else next_prompt_id(folder)
    stamp = date or pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    trial = pd.DataFrame([{"prompt_id": prompt_id, "model_id": model_id,
                           "quality_score": int(rating), "timestamp": stamp}])
    trial.to_csv(folder / TRIALS_FILE, mode="a", index=False,
                 header=not (folder / TRIALS_FILE).exists())
    if not len(same):
        pd.DataFrame([{"prompt_id": prompt_id, "task_category": task_category,
                       "language": language, "prompt_text": str(prompt_text).strip()}]
                     ).to_csv(folder / PROMPTS_FILE, mode="a", index=False,
                              header=not (folder / PROMPTS_FILE).exists())
    return {"prompt_id": prompt_id, "model_id": model_id, "quality_score": int(rating),
            "timestamp": stamp, "task_category": task_category,
            "prompt_text": str(prompt_text).strip()}


def logged_trials(folder: Path) -> pd.DataFrame:
    """Session-review view: logged trials joined to their prompt text."""
    folder = Path(folder)
    trials = _read(folder / TRIALS_FILE, TRIAL_COLUMNS)
    prompts = _read(folder / PROMPTS_FILE, PROMPT_COLUMNS)
    if trials.empty:
        return pd.DataFrame(columns=[*TRIAL_COLUMNS, "prompt_text", "task_category"])
    return trials.merge(prompts[["prompt_id", "prompt_text", "task_category"]],
                        on="prompt_id", how="left")
