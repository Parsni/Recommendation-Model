"""Build the REAL LMArena human-preference battle dataset for the routing engine.

Downloads two public LMArena datasets (one-time, internet required), extracts the
first user prompt of each battle, alias-maps model versions onto the engine's
candidate lineup, and keeps only battles where BOTH sides are candidates.

Sources:
  - lmarena-ai/arena-human-preference-100k  (Jun-Aug 2024, 106,134 battles)
  - lmarena-ai/arena-human-preference-140k  (Apr-Jul 2025, 135,634 battles)

Outputs:
  - data/routing/arena_battles_real.csv    (committed; app/tests run offline)
  - data/routing/KNOWN_FACTS_ARENA.md      (provenance, filters, counts)

Usage:
  python scripts/build_arena_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = PROJECT_ROOT / "data" / "routing" / "arena_battles_real.csv"
FACTS_MD = PROJECT_ROOT / "data" / "routing" / "KNOWN_FACTS_ARENA.md"

# (repo id, split, expected language column value)
SOURCES = [
    ("lmarena-ai/arena-human-preference-100k", "train", "English"),
    ("lmarena-ai/arena-human-preference-140k", "train", "en"),
]

# Alias map: canonical candidate id -> raw Arena model names that refer to it.
# NOTE: qwen-2.5-coder-32b was replaced by qwen3-235b-a22b (real battle data).
MODEL_ALIASES: Dict[str, List[str]] = {
    "gpt-4o": [
        "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20",
        "chatgpt-4o-latest", "chatgpt-4o-latest-20250326",
    ],
    "claude-3-5-sonnet": [
        "claude-3-5-sonnet", "claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022",
    ],
    "deepseek-r1": ["deepseek-r1", "deepseek-r1-0528"],
    "qwen3-235b-a22b": [
        "qwen3-235b-a22b", "qwen3-235b-a22b-no-thinking",
        "qwen3-235b-a22b-thinking-2507", "qwen3-235b-a22b-instruct-2507",
    ],
    "gemini-1.5-pro": [
        "gemini-1.5-pro", "gemini-1.5-pro-api-0514", "gemini-1.5-pro-exp-0801",
    ],
    "llama-3.1-70b": ["llama-3.1-70b", "llama-3.1-70b-instruct"],
    "gpt-4o-mini": ["gpt-4o-mini", "gpt-4o-mini-2024-07-18"],
}

RAW_TO_CANONICAL = {raw.lower(): canon for canon, raws in MODEL_ALIASES.items() for raw in raws}
CANDIDATES = list(MODEL_ALIASES.keys())

CSV_FIELDS = ["prompt", "model_a", "model_b", "winner", "user_id", "source", "is_code", "arena_tags"]


def canonical_model(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    return RAW_TO_CANONICAL.get(str(raw).strip().lower())


def first_user_prompt(conv: Any) -> str:
    """Extract the first user message text from either dataset conversation format."""
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except (json.JSONDecodeError, TypeError):
            return conv.strip()
    if not isinstance(conv, list):
        return ""
    fallback = ""
    for msg in conv:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = msg.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = [
                str(p.get("text", ""))
                for p in content
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
            text = "\n".join(parts).strip()
        else:
            text = ""
        if not text:
            continue
        if role == "user":
            return text
        if not fallback:
            fallback = text
    return fallback


def compact_tags(category_tag: Any, is_code: Any) -> str:
    """Serialize Arena category annotations into a compact, stable string."""
    flags = []
    if isinstance(category_tag, dict):
        math_block = category_tag.get("math_v0.1") or {}
        cw_block = category_tag.get("creative_writing_v0.1") or {}
        if_block = category_tag.get("if_v0.1") or {}
        flags.append(f"math:{1 if math_block.get('math') else 0}")
        flags.append(f"creative:{1 if cw_block.get('creative_writing') else 0}")
        if_score = if_block.get("score")
        flags.append(f"if_score:{if_score if isinstance(if_score, (int, float)) else '-'}")
    flags.append(f"code:{1 if is_code else 0}")
    return ";".join(flags)


def stream_source(repo_id: str, split: str, lang_value: str, stats: Counter) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    print(f"  streaming {repo_id} ...", flush=True)
    ds = load_dataset(repo_id, split=split, streaming=True)
    try:
        ds = ds.select_columns(["model_a", "model_b", "winner", "conversation_a", "language", "is_code", "category_tag"])
    except Exception as exc:  # pragma: no cover - older datasets versions
        print(f"    (column pruning unavailable: {exc})", flush=True)

    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        if i and i % 25000 == 0:
            print(f"    {i} rows scanned, {len(rows)} kept so far", flush=True)
        stats[f"{repo_id}:scanned"] += 1
        if str(row.get("language", "")).strip() != lang_value:
            stats[f"{repo_id}:dropped_language"] += 1
            continue
        winner = str(row.get("winner", "")).strip().lower()
        if winner not in ("model_a", "model_b"):
            stats[f"{repo_id}:dropped_tie_or_bad"] += 1
            continue
        a = canonical_model(row.get("model_a"))
        b = canonical_model(row.get("model_b"))
        if a is None or b is None or a == b:
            stats[f"{repo_id}:dropped_non_candidate"] += 1
            continue
        prompt = first_user_prompt(row.get("conversation_a"))
        if len(prompt) < 10:
            stats[f"{repo_id}:dropped_short_prompt"] += 1
            continue
        stats[f"{repo_id}:kept"] += 1
        rows.append({
            "prompt": " ".join(prompt.split()),
            "model_a": a,
            "model_b": b,
            "winner": "model_a" if winner == "model_a" else "model_b",
            "user_id": "usr_" + hashlib.sha1(prompt.encode("utf-8", "ignore")).hexdigest()[:8],
            "source": "arena-100k" if "100k" in repo_id else "arena-140k",
            "is_code": 1 if row.get("is_code") else 0,
            "arena_tags": compact_tags(row.get("category_tag"), row.get("is_code")),
        })
    return rows


def probe_license(repo_id: str) -> str:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo_id}", timeout=30) as resp:
            data = json.load(resp)
        lic = (data.get("cardData") or {}).get("license")
        if lic:
            return str(lic)
    except Exception:
        pass
    return "see dataset card"


def main() -> int:
    print("Building REAL LMArena battle dataset for the routing engine", flush=True)
    stats: Counter = Counter()
    all_rows: List[Dict[str, Any]] = []

    for repo_id, split, lang_value in SOURCES:
        all_rows.extend(stream_source(repo_id, split, lang_value, stats))

    # Exact-duplicate removal (same prompt + pair + outcome).
    deduped: Dict[tuple, Dict[str, Any]] = {}
    for r in all_rows:
        key = (r["prompt"], r["model_a"], r["model_b"], r["winner"])
        if key not in deduped:
            deduped[key] = r
    rows = list(deduped.values())
    stats["exact_duplicates_removed"] = len(all_rows) - len(rows)

    if len(rows) < 500:
        print(f"FATAL: only {len(rows)} usable battles - check connectivity / filters.", file=sys.stderr)
        return 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---- provenance report ----
    per_model: Counter = Counter()
    per_source: Counter = Counter()
    per_pair: Counter = Counter()
    for r in rows:
        per_model[r["model_a"]] += 1
        per_model[r["model_b"]] += 1
        per_source[r["source"]] += 1
        pair = " vs ".join(sorted((r["model_a"], r["model_b"])))
        per_pair[pair] += 1

    lines = [
        "# KNOWN FACTS - Real LMArena Battle Dataset (`arena_battles_real.csv`)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/build_arena_dataset.py`.",
        "",
        "## Provenance",
        "",
        "Real, publicly released human-preference votes collected by LMArena",
        "(Chatbot Arena). No synthetic rows are present in this file.",
        "",
        "| source | license | rows scanned | kept |",
        "|---|---|---|---|",
    ]
    for repo_id, _split, _lang in SOURCES:
        short = "arena-100k" if "100k" in repo_id else "arena-140k"
        lines.append(
            f"| `{repo_id}` | {probe_license(repo_id)} "
            f"| {stats.get(f'{repo_id}:scanned', 0)} | {stats.get(f'{repo_id}:kept', 0)} |"
        )
    lines += [
        "",
        f"- URL: https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k",
        f"- URL: https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k",
        "",
        "## Filters applied (in order)",
        "",
        "1. Language must be English (`language == 'English'` / `'en'`).",
        "2. `winner` must be `model_a` or `model_b` (ties and `both_bad` dropped).",
        "3. Both battle sides must alias-map onto the engine's 7 candidate models.",
        "4. First user prompt must be >= 10 characters.",
        "5. Exact duplicates of (prompt, model_a, model_b, winner) removed.",
        "",
        "## Candidate alias mapping (raw Arena name -> canonical id)",
        "",
    ]
    for canon, raws in MODEL_ALIASES.items():
        lines.append(f"- `{canon}` <- {', '.join(f'`{r}`' for r in raws)}")
    lines += [
        "",
        "### Honest notes",
        "",
        "- `qwen-2.5-coder-32b` (the project's original candidate) has **zero** battles in",
        "  either source, so the lineup was swapped to `qwen3-235b-a22b` before this file",
        "  was built. See `docs/multi_criteria_llm_routing_architecture.md`.",
        "- Version snapshots of the same model family (e.g. `gpt-4o-2024-05-13` and",
        "  `chatgpt-4o-latest`) are merged into one canonical candidate; per-snapshot",
        "  differences are therefore not modeled.",
        "- `gemini-1.5-pro` and `llama-3.1-70b` battles come only from the 2024 source;",
        "  `deepseek-r1` and `qwen3-235b-a22b` only from the 2025 source. The two eras are",
        "  linked through `gpt-4o`, `gpt-4o-mini` and `claude-3-5-sonnet`, which appear in both.",
        "",
        "## Row counts",
        "",
        f"- **total battles: {len(rows)}**",
        f"- exact duplicates removed: {stats['exact_duplicates_removed']}",
        f"- per source: {dict(per_source)}",
        f"- battles per model (appearances): {dict(per_model)}",
        "",
        "## Head-to-head pair coverage",
        "",
    ]
    for pair, cnt in sorted(per_pair.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {pair}: {cnt}")
    lines.append("")
    FACTS_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {len(rows)} battles -> {OUT_CSV}")
    print(f"Wrote provenance -> {FACTS_MD}")
    print("per model:", dict(per_model))
    print("per source:", dict(per_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
