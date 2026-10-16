"""Path, config and logging helpers shared across the engine."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"

for _d in (OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else CONFIG_DIR / "default_config.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_logger(name: str = "recs") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger


class FailureLog:
    """Append-only, JSON-lines log of failures, fallbacks and fixes."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else LOGS_DIR / "fallback_log.jsonl"

    def record(self, component: str, event: str, detail: str, severity: str = "warning") -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": component,
            "event": event,
            "detail": detail,
            "severity": severity,
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        get_logger("fallback").log(logging.WARNING if severity == "warning" else logging.ERROR,
                                  "[%s] %s: %s", component, event, detail)
        return entry

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries
