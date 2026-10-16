"""Export tables and figures produced by actual runs. No fabricated content:
every artifact here must be generated from real computation results.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display needed
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.helpers import FIGURES_DIR, TABLES_DIR


def save_table(df: pd.DataFrame, name: str) -> str:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def save_bar_chart(values: dict, title: str, name: str, ylabel: str = "value") -> str:
    """Fallback-safe chart: returns the PNG path, or saves the table and
    returns 'table-only' when plotting fails."""
    try:
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(values))), dpi=110)
        keys = list(values)
        ax.barh([str(k) for k in keys], [float(values[k]) for k in keys], color="#3b6ea5")
        ax.set_title(title)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
        fig.tight_layout()
        path = FIGURES_DIR / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    except Exception:
        tmp = pd.DataFrame({"key": list(values), "value": list(values.values())})
        save_table(tmp, name + "_fallback_table")
        return "table-only"


def recommendations_to_df(recs: list) -> pd.DataFrame:
    return pd.DataFrame([r.as_dict() for r in recs])


def load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / f"{name}.csv")
