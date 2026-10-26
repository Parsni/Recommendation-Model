"""Matplotlib charts for the routing app (obsidian-dark theme).

Every helper takes plain engine/app objects and returns a ready-to-render
``matplotlib.figure.Figure``; the Streamlit layer only calls ``st.pyplot``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.routing.inference.classifier_ensemble import ModelSuccessProbability  # noqa: E402
from src.routing.scoring import CandidateScoreRow  # noqa: E402

# Obsidian-dark palette
BG = "#0B111E"
AX_BG = "#0F172A"
TEXT = "#E2E8F0"
MUTED = "#94A3B8"
GRID = "#1E293B"
BAR = "#38BDF8"
WIN = "#10B981"
WARN = "#F59E0B"
CMAP = LinearSegmentedColormap.from_list(
    "obsidian", ["#0F172A", "#1D4ED8", "#38BDF8", "#7DD3FC"]
)


def _style_axes(ax) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, axis="x" if len(ax.get_yticklabels()) else "both", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _finish(fig, title: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    fig.patch.set_facecolor(BG)
    if title:
        fig.suptitle(title, color=TEXT, fontsize=12, fontweight="bold", y=0.97)
    if xlabel:
        fig.axes[0].set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        fig.axes[0].set_ylabel(ylabel, color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _short(model_id: str) -> str:
    names = {
        "claude-3-5-sonnet": "Claude 3.5 Sonnet",
        "deepseek-r1": "DeepSeek R1",
        "gemini-1.5-pro": "Gemini 1.5 Pro",
        "gpt-4o": "GPT-4o",
        "gpt-4o-mini": "GPT-4o Mini",
        "llama-3.1-70b": "Llama 3.1 70B",
        "qwen3-235b-a22b": "Qwen3 235B",
    }
    return names.get(model_id, model_id)


def chart_success_bars(
    success_probs: Dict[str, ModelSuccessProbability],
    winner_id: str,
    title: str = "Predicted Success Probability by Model",
):
    """Horizontal bars: p_success per candidate, selected model highlighted."""
    rows = sorted(success_probs.values(), key=lambda r: r.p_success)
    labels = [_short(r.model_id) for r in rows]
    values = [r.p_success for r in rows]
    colors = [WIN if r.model_id == winner_id else BAR for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor(BG)
    ax.barh(labels, values, color=colors, height=0.6)
    _style_axes(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.grid(False, axis="y")
    ax.set_xlim(0, 1.0)
    for y, (r, v) in enumerate(zip(rows, values)):
        ci = r.confidence_interval
        ax.plot([ci[0], ci[1]], [y, y], color=TEXT, linewidth=1.0, alpha=0.45)
        ax.text(v + 0.015, y, f"{v:.2f}", va="center", color=TEXT, fontsize=9)
    return _finish(fig, title, xlabel="P(success) with pseudo confidence interval")


def chart_composite_bars(
    ranked: Sequence[CandidateScoreRow],
    winner_id: str,
    title: str = "Composite Score by Model",
):
    """Horizontal bars: final composite score, winner highlighted."""
    rows = list(ranked)
    rows_sorted = sorted(rows, key=lambda r: r.composite_score)
    labels = [_short(r.model_id) for r in rows_sorted]
    values = [r.composite_score for r in rows_sorted]
    colors = [WIN if r.model_id == winner_id else BAR for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor(BG)
    ax.barh(labels, values, color=colors, height=0.6)
    _style_axes(ax)
    ax.grid(False, axis="y")
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    for y, v in enumerate(values):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", color=TEXT, fontsize=9)
    return _finish(fig, title, xlabel="Weighted composite score")


def chart_cost_latency_scatter(
    ranked: Sequence[CandidateScoreRow],
    winner_id: str,
    title: str = "Cost vs Latency Trade-off",
):
    """Scatter of predicted cost/latency; marker size ~ composite, winner annotated."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    xs = [r.raw_latency_ms for r in ranked]
    ys = [r.raw_cost_usd for r in ranked]
    sizes = [80 + 600 * r.composite_score for r in ranked]
    colors = [WIN if r.model_id == winner_id else BAR for r in ranked]
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.85, edgecolors=TEXT, linewidths=0.8)
    # place labels in staggered vertical bands ordered by latency so that
    # neighbours on the x-axis get very different label heights
    cycle = [-22, 8, -8, 22, -15, 15, 0]
    x_order = sorted(range(len(ranked)), key=lambda i: ranked[i].raw_latency_ms)
    for slot, i in enumerate(x_order):
        r = ranked[i]
        ax.annotate(
            _short(r.model_id) + (" ★" if r.model_id == winner_id else ""),
            (r.raw_latency_ms, r.raw_cost_usd),
            textcoords="offset points",
            xytext=(0, cycle[slot % len(cycle)]),
            ha="center",
            color=TEXT,
            fontsize=8.5,
        )
    _style_axes(ax)
    ax.grid(True, axis="both", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_xlim(min(xs) * 0.7, max(xs) * 1.25)
    lo, hi = min(ys), max(ys)
    pad = max((hi - lo) * 0.25, 0.05)
    ax.set_ylim(lo - pad, hi + pad)
    return _finish(fig, title, xlabel="Predicted latency (ms)", ylabel="Predicted cost (USD)")


def chart_confusion_matrix(
    confusion: List[List[int]],
    model_order: List[str],
    title: str = "Expected vs Selected (held-out prompts)",
):
    """Rows = empirical best (expected), cols = router pick."""
    mat = np.array(confusion, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(mat, cmap=CMAP, aspect="auto")
    labels = [_short(m) for m in model_order]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", color=MUTED, fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels, color=MUTED, fontsize=8.5)
    ax.set_xlabel("Router selected", color=MUTED, fontsize=9)
    ax.set_ylabel("Expected (empirical best)", color=MUTED, fontsize=9)
    thresh = mat.max() / 2 if mat.size and mat.max() else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            ax.text(
                j, i, str(v),
                ha="center", va="center",
                color=TEXT if mat[i, j] > thresh else MUTED,
                fontsize=9,
            )
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    fig.tight_layout()
    return fig


def chart_domain_accuracy(
    per_domain_a: Dict[str, float],
    domain_counts: Dict[str, int],
    threshold: float = 0.90,
    title: str = "Metric A by Task Category",
):
    """Bar per domain with the 0.90 target line; counts annotated."""
    items = [(d, v) for d, v in per_domain_a.items() if v is not None and domain_counts.get(d)]
    items.sort(key=lambda kv: -kv[1])
    labels = [d for d, _ in items]
    values = [v for _, v in items]
    counts = [domain_counts[d] for d in labels]
    colors = [WIN if v >= threshold else BAR for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor(BG)
    ax.bar(labels, values, color=colors, width=0.6)
    _style_axes(ax)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.grid(False, axis="x")
    ax.axhline(threshold, color=WARN, linestyle="--", linewidth=1.2, label=f"target {threshold:.0%}")
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", rotation=20)
    for x, (v, n) in enumerate(zip(values, counts)):
        ax.text(x, v + 0.03, f"{v:.0%}\nn={n}", ha="center", color=TEXT, fontsize=8.5)
    ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5)
    return _finish(fig, title, xlabel=None)


def chart_win_rate_heatmap(
    domain_win_rates: Dict[str, Dict[str, float]],
    model_order: List[str],
    title: str = "Real Battle Win-rate: Task Category × Model",
):
    """Train-split shrunk win-rates per (domain, model) from actual LMArena battles."""
    domains = [d for d in domain_win_rates.keys()]
    mat = np.array([[domain_win_rates[d].get(m, 0.5) for m in model_order] for d in domains])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(mat, cmap=CMAP, aspect="auto", vmin=0.2, vmax=0.8)
    ax.set_xticks(range(len(model_order)), [_short(m) for m in model_order], rotation=35, ha="right", color=MUTED, fontsize=8.5)
    ax.set_yticks(range(len(domains)), domains, color=MUTED, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=TEXT, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID)
    fig.tight_layout()
    return fig


def chart_overall_win_rates(
    overall_stats: Dict[str, Dict[str, float]],
    title: str = "Real Arena Win-rate (all held-in battles)",
):
    """Appearance-based real win-rate per model with game counts."""
    items = sorted(overall_stats.items(), key=lambda kv: kv[1]["win_rate"])
    labels = [_short(m) for m, _ in items]
    values = [s["win_rate"] for _, s in items]
    games = [int(s["games"]) for _, s in items]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor(BG)
    ax.barh(labels, values, color=BAR, height=0.6)
    _style_axes(ax)
    ax.grid(False, axis="y")
    ax.set_xlim(0, 1.0)
    ax.axvline(0.5, color=MUTED, linestyle=":", linewidth=1.0)
    for y, (v, n) in enumerate(zip(values, games)):
        ax.text(v + 0.015, y, f"{v:.1%}  ({n} battles)", va="center", color=TEXT, fontsize=9)
    return _finish(fig, title, xlabel="Win-rate (appearances basis, real LMArena battles)")


# Diverging palette for signed (centered-logit) affinity values
AFF_CMAP = LinearSegmentedColormap.from_list(
    "obsidian_div", ["#F87171", "#7F1D1D", "#0F172A", "#1D4ED8", "#7DD3FC"]
)

CONTRIB_COLORS = ["#38BDF8", "#818CF8", "#34D399", "#FBBF24"]


def chart_affinity_heatmap(
    domain_affinities: Dict[str, Dict[str, float]],
    model_order: List[str],
    highlight_domain: Optional[str] = None,
    title: str = "Learned Domain Affinity Matrix (centered logits)",
):
    """Models x domains heatmap of learned per-domain utility affinities."""
    domains = [d for d in ("math", "code", "creative", "translation", "doc", "comm", "qa", "general")
               if any(d in domain_affinities.get(m, {}) for m in model_order)]
    if not domains:
        domains = sorted({d for m in model_order for d in domain_affinities.get(m, {})})
    mat = np.array([[domain_affinities.get(m, {}).get(d, 0.0) for d in domains] for m in model_order])
    vmax = max(abs(mat.min()), abs(mat.max())) if mat.size and mat.max() != mat.min() else 1.0

    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(mat, cmap=AFF_CMAP, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(domains)), domains, rotation=35, ha="right", color=MUTED, fontsize=8.5)
    ax.set_yticks(range(len(model_order)), [_short(m) for m in model_order], color=MUTED, fontsize=8.5)
    thresh = vmax / 2
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color=TEXT if abs(v) > thresh else MUTED, fontsize=7.5)
    if highlight_domain in domains:
        j = domains.index(highlight_domain)
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1.0, mat.shape[0],
                                   fill=False, edgecolor=WARN, linewidth=2.0))
    ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID)
    fig.tight_layout()
    return fig


def chart_contribution_breakdown(
    ranked: Sequence[CandidateScoreRow],
    weights: Any,
    winner_id: str,
    title: str = "Weighted Score Contribution by Criterion",
):
    """Stacked horizontal bars: w_i * component per candidate, sum = composite."""
    rows = sorted(ranked, key=lambda r: r.composite_score)
    labels = [_short(r.model_id) + (" ★" if r.model_id == winner_id else "") for r in rows]
    succ = [weights.w_success * r.success_prob for r in rows]
    sim = [weights.w_similarity * r.similarity for r in rows]
    cost = [weights.w_cost * r.cost_efficiency for r in rows]
    lat = [weights.w_latency * r.latency_efficiency for r in rows]
    totals = [r.composite_score for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.patch.set_facecolor(BG)
    left = np.zeros(len(rows))
    for values, color, lbl in (
        (succ, CONTRIB_COLORS[0], "Quality  w·P(Success)"),
        (sim, CONTRIB_COLORS[1], "Similarity  w·Sim"),
        (cost, CONTRIB_COLORS[2], "Cost  w·CostEff"),
        (lat, CONTRIB_COLORS[3], "Speed  w·LatEff"),
    ):
        ax.barh(labels, values, left=left, color=color, height=0.6, label=lbl)
        left = left + np.array(values)
    _style_axes(ax)
    ax.grid(False, axis="y")
    ax.set_xlim(0, max(totals) * 1.18 if totals else 1.0)
    for y, v in enumerate(totals):
        ax.text(v + 0.01, y, f"{v:.3f}", va="center", color=TEXT, fontsize=9)
    ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8, loc="lower right")
    return _finish(fig, title, xlabel="Weighted contribution (components sum to composite)")
