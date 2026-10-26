"""Dual-audience presentation panels for the Streamlit router app.

Three experience levels (Simple / Standard / Expert) expose progressively
deeper layers of the same routing decision. All numbers rendered here come
from live engine outputs — nothing is hardcoded or rounded up.

Pure helpers (classify_tier, cost_context_line, preset_caption, ...) are
side-effect free so they can be unit-tested without a Streamlit runtime.
"""
from __future__ import annotations

import json
import math
from typing import Callable, Dict, List, Sequence

import pandas as pd
import streamlit as st

from src.routing.domains import DOMAINS
from src.routing.featurizer import StructuralFeatures
from src.routing.anchor_similarity import AnchorMatchResult
from src.routing.context_bridge import EnrichedContext
from src.routing.inference.classifier_ensemble import (
    LogisticRegressionEnsemble,
    ModelSuccessProbability,
)
from src.routing.inference.resource_regressors import ResourceForecast
from src.routing.reconciliation import ReconciliationEngine, RoutingDecision
from src.routing.scoring import CandidateScoreRow, MultiCriteriaOptimizer, RoutingPolicyWeights

# -----------------------------------------------------------------------------
# MODES & PLAIN-LANGUAGE MAPPINGS
# -----------------------------------------------------------------------------
MODE_LABELS: Dict[str, str] = {
    "👶 Simple": "simple",
    "Standard": "standard",
    "🔧 Expert": "expert",
}

MODE_HINTS: Dict[str, str] = {
    "simple": "Plain-language results, everyday explanations, and friendly defaults.",
    "standard": "Full results plus ML internals, a what-if sandbox, and deeper panels.",
    "expert": "Everything: latent utilities, affinity matrices, audit trails, exports, telemetry.",
}

PRESET_PLAIN: Dict[str, str] = {
    "Balanced & General Purpose (Recommended)": "Best all-rounder — a safe default when you are not sure.",
    "High-Precision Flagship (Quality Benchmark)": "Maximise answer quality; cost and speed matter less.",
    "Deep Mathematical & Logical Derivation": "Bias towards models with the strongest maths/logic battle record.",
    "Specialized Code & Architecture Engineering": "Bias towards strong coding models with low latency on code tasks.",
    "Ultra-Fast Throughput (Sub-Second Latency)": "Optimise for the fastest reply; quality matters less.",
    "Cost-Constrained / Free Tier (Budget Optimization)": "Optimise for the lowest cost per question.",
}

DOMAIN_PLAIN: Dict[str, str] = {
    "math": "a maths or logic problem",
    "code": "a coding or software task",
    "creative": "a creative writing request",
    "translation": "a translation task",
    "doc": "a long-document or summarisation task",
    "comm": "an email or business-writing task",
    "qa": "a quick factual question",
    "general": "a general-purpose question",
}

GLOSSARY: List[tuple] = [
    ("Composite score",
     "A single number (0-1) blending how well the model is expected to do on your prompt, "
     "how similar your prompt is to known examples, and its predicted cost and speed. "
     "Highest composite wins."),
    ("P(Success)",
     "Estimated probability (0-1) that the model answers your prompt well, learned from "
     "thousands of real head-to-head human battles on LMArena."),
    ("Bradley-Terry",
     "A classic statistics model that turns pairwise win/loss records into per-model "
     "strength ratings ('latent utilities'). Here it drives most of P(Success)."),
    ("Win-rate",
     "Share of battles a model won among the battles it actually fought in the dataset."),
    ("Held-out",
     "A slice of data deliberately kept out of training and used only for honest evaluation."),
    ("Metric A (task-category agreement)",
     "How often the router picks the same model an expert would: the empirical best model "
     "for the prompt's task category, judged by real battle win-rates."),
    ("Metric B (head-to-head agreement)",
     "Whether the router ranks the human-declared winner of a real battle above the loser."),
    ("Failsafe gate",
     "If the top pick's probability or composite score is too low, routing bypasses to a "
     "safe fallback model (GPT-4o) instead of dispatching a weak choice."),
    ("Domain affinity",
     "Learned per-category preference strengths: how relatively favoured each model is in "
     "maths, code, creative writing, and so on, derived from real battle outcomes."),
]

_TIER_COLORS = {
    "good": "#10B981",
    "fair": "#F59E0B",
    "poor": "#F87171",
}


# -----------------------------------------------------------------------------
# PURE HELPERS (unit-tested)
# -----------------------------------------------------------------------------
def classify_tier(value: float, reference_median: float, lower_is_better: bool = True) -> str:
    """Classify value as cheap/fair/pricy (or fast/average/slow) vs a roster median."""
    if reference_median <= 0:
        return "fair"
    ratio = value / reference_median if reference_median else 1.0
    if lower_is_better:
        if ratio <= 0.5:
            return "cheap"
        if ratio <= 1.5:
            return "fair"
        return "pricy"
    else:
        if ratio <= 0.5:
            return "fast"
        if ratio <= 1.5:
            return "average"
        return "slow"


def cost_context_line(cost_usd: float) -> str:
    """Everyday framing of a per-question USD cost."""
    if cost_usd <= 0:
        return "free per question"
    per_dollar = 1.0 / cost_usd
    cost_str = f"${cost_usd:.6f}" if cost_usd < 0.001 else f"${cost_usd:.4f}"
    return f"about {cost_str} per question — roughly {per_dollar:,.0f} questions for $1"


def format_latency_line(latency_ms: float) -> str:
    """Everyday framing of a predicted latency."""
    if latency_ms < 600:
        return "replies in well under a second"
    if latency_ms < 1200:
        return "replies in about half a second to a second"
    if latency_ms < 3000:
        return f"replies in about {latency_ms / 1000:.1f} seconds"
    return f"replies in about {latency_ms / 1000:.0f} seconds"


def preset_caption(preset_name: str) -> str:
    """One-line plain-English explanation of an optimization preset."""
    return PRESET_PLAIN.get(preset_name, "")


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
def render_sidebar(
    engine,
    accuracy_report,
    profile_of: Callable[[str], dict],
) -> str:
    """Render the global sidebar; returns the active mode key."""
    with st.sidebar:
        st.markdown("### 🧭 Experience Level")
        label = st.radio(
            "Experience Level",
            options=list(MODE_LABELS.keys()),
            index=1,
            key="exp_mode_radio",
            label_visibility="collapsed",
            help="Simple = plain language · Standard = deeper panels · Expert = full ML internals",
        )
        mode = MODE_LABELS.get(label, "standard")
        st.caption(MODE_HINTS[mode])

        st.markdown("---")
        st.markdown("### 🤔 What is this?")
        st.markdown(
            "Paste any question — code, maths, an email draft, anything. "
            "The engine studies thousands of real head-to-head AI battles and picks "
            "**which chatbot is most likely to answer it well**. No API keys needed; "
            "everything runs offline in your browser."
        )

        n_battles = 0
        if accuracy_report is not None:
            n_battles = accuracy_report.n_train + accuracy_report.n_test
        chips = f"`{len(engine.model_names)} models` `{n_battles:,} real battles`"
        if accuracy_report is not None:
            chips += f"`A = {accuracy_report.metric_a:.2%}`"
        st.markdown(chips)

        try:
            pending = len(engine.repository.get_unprocessed_feedback())
            if pending:
                st.caption(f"📮 {pending} feedback item(s) waiting for the retraining loop")
        except Exception:
            pass

        with st.expander("📚 Plain-English glossary & FAQ"):
            for term, definition in GLOSSARY:
                st.markdown(f"**{term}** — {definition}")
    return mode


# -----------------------------------------------------------------------------
# LAYMAN PANELS
# -----------------------------------------------------------------------------
def render_plain_summary(
    decision: RoutingDecision,
    top_cand: CandidateScoreRow,
    domain: str,
    ranked: Sequence[CandidateScoreRow],
    profile_of: Callable[[str], dict],
    win_stats: Dict[str, dict],
) -> None:
    """'What this means for you' card — everyday language + traffic-light chips."""
    prof = profile_of(decision.selected_model)
    costs = [c.raw_cost_usd for c in ranked]
    lats = [c.raw_latency_ms for c in ranked]
    median_cost = sorted(costs)[len(costs) // 2]
    median_lat = sorted(lats)[len(lats) // 2]

    cost_tier = classify_tier(top_cand.raw_cost_usd, median_cost, lower_is_better=True)
    lat_tier = classify_tier(top_cand.raw_latency_ms, median_lat, lower_is_better=False)

    cost_cls = {"cheap": "good", "fair": "fair", "pricy": "poor"}.get(cost_tier, "fair")
    lat_cls = {"fast": "good", "average": "fair", "slow": "poor"}.get(lat_tier, "fair")

    if len(ranked) > 1:
        margin = ranked[0].success_prob - ranked[1].success_prob
    else:
        margin = 1.0
    if margin > 0.05:
        conf_text, conf_cls = "high confidence", "good"
    else:
        conf_text, conf_cls = "close call between strong candidates", "fair"

    wr = win_stats.get(decision.selected_model, {}).get("win_rate")
    wr_txt = f" It wins about {wr:.0%} of the real battles it has fought." if wr else ""

    domain_words = DOMAIN_PLAIN.get(domain, "a general-purpose question")
    cluster_txt = (
        f" The engine grouped it as <strong>{decision.intent_cluster}</strong>."
        if decision.intent_cluster else ""
    )
    items = [
        f"<strong>Your request looks like</strong> {domain_words}.{cluster_txt}",
        f"<strong>Why this pick:</strong> {prof['name']} scored highest overall for this kind of task.{wr_txt}",
        f"<strong>Cost:</strong> {cost_context_line(top_cand.raw_cost_usd)}.",
        f"<strong>Speed:</strong> {format_latency_line(top_cand.raw_latency_ms)} "
        f"(about {top_cand.raw_latency_ms:.0f} ms predicted).",
    ]
    items_html = "".join(
        f"<li style='margin-bottom:6px;'>{t}</li>" for t in items
    )

    def _chip(label: str, text: str, cls: str) -> str:
        return (
            f"<span class='stat-chip' style='border:1px solid {_TIER_COLORS[cls]};'>"
            f"{label} <strong style='color:{_TIER_COLORS[cls]};'>{text}</strong></span>"
        )

    chips = (
        _chip("💰 Value:", f"{cost_tier} ({cost_context_line(top_cand.raw_cost_usd).split('—')[0].strip()})", cost_cls)
        + _chip("⚡ Speed:", f"{lat_tier} (~{top_cand.raw_latency_ms:.0f} ms)", lat_cls)
        + _chip("🎯 Certainty:", conf_text, conf_cls)
    )

    st.markdown(f"""
    <div class='report-card' style='border-left: 3px solid #38BDF8;'>
        <div class='report-title' style='color:#7DD3FC;'>🌍 In everyday terms</div>
        <ul style='color:#E2E8F0; line-height:1.6; margin:0; padding-left:18px;'>{items_html}</ul>
        <div style='margin-top:10px;'>{chips}</div>
    </div>
    """, unsafe_allow_html=True)


def render_plain_accuracy(accuracy_report) -> None:
    """One plain-English line next to the honest dual metrics."""
    plain_a = round(accuracy_report.metric_a * 100)
    st.caption(
        f"🌐 In everyday terms: for every 100 new prompts, the router picks the same "
        f"*type* of model an expert would pick about **{plain_a} times**. "
        f"(The exact unrounded number is shown above — nothing is inflated.)"
    )


def render_roster(
    profile_of: Callable[[str], dict],
    win_stats: Dict[str, dict],
    models_cfg: Dict[str, dict],
    compact: bool,
) -> None:
    """Who are these chatbots? Compact cards for Simple, spec table otherwise."""
    with st.expander("🤖 Who are these 7 chatbots?"):
        model_ids = list(models_cfg.keys()) or list(win_stats.keys())
        if compact:
            for mid in model_ids:
                prof = profile_of(mid)
                wr = win_stats.get(mid, {})
                wr_txt = (
                    f" · wins {wr['win_rate']:.0%} of {int(wr['games'])} real battles"
                    if wr else ""
                )
                st.markdown(f"""
                - **{prof['name']}** ({prof['provider']}) — *{prof['tier']}*{wr_txt}. {prof['best_for']}
                """)
        else:
            rows = []
            for mid in model_ids:
                prof = profile_of(mid)
                cfg = models_cfg.get(mid, {})
                wr = win_stats.get(mid, {})
                rows.append({
                    "Model": prof["name"],
                    "Provider": prof["provider"],
                    "Tier": prof["tier"],
                    "Real WR": f"{wr['win_rate']:.1%}" if wr else "—",
                    "Battles": str(int(wr.get("games", 0))),
                    "$/M in": str(cfg.get("cost_input_per_million", "—")),
                    "$/M out": str(cfg.get("cost_output_per_million", "—")),
                    "Base ms": str(cfg.get("baseline_latency_ms", "—")),
                    "Failsafe": "✓" if cfg.get("is_fallback_candidate") else "",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# EXPERT PANELS
# -----------------------------------------------------------------------------
def render_ensemble_internals(
    success_probs: Dict[str, ModelSuccessProbability],
    ensemble: LogisticRegressionEnsemble,
    domain: str,
    winner_id: str,
) -> None:
    """Per-model latent utility, BT probability, blended P + CI, affinities, WR."""
    utilities = {m: sp.latent_utility for m, sp in success_probs.items()}
    mean_u = sum(utilities.values()) / max(1, len(utilities))

    rows = []
    for m, sp in sorted(success_probs.items(), key=lambda kv: -kv[1].p_success):
        delta = sp.latent_utility - mean_u
        p_bt = 1.0 / (1.0 + math.exp(-delta))
        blend = ensemble.bt_blend
        if blend < 1.0:
            p_pw = (sp.p_success - blend * p_bt) / (1.0 - blend)
        else:
            p_pw = None
        aff = ensemble.domain_affinities.get(m, {}).get(domain)
        rows.append({
            "Model": m,
            "Latent utility U(m,x)": f"{sp.latent_utility:.4f}",
            "BT prob σ(U−Ū)": f"{p_bt:.4f}",
            "Pointwise prob": f"{p_pw:.4f}" if p_pw is not None else "—",
            "Blended P(Success)": f"{sp.p_success:.4f}",
            "95% CI": f"[{sp.confidence_interval[0]:.3f}, {sp.confidence_interval[1]:.3f}]",
            "Domain affinity": f"{aff:.3f}" if aff is not None else "—",
            "Train WR": f"{ensemble.empirical_win_rates.get(m, 0.5):.1%}",
            "★": "✓" if m == winner_id else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Detected domain: **{domain}** · Ensemble = {ensemble.bt_blend:.0%} Bradley-Terry / "
        f"{1 - ensemble.bt_blend:.0%} pointwise · C={ensemble.regularization_c} · "
        f"affinity scale={ensemble.domain_affinity_scale} · min games={ensemble.domain_min_games}. "
        "Utilities U(m,x) = w·z + b + domain affinity; BT prob = σ(U−mean U); "
        "blended P clips to [0.05, 0.98]."
    )


def render_sandbox(
    optimizer: MultiCriteriaOptimizer,
    success_probs: Dict[str, ModelSuccessProbability],
    resource_forecasts: Dict[str, ResourceForecast],
    anchor_similarity: float,
    original_ranked: Sequence[CandidateScoreRow],
    original_weights: RoutingPolicyWeights,
    critical_threshold_default: float,
    profile_of: Callable[[str], dict],
) -> None:
    """Interactive what-if re-ranking from cached predictions (no retraining)."""
    st.markdown("#### 🧪 What-If Sandbox — re-rank with your own priorities")
    st.caption(
        "Move the sliders and the engine re-ranks the same cached predictions instantly "
        "(deterministic — no retraining, no API calls). Δ rank = sandbox rank − original rank; "
        "negative Δ means the model improved."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        w_s = st.slider("Quality weight", 0.0, 1.0, float(original_weights.w_success), 0.01, key="sb_ws")
    with c2:
        w_sim = st.slider("Similarity weight", 0.0, 1.0, float(original_weights.w_similarity), 0.01, key="sb_wsim")
    with c3:
        w_c = st.slider("Cost weight", 0.0, 1.0, float(original_weights.w_cost), 0.01, key="sb_wc")
    with c4:
        w_l = st.slider("Speed weight", 0.0, 1.0, float(original_weights.w_latency), 0.01, key="sb_wl")

    crit = st.slider(
        "Failsafe success-probability threshold",
        0.0, 1.0, float(critical_threshold_default), 0.01, key="sb_crit",
        help="If the sandbox winner's P(Success) drops below this line, the real engine would "
             "bypass it and dispatch the safe fallback instead.",
    )

    sandbox_weights = RoutingPolicyWeights(w_success=w_s, w_similarity=w_sim, w_cost=w_c, w_latency=w_l)
    norm = sandbox_weights.normalized()
    reranked = optimizer.evaluate_candidates(
        success_probs=success_probs,
        resource_forecasts=resource_forecasts,
        anchor_similarity=anchor_similarity,
        override_weights=norm,
    )

    orig_rank = {r.model_id: r.rank for r in original_ranked}
    orig_comp = {r.model_id: r.composite_score for r in original_ranked}
    delta_rows = []
    for r in reranked:
        o_rank = orig_rank.get(r.model_id)
        o_comp = orig_comp.get(r.model_id)
        delta_rows.append({
            "Model": profile_of(r.model_id)["name"],
            "Sandbox rank": str(r.rank),
            "Original rank": str(o_rank) if o_rank is not None else "—",
            "Δ rank": str(r.rank - o_rank) if o_rank is not None else "—",
            "Sandbox composite": f"{r.composite_score:.4f}",
            "Original composite": f"{o_comp:.4f}" if o_comp is not None else "—",
        })
    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

    new_leader = reranked[0]
    old_leader = original_ranked[0]
    leader_name = profile_of(new_leader.model_id)["name"]
    if new_leader.model_id != old_leader.model_id:
        st.success(
            f"**New leader under your sandbox weights: {leader_name}** "
            f"(original pick was {profile_of(old_leader.model_id)['name']})"
        )
    else:
        st.info(f"**Leader unchanged: {leader_name}** — still #1 under your sandbox weights.")

    if new_leader.success_prob < crit:
        st.warning(
            f"Failsafe simulation: {leader_name}'s P(Success) = {new_leader.success_prob:.3f} "
            f"is below your threshold ({crit:.2f}) — the real engine would dispatch the fallback model instead."
        )


def render_feature_inspector(
    structural: StructuralFeatures,
    anchor_match: AnchorMatchResult,
    enriched_ctx: EnrichedContext,
) -> None:
    """Structural features, domain one-hot, cluster, nearest anchor."""
    st.markdown("#### 🔬 Feature Inspector")
    feats = {
        "char_length": str(structural.char_length),
        "word_count": str(structural.word_count),
        "estimated_tokens": str(structural.estimated_tokens),
        "language": structural.language,
        "code_indicator": f"{structural.code_indicator:.4f}",
        "math_indicator": f"{structural.math_indicator:.4f}",
        "question_indicator": str(structural.question_indicator),
        "uppercase_ratio": f"{structural.uppercase_ratio:.4f}",
        "digit_ratio": f"{structural.digit_ratio:.4f}",
        "avg_word_length": f"{structural.avg_word_length:.3f}",
    }
    st.dataframe(pd.DataFrame([feats]), use_container_width=True, hide_index=True)
    one_hot = {d: 1.0 if d == enriched_ctx.domain else 0.0 for d in DOMAINS}
    st.markdown(
        f"- **Detected domain one-hot:** `{one_hot}` (appended as the last block of the feature vector)\n"
        f"- **K-Means intent cluster:** `#{enriched_ctx.cluster_id}` — {enriched_ctx.cluster_label}\n"
        f"- **Nearest production anchor:** `{anchor_match.nearest_anchor_id}` "
        f"(category `{anchor_match.nearest_category}`, cosine sim `{anchor_match.max_similarity:.4f}`)\n"
        f"- **Nearest anchor text:** \"{anchor_match.nearest_text[:100]}…\"\n"
        f"- **Feature vector dimension:** `{len(enriched_ctx.feature_vector)}`"
    )


def render_model_specs(
    engine,
    resource_forecasts: Dict[str, ResourceForecast],
    profile_of: Callable[[str], dict],
) -> None:
    """Config economics table + this prompt's forecasts per model."""
    st.markdown("#### 💲 Cost & Latency Model Economics")
    rows = []
    for mid, cfg in engine.config.get("models", {}).items():
        fc = resource_forecasts.get(mid)
        rows.append({
            "Model": profile_of(mid)["name"],
            "$/M input": str(cfg.get("cost_input_per_million", "—")),
            "$/M output": str(cfg.get("cost_output_per_million", "—")),
            "Baseline ms": str(cfg.get("baseline_latency_ms", "—")),
            "Per-token ms": str(cfg.get("per_token_latency_ms", "—")),
            "Forecast output tokens": str(getattr(fc, "estimated_output_tokens", "—") if fc else "—"),
            "Predicted $": f"{fc.predicted_cost_usd:.6f}" if fc else "—",
            "Predicted ms": f"{fc.predicted_latency_ms:.1f}" if fc else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Forecast columns are Ridge-regressor predictions for *this* prompt; "
               "cost columns are the configured per-million-token API prices.")


def render_audit_panel(decision: RoutingDecision, reconciliation: ReconciliationEngine) -> None:
    """Failsafe gate status + audit trail."""
    st.markdown("#### 🛡️ Failsafe Gate & Audit Trail")
    if decision.is_fallback_triggered:
        st.error(f"Fallback **triggered**: {decision.fallback_reason}")
    else:
        top = decision.ranked_candidates[0] if decision.ranked_candidates else None
        if top and top.success_prob >= reconciliation.critical_success_threshold:
            st.success(
                f"Fallback **not** needed: top pick P(Success)={top.success_prob:.4f} ≥ "
                f"threshold {reconciliation.critical_success_threshold:.2f}."
            )
        else:
            st.warning("Fallback status indeterminate (no candidates).")
    st.json(decision.audit_trail)


def render_decision_export(
    decision: RoutingDecision,
    sanitized_prompt: str,
    enriched_ctx: EnrichedContext,
    structural: StructuralFeatures,
    anchor_match: AnchorMatchResult,
    weights: RoutingPolicyWeights,
) -> None:
    """Downloadable decision JSON + API-style payload preview."""
    st.markdown("#### 📦 Decision Export")
    payload = {
        "request": {
            "user_id": decision.user_id,
            "prompt": sanitized_prompt,
            "policy_weights": {
                "w_success": weights.w_success,
                "w_similarity": weights.w_similarity,
                "w_cost": weights.w_cost,
                "w_latency": weights.w_latency,
            },
        },
        "features": {
            "char_length": structural.char_length,
            "estimated_tokens": structural.estimated_tokens,
            "code_indicator": structural.code_indicator,
            "math_indicator": structural.math_indicator,
            "domain": enriched_ctx.domain,
            "cluster_id": enriched_ctx.cluster_id,
            "cluster_label": enriched_ctx.cluster_label,
            "anchor_id": anchor_match.nearest_anchor_id,
            "anchor_similarity": round(anchor_match.max_similarity, 4),
            "feature_vector_dim": len(enriched_ctx.feature_vector),
        },
        "decision": {
            "decision_id": decision.decision_id,
            "selected_model": decision.selected_model,
            "composite_score": decision.top_composite_score,
            "success_prob": decision.top_success_prob,
            "routing_latency_ms": decision.routing_latency_ms,
            "is_fallback_triggered": decision.is_fallback_triggered,
            "fallback_reason": decision.fallback_reason,
            "audit_trail": decision.audit_trail,
        },
        "candidates": [
            {
                "model_id": r.model_id,
                "rank": r.rank,
                "composite_score": r.composite_score,
                "success_prob": r.success_prob,
                "cost_usd": r.raw_cost_usd,
                "latency_ms": r.raw_latency_ms,
            }
            for r in decision.ranked_candidates
        ],
    }
    blob = json.dumps(payload, indent=2)
    st.download_button(
        "⬇️ Download full decision JSON",
        data=blob,
        file_name=f"routing_decision_{decision.decision_id[:8]}.json",
        mime="application/json",
    )
    with st.expander("API-style payload preview"):
        st.code(blob, language="json")


def render_hyperparams(engine, accuracy_report) -> None:
    """Hyperparameters, training provenance, and honest top-line metrics."""
    st.markdown("#### ⚙️ Hyperparameters & Provenance")
    inf = engine.config.get("inference", {})
    weights = engine.config.get("default_weights", {})
    rows = [
        {"Key": "inference.regularization_c", "Value": inf.get("regularization_c")},
        {"Key": "inference.bt_blend", "Value": inf.get("bt_blend")},
        {"Key": "inference.domain_affinity_scale", "Value": inf.get("domain_affinity_scale")},
        {"Key": "inference.domain_min_games", "Value": inf.get("domain_min_games")},
        {"Key": "inference.training_data", "Value": inf.get("training_data")},
        {"Key": "default_weights.w_success", "Value": weights.get("w_success")},
        {"Key": "default_weights.w_similarity", "Value": weights.get("w_similarity")},
        {"Key": "default_weights.w_cost", "Value": weights.get("w_cost")},
        {"Key": "default_weights.w_latency", "Value": weights.get("w_latency")},
        {"Key": "failsafe.critical_success_threshold",
         "Value": engine.config.get("failsafe", {}).get("critical_success_threshold")},
        {"Key": "failsafe.critical_composite_threshold",
         "Value": engine.config.get("failsafe", {}).get("critical_composite_threshold")},
        {"Key": "clustering.random_state", "Value": engine.config.get("clustering", {}).get("random_state")},
    ]
    if accuracy_report is not None:
        rows += [
            {"Key": "eval.n_train", "Value": accuracy_report.n_train},
            {"Key": "eval.n_test", "Value": accuracy_report.n_test},
            {"Key": "eval.metric_a (unrounded)", "Value": accuracy_report.metric_a},
            {"Key": "eval.metric_b (unrounded)", "Value": accuracy_report.metric_b},
            {"Key": "eval.cv_metric_a_mean",
             "Value": accuracy_report.cv_metric_a_mean},
            {"Key": "eval.cv_metric_a_std",
             "Value": accuracy_report.cv_metric_a_std},
        ]
    # stringify: mixed float/str columns break pyarrow-backed dataframe rendering
    for r in rows:
        r["Value"] = "—" if r["Value"] is None else str(r["Value"])
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_telemetry(engine) -> None:
    """Recent persisted decisions + unprocessed feedback (best-effort)."""
    st.markdown("#### 📡 Telemetry (local SQLite)")
    try:
        decisions = engine.repository.get_recent_decisions(limit=10)
        if decisions:
            st.dataframe(pd.DataFrame(decisions), use_container_width=True, hide_index=True)
        else:
            st.caption("No persisted decisions yet in the telemetry database.")
    except Exception as exc:
        st.caption(f"Telemetry unavailable: {exc}")
    try:
        pending = engine.repository.get_unprocessed_feedback()
        if pending:
            st.markdown(f"**Unprocessed feedback ({len(pending)}):**")
            st.dataframe(pd.DataFrame(pending), use_container_width=True, hide_index=True)
        else:
            st.caption("No pending feedback rows.")
    except Exception:
        pass
