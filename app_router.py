"""Multi-Criteria LLM Selection & Contextual Routing Engine.

Obsidian Dark-Themed User Experience with Automated Buffering Slideshow:
1. Fresh Un-Evaluated Initial State (No hardcoded pre-selection - Surprise Element)
2. Multi-Select Task Onboarding (10 prompt categories) & Custom Prompt Input
3. 6 Strategic Optimization Presets (plain-English explanations)
4. Automated 6-Stage Buffering Slideshow (persistent final trail)
5. Score-Based Final Animated Surprise Reveal + "In everyday terms" card
6. Honest Dual-Metric Accuracy Report (held-out LMArena battles - Metric A & B)
7. Decision Analytics (success, composite, cost/latency, contribution charts)
8. Under the Hood (Standard/Expert): ensemble internals, what-if sandbox,
   domain-affinity heatmap, feature inspector, audit trail, JSON export,
   hyperparameters & telemetry
9. Full 7-Model Candidate Comparison Deck (real Arena win-rates) + roster
10. Interactive Offline Chat Assistant (rule-based, no API calls)
11. Dedicated Feedback Page & Continuous Retraining Loop
12. Interactive Start Over / Reset State Machine
13. Global sidebar Experience Level: 👶 Simple / Standard / 🔧 Expert

All accuracy numbers come from real LMArena battle data evaluated on held-out
prompts (see scripts/build_accuracy_report.py). Nothing is fabricated.
"""
from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st

from src.routing import charts as ch
from src.routing import chat_assistant as chat
from src.routing import ui_panels as ui
from src.routing.engine import LLMSelectionRoutingEngine
from src.routing.evaluation import (
    RoutingAccuracyReport,
    load_anchors,
    load_battles,
    resolve_dataset_path,
)
from src.routing.scoring import RoutingPolicyWeights
from src.utils.helpers import PROJECT_ROOT

st.set_page_config(
    page_title="Contextual LLM Router",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# SLEEK OBSIDIAN DARK THEME & KEYFRAME ANIMATIONS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Obsidian Dark Theme */
    .stApp {
        background-color: #060911;
        color: #E2E8F0;
    }
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 4rem;
        max-width: 1220px;
    }

    /* Enterprise Header */
    .app-header {
        background: linear-gradient(180deg, #0C1220 0%, #070B14 100%);
        border: 1px solid #1B263B;
        border-radius: 12px;
        padding: 22px 26px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    .app-title {
        font-size: 1.65rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        margin-bottom: 3px;
    }
    .app-subtitle {
        font-size: 0.92rem;
        color: #94A3B8;
    }

    /* Automated Buffering / Logging-in Terminal Console */
    .buffering-box {
        background: #080D1A;
        border: 1px solid #1E293B;
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 20px 0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
        animation: fadeIn 0.3s ease-out;
    }
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(6px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .buffering-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .buffering-title {
        font-family: monospace;
        font-size: 0.82rem;
        font-weight: 700;
        color: #38BDF8;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .buffering-pct {
        font-family: monospace;
        font-size: 0.95rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .buffering-stage-name {
        font-size: 1.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .buffering-status {
        font-size: 0.92rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .buffering-stats {
        font-family: monospace;
        font-size: 0.85rem;
        color: #64748B;
        background: #0B1220;
        border: 1px solid #1A2740;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 14px;
    }
    .buffering-bar-bg {
        width: 100%;
        height: 8px;
        background: #111A2E;
        border-radius: 4px;
        overflow: hidden;
    }
    .buffering-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #0284C7 0%, #38BDF8 100%);
        transition: width 0.35s ease;
    }

    /* Persistent final pipeline trail */
    .trail-box {
        background: #080D1A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 16px 0;
    }
    .trail-stage {
        display: flex;
        align-items: baseline;
        gap: 10px;
        padding: 5px 0;
        border-bottom: 1px solid #111A2E;
        font-size: 0.88rem;
    }
    .trail-stage:last-child { border-bottom: none; }
    .trail-check { color: #10B981; font-weight: 800; }
    .trail-name { color: #E2E8F0; font-weight: 600; min-width: 320px; }
    .trail-stats { color: #64748B; font-family: monospace; font-size: 0.78rem; }

    /* Pipeline Complete Banner */
    .complete-banner {
        background: rgba(16, 185, 129, 0.10);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #10B981;
        border-radius: 8px;
        padding: 10px 16px;
        font-family: monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 20px;
        letter-spacing: 0.05em;
    }

    /* Surprise Reveal Box */
    .reveal-box {
        background: linear-gradient(135deg, #091224 0%, #101B34 100%);
        border: 2px solid #38BDF8;
        border-radius: 14px;
        padding: 26px;
        margin: 24px 0;
        box-shadow: 0 0 35px rgba(56, 189, 248, 0.22);
        animation: pulseNeon 3s infinite alternate ease-in-out, popIn 0.5s ease-out;
    }
    @keyframes pulseNeon {
        0% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.15); border-color: #38BDF8; }
        100% { box-shadow: 0 0 38px rgba(56, 189, 248, 0.38); border-color: #60A5FA; }
    }
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.96); }
        100% { opacity: 1; transform: scale(1.0); }
    }

    .reveal-tag {
        display: inline-block;
        background: #10B981;
        color: #060911;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 3px 10px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
    .reveal-name {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
    }
    .reveal-provider {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 16px;
    }
    .reveal-quote {
        background: rgba(15, 23, 42, 0.65);
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 1.02rem;
        color: #F1F5F9;
        margin-bottom: 18px;
        line-height: 1.55;
    }

    /* Metric Badges */
    .stat-chip {
        display: inline-block;
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 8px 14px;
        font-size: 0.85rem;
        color: #94A3B8;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .stat-chip strong {
        color: #38BDF8;
        font-family: monospace;
        font-size: 0.95rem;
    }

    /* Accuracy Report Box */
    .report-card {
        background: #0B111E;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 24px;
    }
    .report-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 12px;
    }
    .report-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-bottom: 14px;
    }
    .report-metric-tile {
        background: #0F172A;
        border: 1px solid #1E2C48;
        border-radius: 8px;
        padding: 12px 14px;
    }
    .report-metric-lbl {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .report-metric-num {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38BDF8;
    }

    /* Dual metric hero numbers */
    .metric-hero {
        background: #0B111E;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-hero .big {
        font-size: 2.4rem;
        font-weight: 800;
        color: #10B981;
        font-family: monospace;
    }
    .metric-hero .big.miss { color: #F59E0B; }
    .metric-hero .lbl {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-hero .sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 6px;
    }

    /* Horizontal Candidate Cards */
    .cand-card {
        background: #0B111E;
        border: 1px solid #1E293B;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .cand-card.is-winner {
        border: 2px solid #38BDF8;
        background: #0D1629;
    }
    .cand-rank {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 4px;
    }
    .cand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .cand-prov {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .cand-score {
        font-size: 1.35rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 8px;
    }
    .cand-line {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 3px;
    }
    .cand-line strong {
        color: #E2E8F0;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #162032;
        color: #F8FAFC;
        border: 1px solid #283953;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #243452;
        border-color: #3D557B;
        color: #FFFFFF;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%);
        border: 1px solid #38BDF8;
        color: #FFFFFF;
    }
    .stButton>button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369A1 0%, #075985 100%);
    }

    /* Dark overrides for light-theme Streamlit input widgets */
    .stTextArea textarea {
        background-color: #0B1220 !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E293B !important;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: #0B1220 !important;
    }
    .stSelectbox [data-baseweb="select"] > div,
    .stSelectbox label > div {
        background-color: #0B1220 !important;
        color: #E2E8F0 !important;
    }
    .stCode pre, [data-testid="stCode"] pre {
        background-color: #0B1220 !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E293B !important;
    }
    .stChatInput textarea {
        color: #E2E8F0 !important;
    }
    .stChatMessage {
        background-color: transparent !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"],
    .stChatMessage p,
    .stChatMessage li {
        color: #CBD5E1 !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# ENGINE & REPORT CACHE
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Contextual ML Routing Engine (training on real Arena battles)...")
def get_routing_engine() -> LLMSelectionRoutingEngine:
    engine = LLMSelectionRoutingEngine()
    engine.train(load_battles(), load_anchors())
    return engine


@st.cache_resource(show_spinner="Loading held-out accuracy report...")
def get_accuracy_report():
    path = PROJECT_ROOT / "outputs" / "accuracy" / "routing_accuracy_report.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RoutingAccuracyReport(**data)
    except Exception:
        return None


engine = get_routing_engine()
accuracy_report = get_accuracy_report()

# Per-model REAL battle win-rates (appearances basis) from the local dataset
REAL_WIN_STATS: dict = {}
if accuracy_report is not None:
    for m in accuracy_report.model_order:
        REAL_WIN_STATS[m] = {
            "win_rate": accuracy_report.overall_win_rates.get(m, 0.5),
            "games": accuracy_report.overall_game_counts.get(m, 0),
        }

# Candidate profiles: qualitative facts only. All quantitative claims are
# computed from real battle data above - nothing is hardcoded.
MODEL_PROFILES = {
    "gpt-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "tier": "Flagship Balanced Intelligence",
        "best_for": "Complex reasoning, full-stack programming, logic, and multi-step technical instructions.",
        "strengths": ["Strict instruction adherence", "Complex multi-step problem solving", "Structured output formatting"],
        "weakness": "Higher API cost per query compared to open-weight models.",
        "chat_url": "https://chatgpt.com"
    },
    "claude-3-5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "provider": "Anthropic",
        "tier": "Flagship Reasoning & Creative Writing",
        "best_for": "Nuanced creative writing, diplomatic correspondence, literature analysis, and deep code refactoring.",
        "strengths": ["High narrative eloquence", "Detailed chain-of-thought", "Low sycophancy"],
        "weakness": "Moderate response latency on lengthy generation tasks.",
        "chat_url": "https://claude.ai"
    },
    "deepseek-r1": {
        "name": "DeepSeek R1",
        "provider": "DeepSeek / Open-Weights",
        "tier": "Deep Mathematical & Logical Reasoning",
        "best_for": "Rigorous mathematics, theorem proofs, formal symbolic logic, and algorithmic derivations.",
        "strengths": ["Self-verification reasoning loop", "Competition-level mathematics", "Open architecture"],
        "weakness": "Extensive thinking tokens require higher inference compute time.",
        "chat_url": "https://chat.deepseek.com"
    },
    "qwen3-235b-a22b": {
        "name": "Qwen3 235B A22B",
        "provider": "Alibaba / Open-Weights",
        "tier": "Flagship Open-Weights Reasoning & Code",
        "best_for": "Reasoning-heavy code generation, math, and strong general chat at open-weights cost.",
        "strengths": ["Large MoE open-weights architecture", "Strong math & code battle record", "Low serving cost"],
        "weakness": "Higher latency than small specialist models on trivial prompts.",
        "chat_url": "https://chat.qwen.ai"
    },
    "gemini-1.5-pro": {
        "name": "Gemini 1.5 Pro",
        "provider": "Google",
        "tier": "Large Context & Search Synthesis",
        "best_for": "Massive context documents (up to 2M tokens), research synthesis, and live information retrieval.",
        "strengths": ["2-Million token context window", "Native Google Search grounding", "Audio/Video processing"],
        "weakness": "Can be overly verbose on concise programmatic prompts.",
        "chat_url": "https://gemini.google.com"
    },
    "llama-3.1-70b": {
        "name": "Llama 3.1 70B",
        "provider": "Meta / Self-Hosted",
        "tier": "High Throughput & Low Latency",
        "best_for": "High-volume pipelines, fast general QA, and cost-free private/self-hosted execution.",
        "strengths": ["Sub-second execution speeds", "Zero proprietary licensing fees", "Easily fine-tuned"],
        "weakness": "Occasionally trails flagship models on edge-case logic traps.",
        "chat_url": "https://groq.com"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "tier": "Cost-Effective Fast Utility",
        "best_for": "High-speed routine queries, basic summarization, email drafting, and budget-constrained apps.",
        "strengths": ["Fraction of the cost of flagship models", "Quick turnaround", "Solid general knowledge"],
        "weakness": "Less depth for graduate-level mathematics or large architectural refactors.",
        "chat_url": "https://chatgpt.com"
    }
}

# 10 Real-World Benchmark Task Categories
REAL_WORLD_TOPICS = {
    "Creative Fiction": "Write a short scifi story about a rogue satellite drifting past Saturn's rings that sends a faint signal home.",
    "Calculus & Math": "Solve the integral of x^2 * exp(x) from 0 to 3 using integration by parts, step by step with proof.",
    "Backend & Python": "Write a high-performance Python script to sort 10 million records with multiprocessing and memory profiling.",
    "Executive Email": "Draft a diplomatic corporate email requesting an update on a delayed vendor contract without sounding aggressive.",
    "Fast Facts": "What is the capital of France and what is its population in 2024?",
    "100-Page Document": "Summarize the key regulatory compliance takeaways from this lengthy 100-page financial PDF.",
    "Frontend & React": "Debug this React component that causes infinite re-renders when updating state from an async useEffect.",
    "Formal Logic Proof": "Prove that the square root of 2 is irrational using a proof by contradiction with complete deductive steps.",
    "Data Analysis & SQL": "Write an optimized PostgreSQL query with window functions to compute 7-day rolling revenue per customer segment.",
    "Language Translation": "Translate this legal disclaimer from English to Japanese using formal Keigo business register."
}

# 6 Strategic Optimization Presets
OPTIMIZATION_GOALS = {
    "Balanced & General Purpose (Recommended)": RoutingPolicyWeights(w_success=0.70, w_similarity=0.15, w_cost=0.08, w_latency=0.07),
    "High-Precision Flagship (Quality Benchmark)": RoutingPolicyWeights(w_success=0.75, w_similarity=0.15, w_cost=0.05, w_latency=0.05),
    "Deep Mathematical & Logical Derivation": RoutingPolicyWeights(w_success=0.75, w_similarity=0.15, w_cost=0.05, w_latency=0.05),
    "Specialized Code & Architecture Engineering": RoutingPolicyWeights(w_success=0.70, w_similarity=0.15, w_cost=0.05, w_latency=0.10),
    "Ultra-Fast Throughput (Sub-Second Latency)": RoutingPolicyWeights(w_success=0.20, w_similarity=0.10, w_cost=0.15, w_latency=0.55),
    "Cost-Constrained / Free Tier (Budget Optimization)": RoutingPolicyWeights(w_success=0.25, w_similarity=0.15, w_cost=0.45, w_latency=0.15),
}


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
_defaults = {
    "has_evaluated": False,
    "prompt_input": "",
    "session_history": [],
    "should_animate": False,
    "view": "home",
    "chat_history": [],
    "pending_action": None,
    "decision_meta": None,
    "feedback_submitted": False,
    "show_copy_prompt": False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# Consume pending chat actions BEFORE any widget is instantiated this run
if st.session_state.pending_action is not None:
    _action = st.session_state.pending_action
    st.session_state.pending_action = None
    if _action.type == "route_prompt":
        _p = (_action.payload.get("prompt") or "").strip()
        if _p:
            st.session_state.prompt_input = _p
            st.session_state.has_evaluated = True
            st.session_state.should_animate = False
            st.session_state.feedback_submitted = False
    elif _action.type == "switch_policy":
        _pol = _action.payload.get("policy")
        if _pol in OPTIMIZATION_GOALS:
            st.session_state.opt_mode = _pol
    elif _action.type == "open_feedback":
        if st.session_state.has_evaluated:
            st.session_state.view = "feedback"


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def profile_of(model_id: str) -> dict:
    return MODEL_PROFILES.get(model_id, {
        "name": model_id, "provider": "Provider", "tier": "General LLM",
        "best_for": "General completion", "strengths": ["Broad knowledge"],
        "weakness": "None", "chat_url": "#"
    })


def metric_hero(value: float, label: str, sub: str, is_good=None) -> str:
    cls = ""
    if is_good is not None:
        cls = "" if is_good else " miss"
    return (
        f"<div class='metric-hero'><div class='lbl'>{label}</div>"
        f"<div class='big{cls}'>{value:.2%}</div>"
        f"<div class='sub'>{sub}</div></div>"
    )


def trail_html(stages: list) -> str:
    rows = "".join(
        f"<div class='trail-stage'><span class='trail-check'>&#10003;</span>"
        f"<span class='trail-name'>{s['name']}</span>"
        f"<span class='trail-stats'>{s['stats']}</span></div>"
        for s in stages
    )
    return f"<div class='trail-box'>{rows}</div>"


def stage_box(s: dict) -> str:
    return f"""
    <div class='buffering-box'>
        <div class='buffering-header'>
            <span class='buffering-title'>{s['stage']} · BUFFERING INFERENCE PIPELINE</span>
            <span class='buffering-pct'>{s['pct']}%</span>
        </div>
        <div class='buffering-stage-name'>{s['name']}</div>
        <div class='buffering-status'>{s['status']}</div>
        <div class='buffering-stats'>{s['stats']}</div>
        <div class='buffering-bar-bg'><div class='buffering-bar-fill' style='width: {s['pct']}%;'></div></div>
    </div>
    """


# -----------------------------------------------------------------------------
# GLOBAL SIDEBAR & EXPERIENCE LEVEL (Simple / Standard / Expert)
# -----------------------------------------------------------------------------
exp_mode = ui.render_sidebar(engine, accuracy_report, profile_of)


# -----------------------------------------------------------------------------
# APPLICATION HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class='app-header'>
    <div>
        <div class='app-title'>Contextual LLM Routing Engine</div>
        <div class='app-subtitle'>
            Paste any question and get the best chatbot for it — chosen by a machine-learning router
            trained and honestly evaluated on real head-to-head AI battles.
            <em>Want the maths? Switch the Experience Level in the sidebar (Expert).</em>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# FEEDBACK PAGE (dedicated view)
# -----------------------------------------------------------------------------
if st.session_state.view == "feedback":
    st.markdown("## Ground Truth Feedback")
    meta = st.session_state.decision_meta
    if meta is None:
        st.info("No routing decision yet — go back, enter a prompt, and run the pipeline first.")
        if st.button("← Back to Router"):
            st.session_state.view = "home"
            st.rerun()
        st.stop()

    winner_prof = profile_of(meta["selected"])
    st.markdown(f"""
    <div class='report-card'>
        <div class='report-title'>Decision under review</div>
        <div style='font-size: 0.95rem; color: #CBD5E1; line-height: 1.7;'>
            <div><strong>Prompt:</strong> {meta['prompt'][:220]}{'…' if len(meta['prompt']) > 220 else ''}</div>
            <div><strong>Selected model:</strong> {winner_prof['name']} ({winner_prof['provider']})</div>
            <div><strong>Composite score:</strong> {meta['composite']:.4f} · <strong>Intent cluster:</strong> {meta['intent']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.feedback_submitted:
        st.success("Feedback recorded — thank you. It feeds the continuous retraining loop (ground-truth pairs).")
        with st.expander("Trigger Continuous Retraining (Incorporate Accumulated Feedback)"):
            st.write("Retrains the Bradley-Terry and pointwise Logistic Regression ensemble using newly accumulated ground-truth feedback pairs.")
            if st.button("Execute Model Retraining"):
                with st.spinner("Retraining Bradley-Terry Logistic Regression Ensemble..."):
                    retrain_res = engine.trigger_retraining()
                    st.success(
                        f"Retraining complete! Processed {retrain_res.feedback_samples_incorporated} feedback pairs. "
                        f"Total training samples: {retrain_res.total_comparisons_trained}."
                    )
                    st.json(retrain_res.updated_win_rates)
        fb_c1, fb_c2 = st.columns(2)
        with fb_c1:
            if st.button("← Back to Results", use_container_width=True):
                st.session_state.view = "home"
                st.session_state.feedback_submitted = False
                st.rerun()
        with fb_c2:
            if st.button("Start Over", use_container_width=True):
                st.session_state.prompt_input = ""
                st.session_state.has_evaluated = False
                st.session_state.should_animate = False
                st.session_state.chat_history = []
                st.session_state.decision_meta = None
                st.session_state.feedback_submitted = False
                st.session_state.view = "home"
                st.rerun()
        st.stop()

    fb_rating = st.radio("How good was this routing decision?", [1, 2, 3, 4, 5], horizontal=True,
                         help="1 = poor match, 5 = excellent match")
    fb_agree = st.selectbox(
        "Preferred model (if different):",
        ["— Agree with selection —"] + [m for m in engine.model_names if m != meta["selected"]],
        format_func=lambda m: profile_of(m)["name"],
    )
    fb_comment = st.text_area("Optional comment (what was right/wrong?):", height=90)

    fb_submit = st.button("Submit Feedback", type="primary")
    if fb_submit:
        is_agree = fb_agree == "— Agree with selection —"
        engine.repository.record_user_feedback(
            decision_id=meta["decision_id"],
            user_id="usr_terminal",
            is_positive=is_agree and fb_rating >= 4,
            rating=fb_rating,
            preferred_winner_model=None if is_agree else fb_agree,
            feedback_comment=fb_comment.strip() or None,
        )
        st.session_state.feedback_submitted = True
        st.rerun()

    if st.button("← Back to Results (skip)"):
        st.session_state.view = "home"
        st.rerun()
    st.stop()


# -----------------------------------------------------------------------------
# STEP 1: PROMPT INPUT & ONBOARDING TEMPLATES
# -----------------------------------------------------------------------------
st.markdown("##### Step 1: Enter Prompt or Choose Task Template")
st.caption("Click one of the example tasks below, or type/paste your own question — code, maths, emails, anything.")

row1_keys = list(REAL_WORLD_TOPICS.keys())[:5]
row2_keys = list(REAL_WORLD_TOPICS.keys())[5:]

cols_r1 = st.columns(len(row1_keys))
for i, key in enumerate(row1_keys):
    with cols_r1[i]:
        if st.button(key, key=f"chip_r1_{i}", use_container_width=True):
            st.session_state.prompt_input = REAL_WORLD_TOPICS[key]

cols_r2 = st.columns(len(row2_keys))
for i, key in enumerate(row2_keys):
    with cols_r2[i]:
        if st.button(key, key=f"chip_r2_{i}", use_container_width=True):
            st.session_state.prompt_input = REAL_WORLD_TOPICS[key]

current_prompt_val = st.text_area(
    label="Inbound Prompt String:",
    value=st.session_state.prompt_input,
    height=90,
    placeholder="Type or paste any programming question, math problem, creative draft, or inquiry...",
    label_visibility="collapsed"
)
st.session_state.prompt_input = current_prompt_val


# -----------------------------------------------------------------------------
# STEP 2: OPTIMIZATION PRESET
# -----------------------------------------------------------------------------
st.markdown("##### Step 2: Choose Optimization Priority")

opt_mode_selected = st.selectbox(
    label="Optimization Mode:",
    options=list(OPTIMIZATION_GOALS.keys()),
    index=0,
    key="opt_mode",
    label_visibility="collapsed"
)
st.caption(f"💡 {ui.preset_caption(opt_mode_selected)}")
active_policy_weights = OPTIMIZATION_GOALS[opt_mode_selected]

col_run1, col_run2, col_run3 = st.columns([1.6, 1.2, 4])
with col_run1:
    execute_button = st.button("Launch Analysis Pipeline", type="primary", use_container_width=True)
    if execute_button:
        if st.session_state.prompt_input.strip():
            st.session_state.has_evaluated = True
            st.session_state.should_animate = True
            st.session_state.feedback_submitted = False
        else:
            st.warning("Please type a prompt or click one of the task templates above.")

with col_run2:
    if st.button("Reset / Start Over", use_container_width=True):
        st.session_state.prompt_input = ""
        st.session_state.has_evaluated = False
        st.session_state.should_animate = False
        st.session_state.chat_history = []
        st.session_state.decision_meta = None
        st.session_state.feedback_submitted = False
        st.rerun()


# -----------------------------------------------------------------------------
# STEP 3: BUFFERING PIPELINE & SURPRISE WINNER REVEAL
# -----------------------------------------------------------------------------
decision = None
if st.session_state.has_evaluated and st.session_state.prompt_input.strip():
    prompt_str = st.session_state.prompt_input.strip()

    sanitized = engine.sanitizer.sanitize({"user_id": "usr_terminal", "prompt": prompt_str})
    structural = engine.featurizer.extract(sanitized.sanitized_prompt)
    anchor_match = engine.anchor_matrix.compute_similarity(sanitized.sanitized_prompt)
    enriched_ctx = engine.context_bridge.build_context(
        sanitized_prompt=sanitized.sanitized_prompt,
        user_id=sanitized.user_id,
        structural=structural,
        anchor_match=anchor_match
    )
    success_probs = engine.classifier_ensemble.predict_success_probabilities(
        prompt_feature_vector=enriched_ctx.feature_vector,
        prompt_text=sanitized.sanitized_prompt,
        intent_label=enriched_ctx.cluster_label
    )
    resource_forecasts = engine.resource_regressors.forecast(structural, enriched_ctx.cluster_id)
    ranked_candidates = engine.optimizer.evaluate_candidates(
        success_probs=success_probs,
        resource_forecasts=resource_forecasts,
        anchor_similarity=anchor_match.max_similarity,
        override_weights=active_policy_weights
    )
    decision = engine.reconciliation.reconcile_and_dispatch(
        request_id="req_" + str(int(time.time())),
        user_id="usr_terminal",
        sanitized_prompt=sanitized.sanitized_prompt,
        intent_cluster=enriched_ctx.cluster_label,
        ranked_candidates=ranked_candidates,
        pipeline_start_time=time.perf_counter()
    )

    top_cand = decision.ranked_candidates[0]

    # Stable decision identity for the feedback page (survives recomputation)
    if (st.session_state.decision_meta is None
            or st.session_state.decision_meta.get("prompt") != prompt_str):
        st.session_state.decision_meta = {
            "prompt": prompt_str,
            "decision_id": decision.decision_id,
            "selected": decision.selected_model,
            "composite": decision.top_composite_score,
            "intent": enriched_ctx.cluster_label,
        }

    if not any(h["prompt"] == prompt_str for h in st.session_state.session_history):
        st.session_state.session_history.insert(0, {
            "prompt": prompt_str,
            "winner": decision.selected_model,
            "intent": enriched_ctx.cluster_label.split("(")[0].strip(),
            "score": decision.top_composite_score
        })

    _bt_blend = getattr(engine.classifier_ensemble, "bt_blend", 0.85)
    anim_stages = [
        {
            "stage": "STAGE 01/06",
            "name": "In-Flight Structural Featurization Engine",
            "status": "Scanning syntactic morphology, character length, subword BPE token counts, and language orientation...",
            "stats": f"Extracted {structural.char_length} chars (~{structural.estimated_tokens} tokens) · Code: {structural.code_indicator:.2f} · Math: {structural.math_indicator:.2f} · Domain: {enriched_ctx.domain}",
            "pct": 16
        },
        {
            "stage": "STAGE 02/06",
            "name": "Latent Space Spatial Clustering (K-Means)",
            "status": "Projecting prompt coordinates into latent TF-IDF manifold and mapping nearest intent centroid...",
            "stats": f"Assigned: {enriched_ctx.cluster_label} · Latent Centroid #{enriched_ctx.cluster_id}",
            "pct": 34
        },
        {
            "stage": "STAGE 03/06",
            "name": "Enterprise Anchor Cosine Similarity Matrix",
            "status": "Computing high-dimensional cosine angle against enterprise verified baseline anchors: Sim(x, a)...",
            "stats": f"Max Proximity: {anchor_match.max_similarity:.4f} · Nearest Anchor: {anchor_match.nearest_anchor_id} ({anchor_match.nearest_category})",
            "pct": 52
        },
        {
            "stage": "STAGE 04/06",
            "name": "Bradley-Terry & Pointwise Logistic Regression Ensemble",
            "status": "Evaluating pairwise latent utilities U(m, x) = w·z + b and estimating calibrated P(Success)...",
            "stats": f"Ensemble: {_bt_blend:.0%} Bradley-Terry / {1 - _bt_blend:.0%} Pointwise · Calibrated Success Bounds: [0.05, 0.98]",
            "pct": 70
        },
        {
            "stage": "STAGE 05/06",
            "name": "Bifurcated Parallel Ridge Regression Network",
            "status": "Dual-headed parallel regressors forecasting computational token volume, inference cost ($), and response latency (ms)...",
            "stats": f"Output Tokens: ~{resource_forecasts[top_cand.model_id].estimated_output_tokens} · Latency Bounds: {min(c.raw_latency_ms for c in decision.ranked_candidates):.0f}ms - {max(c.raw_latency_ms for c in decision.ranked_candidates):.0f}ms",
            "pct": 88
        },
        {
            "stage": "STAGE 06/06",
            "name": "Bounded Linear Min-Max Composite Scoring Formula",
            "status": "Inverting resource costs to linear efficiency [0, 1] and computing weighted policy composite scores...",
            "stats": f"Formula: ({active_policy_weights.w_success:.2f}*P) + ({active_policy_weights.w_similarity:.2f}*Sim) + ({active_policy_weights.w_cost:.2f}*CostEff) + ({active_policy_weights.w_latency:.2f}*LatEff)",
            "pct": 100
        }
    ]

    if st.session_state.should_animate:
        buffer_ph = st.empty()
        for s in anim_stages:
            buffer_ph.markdown(stage_box(s), unsafe_allow_html=True)
            time.sleep(0.35)
        # Persist the completed trail instead of clearing it
        buffer_ph.markdown(trail_html(anim_stages), unsafe_allow_html=True)
        st.session_state.should_animate = False
    else:
        st.markdown(trail_html(anim_stages), unsafe_allow_html=True)

    st.markdown(f"""
    <div class='complete-banner'>
        [PIPELINE EXECUTED] All 6 inference stages completed successfully in {decision.routing_latency_ms:.2f} ms. Optimal chatbot determined strictly by highest composite score.
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STEP 4: THE SURPRISE WINNER REVEAL
    # -------------------------------------------------------------------------
    st.markdown("### The Algorithm Selected Your Optimal Chatbot")
    st.caption("Derived mathematically through multi-criteria weighted scoring across all 7 candidate models:")

    winner_id = decision.selected_model
    winner_prof = profile_of(winner_id)
    score_pct = int(decision.top_composite_score * 100)

    st.markdown(f"""
    <div class='reveal-box'>
        <div class='reveal-tag'>Rank #1 Recommendation · Composite Score: {decision.top_composite_score:.4f}</div>
        <div class='reveal-name'>{winner_prof['name']}</div>
        <div class='reveal-provider'>{winner_prof['provider']} · {winner_prof['tier']}</div>
        <div class='reveal-quote'>
            <strong>Algorithmic Justification:</strong> Based on the multi-criteria evaluation of your prompt, <strong>{winner_prof['name']}</strong> achieved the highest composite score of <strong>{decision.top_composite_score:.4f}</strong>. {winner_prof['best_for']}
        </div>
        <div>
            <span class='stat-chip'>Composite Match: <strong>{score_pct}%</strong></span>
            <span class='stat-chip'>P(Success): <strong>{top_cand.success_prob:.3f}</strong></span>
            <span class='stat-chip'>Cost Efficiency: <strong>{top_cand.cost_efficiency:.3f}</strong></span>
            <span class='stat-chip'>Latency Efficiency: <strong>{top_cand.latency_efficiency:.3f}</strong></span>
            <span class='stat-chip'>Est. Latency: <strong>{top_cand.raw_latency_ms:.1f} ms</strong></span>
            <span class='stat-chip'>Query Cost: <strong>${top_cand.raw_cost_usd:.6f}</strong></span>
            <span class='stat-chip'>Router Overhead: <strong>{decision.routing_latency_ms:.2f} ms</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ui.render_plain_summary(
        decision=decision,
        top_cand=top_cand,
        domain=enriched_ctx.domain,
        ranked=decision.ranked_candidates,
        profile_of=profile_of,
        win_stats=REAL_WIN_STATS,
    )

    action_c1, action_c2, action_c3 = st.columns([1.5, 1.5, 3])
    with action_c1:
        st.link_button(f"Open {winner_prof['name']}", winner_prof["chat_url"], type="primary", use_container_width=True)
    with action_c2:
        if st.button("Copy Prompt String", use_container_width=True):
            st.session_state.show_copy_prompt = not st.session_state.show_copy_prompt
    with action_c3:
        if st.button("Ask Another Question / Reset", use_container_width=True):
            st.session_state.prompt_input = ""
            st.session_state.has_evaluated = False
            st.session_state.should_animate = False
            st.session_state.chat_history = []
            st.session_state.decision_meta = None
            st.rerun()
    if st.session_state.show_copy_prompt:
        st.code(prompt_str, language=None)

    # Detailed method calculation accordion
    with st.expander("Inspect Complete 6-Stage Method Calculations (Mathematical Breakdown)"):
        st.markdown(f"""
        **Stage 1: Structural Featurization**
        - Character Length: `{structural.char_length}` | Subword Token Estimate: `~{structural.estimated_tokens}`
        - Code Syntactic Density: `{structural.code_indicator:.4f}` | Math Keyword Density: `{structural.math_indicator:.4f}`
        - Script: `{structural.language.upper()}` | Question Indicator: `{structural.question_indicator}` | Domain: `{enriched_ctx.domain}`

        **Stage 2: K-Means Latent Clustering**
        - Assigned Intent Segment: `{enriched_ctx.cluster_label}`
        - Centroid ID: `Cluster #{enriched_ctx.cluster_id}` | Dimension Block: `{len(enriched_ctx.feature_vector)} features`

        **Stage 3: Vector Anchor Proximity**
        - Nearest Production Baseline: `"{anchor_match.nearest_text[:60]}..."`
        - Category: `{anchor_match.nearest_category}` | Max Cosine Similarity: `{anchor_match.max_similarity:.4f}`

        **Stage 4: Bradley-Terry Logistic Inference Ensemble**
        - Predicted P(Success): `{top_cand.success_prob:.4f}` (Ensemble: {_bt_blend:.0%} Bradley-Terry / {1 - _bt_blend:.0%} Pointwise Prior)
        - Latent Utility: `U({winner_id}) = {top_cand.success_prob:.4f}`

        **Stage 5: Parallel Resource Regressors (Ridge)**
        - Output Tokens Forecast: `~{resource_forecasts[top_cand.model_id].estimated_output_tokens} tokens`
        - Expected Execution Latency: `{top_cand.raw_latency_ms:.1f} ms` | Estimated Query Cost: `${top_cand.raw_cost_usd:.6f} USD`

        **Stage 6: Multi-Criteria Composite Scoring**
        - Active Policy Weights: `w_success = {active_policy_weights.w_success:.2f}, w_sim = {active_policy_weights.w_similarity:.2f}, w_cost = {active_policy_weights.w_cost:.2f}, w_lat = {active_policy_weights.w_latency:.2f}`
        - Composite Formula: `Score = ({active_policy_weights.w_success:.2f} * {top_cand.success_prob:.3f}) + ({active_policy_weights.w_similarity:.2f} * {anchor_match.max_similarity:.3f}) + ({active_policy_weights.w_cost:.2f} * {top_cand.cost_efficiency:.3f}) + ({active_policy_weights.w_latency:.2f} * {top_cand.latency_efficiency:.3f}) = {decision.top_composite_score:.4f}`
        """)

    # -------------------------------------------------------------------------
    # STEP 5: HONEST DUAL-METRIC ACCURACY REPORT
    # -------------------------------------------------------------------------
    st.markdown("### Routing Accuracy — Honest Dual Metrics")
    st.caption(
        "Measured on held-out LMArena battles (prompts never seen during training). "
        "Metric A = router picks the empirical best model for the task category (real battle win-rates). "
        "Metric B = router ranks the actual battle winner above the loser."
    )
    if accuracy_report is None:
        st.info(
            "Accuracy report not found. Generate it offline with: "
            "`python scripts/build_accuracy_report.py` (one-time, ~30s), then reload."
        )
    else:
        m1c, m2c, m3c = st.columns([1, 1, 1])
        with m1c:
            st.markdown(metric_hero(
                accuracy_report.metric_a, "Metric A · Task-Category Agreement",
                f"target ≥ {accuracy_report.threshold:.0%} — "
                + ("MET" if accuracy_report.threshold_met else "not yet met (unrounded real number)"),
                is_good=accuracy_report.threshold_met,
            ), unsafe_allow_html=True)
        with m2c:
            st.markdown(metric_hero(
                accuracy_report.metric_b, "Metric B · Head-to-Head Agreement",
                f"router vs human winner on {accuracy_report.n_test} held-out battles",
            ), unsafe_allow_html=True)
        with m3c:
            st.markdown(metric_hero(
                accuracy_report.n_test / max(1, accuracy_report.n_test + accuracy_report.n_train),
                "Held-Out Evaluation Base",
                f"{accuracy_report.n_test} test / {accuracy_report.n_train} train battles · `{accuracy_report.dataset}`",
            ), unsafe_allow_html=True)

        ui.render_plain_accuracy(accuracy_report)

        eb_rows = []
        for d, best in accuracy_report.empirical_best.items():
            n_test_d = accuracy_report.domain_test_counts.get(d, 0)
            a_val = accuracy_report.per_domain_a.get(d)
            wr = accuracy_report.domain_win_rates.get(d, {}).get(best)
            eb_rows.append({
                "Category": d,
                "Empirical Best (train win-rate)": f"{profile_of(best)['name']} ({wr:.1%})" if wr else best,
                "Held-out prompts": n_test_d,
                "Metric A": f"{a_val:.1%}" if isinstance(a_val, (int, float)) else "—",
            })
        st.dataframe(pd.DataFrame(eb_rows), use_container_width=True, hide_index=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            st.pyplot(ch.chart_domain_accuracy(
                accuracy_report.per_domain_a, accuracy_report.domain_test_counts,
                threshold=accuracy_report.threshold,
            ), clear_figure=False)
        with ch2:
            st.pyplot(ch.chart_confusion_matrix(
                accuracy_report.confusion, accuracy_report.model_order,
            ), clear_figure=False)
        ch3, ch4 = st.columns(2)
        with ch3:
            st.pyplot(ch.chart_win_rate_heatmap(
                accuracy_report.domain_win_rates, accuracy_report.model_order,
            ), clear_figure=False)
        with ch4:
            st.pyplot(ch.chart_overall_win_rates({
                m: {"win_rate": accuracy_report.overall_win_rates[m],
                    "games": accuracy_report.overall_game_counts[m]}
                for m in accuracy_report.model_order
            }), clear_figure=False)

        st.caption(
            "Definitions: Metric A compares the router's pick to the model with the highest "
            "shrunk real battle win-rate in the prompt's task category (Beta(1,1) prior, train split). "
            "Metric B replays held-out battles: does the router's composite score rank the human "
            "winner above the loser? No metric is simulated or rounded up."
            + (
                f" Stability check — 5-fold grouped CV over prompts: "
                f"A = {accuracy_report.cv_metric_a_mean:.2%} ± {accuracy_report.cv_metric_a_std:.2%}, "
                f"B = {accuracy_report.cv_metric_b_mean:.2%} ± {accuracy_report.cv_metric_b_std:.2%}."
                if accuracy_report.cv_metric_a_mean is not None else ""
            )
        )

    # -------------------------------------------------------------------------
    # STEP 6: DECISION ANALYTICS CHARTS
    # -------------------------------------------------------------------------
    st.markdown("### Decision Analytics")
    st.caption("Model-level breakdown for this prompt under the selected preset:")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.pyplot(ch.chart_success_bars(success_probs, winner_id), clear_figure=False)
    with ac2:
        st.pyplot(ch.chart_composite_bars(decision.ranked_candidates, winner_id), clear_figure=False)
    st.pyplot(ch.chart_cost_latency_scatter(decision.ranked_candidates, winner_id), clear_figure=False)

    if exp_mode != "simple":
        st.pyplot(ch.chart_contribution_breakdown(
            decision.ranked_candidates, active_policy_weights, winner_id,
        ), clear_figure=False)

    # -------------------------------------------------------------------------
    # UNDER THE HOOD (Standard / Expert)
    # -------------------------------------------------------------------------
    if exp_mode != "simple":
        st.markdown("### 🧠 Under the Hood")
        st.caption("Live ML internals for this decision — switch to 👶 Simple in the sidebar to hide them.")

        st.markdown("#### Ensemble Internals (per model)")
        ui.render_ensemble_internals(
            success_probs, engine.classifier_ensemble, enriched_ctx.domain, winner_id,
        )

        ui.render_sandbox(
            optimizer=engine.optimizer,
            success_probs=success_probs,
            resource_forecasts=resource_forecasts,
            anchor_similarity=anchor_match.max_similarity,
            original_ranked=decision.ranked_candidates,
            original_weights=active_policy_weights,
            critical_threshold_default=engine.reconciliation.critical_success_threshold,
            profile_of=profile_of,
        )

        col_aff, col_feat = st.columns(2)
        with col_aff:
            st.markdown("#### Learned Domain Affinities")
            st.pyplot(ch.chart_affinity_heatmap(
                engine.classifier_ensemble.domain_affinities,
                list(engine.classifier_ensemble.model_names),
                highlight_domain=enriched_ctx.domain,
            ), clear_figure=False)
        with col_feat:
            ui.render_feature_inspector(structural, anchor_match, enriched_ctx)

        with st.expander("💲 Cost & latency model economics"):
            ui.render_model_specs(engine, resource_forecasts, profile_of)

        with st.expander("🛡️ Failsafe gate, audit trail & decision export"):
            ui.render_audit_panel(decision, engine.reconciliation)
            ui.render_decision_export(
                decision, sanitized.sanitized_prompt, enriched_ctx,
                structural, anchor_match, active_policy_weights,
            )

        if exp_mode == "expert":
            with st.expander("⚙️ Hyperparameters, provenance & telemetry"):
                ui.render_hyperparams(engine, accuracy_report)
                ui.render_telemetry(engine)

    # -------------------------------------------------------------------------
    # STEP 7: HORIZONTAL COMPARISON DECK (ALL 7 CANDIDATES)
    # -------------------------------------------------------------------------
    st.markdown("### Candidate Model Comparison Deck (Ranked by Score)")
    st.caption("Full comparative evaluation of all candidate engines under the selected criteria:")
    ui.render_roster(
        profile_of, REAL_WIN_STATS, engine.config.get("models", {}),
        compact=(exp_mode == "simple"),
    )

    cand_cols = st.columns(len(decision.ranked_candidates))
    for idx, cand in enumerate(decision.ranked_candidates):
        prof = profile_of(cand.model_id)
        is_sel = (cand.model_id == decision.selected_model)
        border_cls = "is-winner" if is_sel else ""
        rank_badge = f"Rank #{cand.rank}" + (" (Selected)" if is_sel else "")
        wr_info = REAL_WIN_STATS.get(cand.model_id)
        wr_line = (
            f"<div class='cand-line'>Real Arena WR: <strong>{wr_info['win_rate']:.1%}</strong> ({wr_info['games']} battles)</div>"
            if wr_info else ""
        )

        with cand_cols[idx]:
            st.markdown(f"""
            <div class='cand-card {border_cls}'>
                <div>
                    <div class='cand-rank'>{rank_badge}</div>
                    <div class='cand-name'>{prof['name']}</div>
                    <div class='cand-prov'>{prof['provider']}</div>
                    <div class='cand-score'>{cand.composite_score:.4f}</div>
                    <div class='cand-line'>P(Success): <strong>{cand.success_prob:.3f}</strong></div>
                    <div class='cand-line'>Est. Latency: <strong>{cand.raw_latency_ms:.0f}ms</strong></div>
                    <div class='cand-line'>Query Cost: <strong>${cand.raw_cost_usd:.5f}</strong></div>
                    <div class='cand-line'>Cost Eff: <strong>{cand.cost_efficiency:.2f}</strong></div>
                    <div class='cand-line'>Lat Eff: <strong>{cand.latency_efficiency:.2f}</strong></div>
                    {wr_line}
                </div>
                <div style='margin-top: 10px; font-size: 0.75rem; color: #64748B;'>
                    {prof['tier']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STEP 8: OFFLINE CHAT ASSISTANT
    # -------------------------------------------------------------------------
    st.markdown("### Ask the Router (Offline Assistant)")
    st.caption("Rule-based assistant — runs 100% locally (no API keys, no LLM calls).")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_text = st.chat_input("Ask why, compare models, switch preset (cheapest/fastest/quality), or paste a new prompt…")
    if user_text:
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        intent = chat.parse_intent(user_text)
        reply, action = chat.respond(
            intent,
            decision=decision,
            report=accuracy_report,
            profiles=MODEL_PROFILES,
            policy_name=opt_mode_selected,
        )
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        if action.type != "none":
            st.session_state.pending_action = action
        st.rerun()

    # Continue to dedicated feedback page
    fb_nav1, fb_nav2 = st.columns([1.4, 3])
    with fb_nav1:
        if st.button("Continue to Feedback →", type="primary", use_container_width=True):
            st.session_state.view = "feedback"
            st.rerun()


# -----------------------------------------------------------------------------
# SESSION PROMPT HISTORY
# -----------------------------------------------------------------------------
if st.session_state.session_history:
    st.markdown("---")
    st.markdown("##### Session History")
    st.caption("Re-inspect previous evaluations from this session:")

    hist_cols = st.columns(min(4, len(st.session_state.session_history)))
    for idx, hist in enumerate(st.session_state.session_history[:4]):
        with hist_cols[idx]:
            label = f"{hist['winner'].upper()} ({hist['intent']})"
            if st.button(label, key=f"hist_btn_{idx}", use_container_width=True):
                st.session_state.prompt_input = hist["prompt"]
                st.session_state.has_evaluated = True
                st.session_state.should_animate = False
                st.rerun()
