"""Offline rule-based chat assistant for the routing app.

No LLM/API calls — intents are matched with deterministic rules and answered
from local decision state, the accuracy report, and model profiles. The app
executes returned side-effect actions (re-route, switch policy, open feedback).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.routing.evaluation import RoutingAccuracyReport

# Preset names as defined by app_router.OPTIMIZATION_GOALS
POLICY_BALANCED = "Balanced & General Purpose (Recommended)"
POLICY_QUALITY = "High-Precision Flagship (Quality Benchmark)"
POLICY_MATH = "Deep Mathematical & Logical Derivation"
POLICY_CODE = "Specialized Code & Architecture Engineering"
POLICY_FAST = "Ultra-Fast Throughput (Sub-Second Latency)"
POLICY_BUDGET = "Cost-Constrained / Free Tier (Budget Optimization)"

_MODEL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "claude-3-5-sonnet": ("claude", "sonnet", "anthropic"),
    "deepseek-r1": ("deepseek", " r1", "deep-seek"),
    "gemini-1.5-pro": ("gemini", "google"),
    "gpt-4o": ("gpt-4o", "gpt4o", "chatgpt", "openai gpt", "4o"),
    "gpt-4o-mini": ("gpt-4o-mini", "gpt4o-mini", "4o mini", "mini"),
    "llama-3.1-70b": ("llama", "meta"),
    "qwen3-235b-a22b": ("qwen", "qwq", "alibaba"),
}

_GREETING = re.compile(r"^\s*(hi|hey|hello|yo|namaste|good (morning|evening))\b", re.I)


@dataclass
class ChatIntent:
    kind: str  # help | why | compare | switch_policy | metrics | feedback | status | new_prompt
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class ChatAction:
    """Side effect the app should perform after rendering the reply."""
    type: str  # route_prompt | switch_policy | open_feedback | none
    payload: Dict[str, str] = field(default_factory=dict)


def _detect_models(text: str) -> List[str]:
    found: List[str] = []
    lower = text.lower()
    for model_id, aliases in _MODEL_ALIASES.items():
        if any(alias in lower for alias in aliases):
            found.append(model_id)
    # "gpt-4o-mini" also matches "gpt-4o" alias — keep the longer match only
    if "gpt-4o-mini" in found and "gpt-4o" in found:
        if lower.count("mini") == 0 or lower.count("gpt-4o") == 1:
            found.remove("gpt-4o")
    return found


def _detect_policy(text: str) -> Optional[str]:
    lower = text.lower()
    if re.search(r"\b(balanc|default|general purpose)\w*", lower):
        return POLICY_BALANCED
    if re.search(r"\b(cheap|cheapest|budget|save money|cost[- ]?saving|free tier|lowest cost)\b", lower):
        return POLICY_BUDGET
    if re.search(r"\b(fast|fastest|speed|latency|throughput|sub-second|quick)\w*\b", lower):
        return POLICY_FAST
    if re.search(r"\b(math|mathematical|calculus|algebra|proof)\w*\b", lower):
        return POLICY_MATH
    if re.search(r"\b(code|coding|coding|architecture|engineering|refactor)\w*\b", lower):
        return POLICY_CODE
    if re.search(r"\b(quality|best|precis|flagship|accurate|strongest|top)\w*\b", lower):
        return POLICY_QUALITY
    return None


def parse_intent(text: str) -> ChatIntent:
    """Map a raw user message to a deterministic intent."""
    raw = (text or "").strip()
    lower = raw.lower()

    if not raw:
        return ChatIntent("help")
    if _GREETING.match(raw) or re.fullmatch(r"\W+", raw):
        return ChatIntent("help")
    if re.search(r"\b(what can you|help (me )?out|help|commands?|how to use|options)\b", lower):
        return ChatIntent("help")
    if re.search(r"\b(why|how did you|explain|justify|reason)\b", lower):
        return ChatIntent("why")
    if re.search(r"\b(accuracy|accurate|metric|scorecard|how good|benchmark|how well)\b", lower):
        return ChatIntent("metrics")
    if re.search(r"\b(feedback|rat(e|ing)|review|score this)\b", lower):
        return ChatIntent("feedback")
    if re.search(r"\b(compare|versus|vs\.?|better than|difference)\b", lower):
        models = _detect_models(raw)
        return ChatIntent("compare", {"models": ",".join(models)})
    if re.search(r"\b(switch|change|use|apply|set|prefer|want|route)\b", lower) and _detect_policy(raw):
        return ChatIntent("switch_policy", {"policy": _detect_policy(raw) or ""})
    if re.search(r"\b(recommend|current|current pick|now using|status|summary)\b", lower):
        return ChatIntent("status")
    if _detect_policy(raw) and re.search(r"\b(cheaper|faster|quality|budget|policy|preset)\b", lower):
        return ChatIntent("switch_policy", {"policy": _detect_policy(raw) or ""})
    if len(raw) >= 8:
        return ChatIntent("new_prompt", {"prompt": raw})
    return ChatIntent("help")


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "n/a"


def respond(
    intent: ChatIntent,
    decision=None,
    report: Optional[RoutingAccuracyReport] = None,
    profiles: Optional[Dict[str, dict]] = None,
    policy_name: str = POLICY_BALANCED,
) -> Tuple[str, ChatAction]:
    """Produce a markdown reply plus the app action to execute (if any)."""
    profiles = profiles or {}

    if intent.kind == "help":
        return (
            "**Offline routing assistant** — I run entirely on local rules (no API calls). Try:\n"
            "- `Why this model?` — rationale for the current pick\n"
            "- `Compare Claude vs GPT-4o` — side-by-side\n"
            "- `Switch to fastest / cheapest / quality / code / math`\n"
            "- `How accurate is the router?` — real dual-metric numbers\n"
            "- `Open feedback` — rate this routing decision\n"
            "- or just paste a new prompt and I'll route it",
            ChatAction("none"),
        )

    if intent.kind == "why":
        if decision is None:
            return ("No decision yet — enter a prompt first.", ChatAction("none"))
        ranked = decision.ranked_candidates
        top = ranked[0] if ranked else None
        runner = ranked[1] if len(ranked) > 1 else None
        msg = [f"**{profiles.get(top.model_id, {}).get('name', top.model_id)}** "
               f"was selected under **{policy_name}**."]
        if top and runner:
            msg.append(
                f"Its composite score `{top.composite_score:.3f}` beats "
                f"**{profiles.get(runner.model_id, {}).get('name', runner.model_id)}** "
                f"`{runner.composite_score:.3f}` "
                f"(success {top.success_prob:.2f} vs {runner.success_prob:.2f}, "
                f"cost efficiency {top.cost_efficiency:.2f} vs {runner.cost_efficiency:.2f}, "
                f"latency efficiency {top.latency_efficiency:.2f} vs {runner.latency_efficiency:.2f})."
            )
        prof = profiles.get(top.model_id, {}) if top else {}
        if prof.get("strengths"):
            strengths = " · ".join(str(s) for s in prof.get("strengths", []))
            if strengths:
                msg.append(f"Profile strengths: {strengths}")
        if decision.is_fallback_triggered:
            msg.append(f"_Note:_ failsafe triggered — {decision.fallback_reason}")
        return ("\n\n".join(msg), ChatAction("none"))

    if intent.kind == "compare":
        ids = [m for m in intent.params.get("models", "").split(",") if m]
        if len(ids) < 2 and decision is not None and ids:
            other = next(
                (r.model_id for r in decision.ranked_candidates if r.model_id != ids[0]),
                None,
            )
            if other and other not in ids:
                ids.append(other)
        if len(ids) < 2:
            return (
                "Tell me two models, e.g. `Compare Claude vs GPT-4o`.",
                ChatAction("none"),
            )
        lines = []
        for m in ids:
            prof = profiles.get(m, {})
            extra = ""
            if decision is not None:
                row = next((r for r in decision.ranked_candidates if r.model_id == m), None)
                if row:
                    extra = (f" — composite `{row.composite_score:.3f}`, "
                             f"success `{row.success_prob:.2f}`")
            lines.append(
                f"- **{prof.get('name', m)}** ({prof.get('provider', '?')}, "
                f"{prof.get('tier', '?')}){extra}"
            )
        return ("\n".join(lines), ChatAction("none"))

    if intent.kind == "switch_policy":
        policy = intent.params.get("policy") or POLICY_BALANCED
        return (
            f"Switching preset to **{policy}** and re-routing your prompt.",
            ChatAction("switch_policy", {"policy": policy}),
        )

    if intent.kind == "metrics":
        if report is None:
            return ("Accuracy report not loaded yet.", ChatAction("none"))
        return (
            "**Honest dual-metric accuracy** (held-out LMArena battles):\n"
            f"- **Metric A — Task-Category Agreement: {_fmt_pct(report.metric_a)}** "
            f"(target {_fmt_pct(report.threshold)}) — router picks the empirical best model "
            "for the task category, per real battle win-rates.\n"
            f"- **Metric B — Head-to-Head Preference Agreement: {_fmt_pct(report.metric_b)}** — "
            "on real held-out battles, the router ranks the actual winner above the loser.\n"
            f"- Evaluated on {report.n_test} held-out battles "
            f"(train {report.n_train}), dataset `{report.dataset}`. "
            "Neither number is rounded up or simulated.",
            ChatAction("none"),
        )

    if intent.kind == "feedback":
        return ("Opening the feedback page — rate the current decision.", ChatAction("open_feedback"))

    if intent.kind == "status":
        if decision is None:
            return ("No decision yet — enter a prompt first.", ChatAction("none"))
        top = decision.ranked_candidates[0] if decision.ranked_candidates else None
        if not top:
            return ("Last decision fell back to the failsafe default.", ChatAction("none"))
        prof = profiles.get(top.model_id, {})
        return (
            f"Current pick: **{prof.get('name', top.model_id)}** — "
            f"composite `{top.composite_score:.3f}`, success `{top.success_prob:.2f}`, "
            f"policy **{policy_name}**, cluster `{decision.intent_cluster}`.",
            ChatAction("none"),
        )

    if intent.kind == "new_prompt":
        prompt = intent.params.get("prompt", "")
        return (
            f"Routing your new prompt: _{prompt[:80]}{'…' if len(prompt) > 80 else ''}_",
            ChatAction("route_prompt", {"prompt": prompt}),
        )

    return ("Sorry — I didn't catch that. Type `help` for examples.", ChatAction("none"))
