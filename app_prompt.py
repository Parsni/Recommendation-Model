"""Prompt -> AI Model Recommender (domain app built on the generic engine).

Launch:  streamlit run app_prompt.py   (appearance: .streamlit/config.toml)
Concept: prompt = "user", AI model = "item", your 1-5 trial rating =
interaction_value. For a brand-new prompt the matcher finds your most
similar logged prompts and blends their recommendations.

Interface: app-style front page (hero search + browse carousel), then a
comparison card grid with one reason chip per card, icon feedback taps and a
collapsible filter strip. Explanation lives in tooltips and one collapsed
"How these rankings work" panel rather than on the canvas.

The bundled sample ratings are deterministic placeholders, NOT benchmark
results - collect your own trials with the taps or the Log-a-trial form
(data/prompt_rec/README_DATA_COLLECTION.md) for real DA findings.

Workflow: 1 Find a model -> 2 My trial log -> 3 Report & checks.
"""
from __future__ import annotations

import hashlib
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.domain import trial_log
from src.domain.prompt_matcher import PromptMatcher
from src.evaluation.compare import run_chronological_comparison
from src.reporting.exporter import recommendations_to_df, save_table
from src.statistics import descriptive as desc
from src.utils.helpers import PROJECT_ROOT

SAMPLE_DIR = PROJECT_ROOT / "data" / "prompt_rec"
MY_DIR = SAMPLE_DIR / "uploaded"
STEPS = ["🔍  Find a model", "📝  My trial log", "📊  Report & checks"]

# One-tap feedback has to land on the same 1-5 scale the typed form uses, so the
# mapping is stated in the UI and in docs rather than left implicit.
TAP_RATING = {"good": 4, "poor": 2}

DEMO_CHIP = "Demo data · placeholder ratings, not real benchmarks"

EXAMPLE_PROMPTS = [
    "write a python script to clean survey responses in a csv and summarize by region",
    "create an isometric illustration of a cozy bookshop for a website banner",
    "translate this marketing email from English to Japanese, keep it formal",
    "explain a t-test to a non-technical manager in five bullets",
]

CATEGORY_LABELS = {
    "coding": "coding", "creative_writing": "creative writing",
    "image_generation": "image generation", "reasoning_math": "reasoning & math",
    "translation": "translation", "data_analysis": "data analysis",
    "research_summary": "research summary",
}


@st.cache_resource(show_spinner="Building recommender from trial data...")
def get_matcher(trials, prompts, models):
    return PromptMatcher(trials, prompts, models)


# --------------------------------------------------------------------- helpers
def model_label(eng, model_id: str) -> str:
    return str(eng.data.item_metadata(model_id).get("provider_label", model_id))


def model_meta(eng, model_id: str) -> dict:
    return eng.data.item_metadata(model_id)


def chip(text: str, tone: str = "") -> None:
    st.markdown(f"<span class='chip {tone}'>{html.escape(text)}</span>",
                unsafe_allow_html=True)


def chips(items: list[tuple[str, str]]) -> None:
    if not items:
        return
    st.markdown("<div class='tagrow'>" + "".join(
        f"<span class='chip {tone}'>{html.escape(text)}</span>"
        for text, tone in items) + "</div>", unsafe_allow_html=True)


def section(title: str, note: str = "") -> None:
    st.markdown(f"<div class='section'><span class='sectionTitle'>{title}</span>"
                f"<span class='sectionNote'>{html.escape(note)}</span></div>",
                unsafe_allow_html=True)


def line(text: str) -> None:
    """One quiet status line on the canvas - no coloured callout box."""
    st.markdown(f"<div class='feedline'>{html.escape(text)}</div>",
                unsafe_allow_html=True)


def fineprint(text: str) -> None:
    st.markdown(f"<div class='muted'>{html.escape(text)}</div>",
                unsafe_allow_html=True)


def match_strength(best_similarity: float) -> tuple[str, str]:
    if best_similarity >= 0.45:
        return "Strong match", "green"
    if best_similarity >= 0.25:
        return "Fair match", "amber"
    if best_similarity > 0:
        return "Weak match", "red"
    return "No wording match", "grey"


def logged_rows() -> pd.DataFrame:
    """The user's own in-app trial log, if they have started one."""
    if not (MY_DIR / trial_log.TRIALS_FILE).exists():
        return pd.DataFrame()
    return trial_log.logged_trials(MY_DIR)


def signal_for(item_id: str, neighbours: list[dict], rated: dict, stats: dict,
               logged: bool = False) -> str:
    """The one 'because' chip on a card, from trial rows that actually exist."""
    for nb in neighbours or []:
        value = rated.get((str(nb["prompt_id"]), str(item_id)))
        if value is not None:
            if logged:
                return f"You rated it {value}/5 on this prompt"
            return (f"You rated it {value}/5 on \"{nb['prompt_text'][:52]}\" "
                    f"· wording match {nb['similarity']:.2f}")
    row = stats.get(str(item_id))
    if row:
        return f"In your log: {row[0]} trial{'s' if row[0] != 1 else ''}, mean {row[1]:.2f}/5"
    return "No trial row for this model - shown by catalogue metadata only"


def bars(df: pd.DataFrame, eng, neighbours: list[dict] | None = None,
         logged: bool = False) -> list[dict]:
    """Convert a recommendations frame into card payloads."""
    rated, stats = _rated_pairs(eng), _model_stats(eng)
    rows = []
    for _, r in df.iterrows():
        meta = model_meta(eng, r["item_id"])
        rows.append({
            "rank": int(r["rank"]), "item_id": str(r["item_id"]),
            "name": model_label(eng, r["item_id"]),
            "modality": str(meta.get("modality", "")),
            "pricing": str(meta.get("pricing", "")),
            "best_for": str(meta.get("best_for", "")),
            "score": float(r["score"]),
            "score_kind": r["score_kind"],
            "sources": str(r["sources"]).split(";"),
            "explanation": r["explanation"],
            "limitations": r["limitations"],
            "signal": signal_for(str(r["item_id"]), neighbours, rated, stats, logged),
        })
    ceiling = max((x["score"] for x in rows), default=0.0) or 1.0
    for x in rows:
        x["share"] = min(x["score"] / ceiling, 1.0)
    return rows


def _rated_pairs(eng) -> dict:
    inter = eng.data.interactions
    return {(str(u), str(i)): int(v) for u, i, v in zip(inter["user_id"],
                                                        inter["item_id"],
                                                        inter["interaction_value"])}


def _model_stats(eng) -> dict:
    agg = eng.data.interactions.groupby("item_id")["interaction_value"].agg(
        ["count", "mean"])
    return {str(i): (int(r["count"]), float(r["mean"])) for i, r in agg.iterrows()}


def prompt_category(matcher, prompt_text: str, fallback: str) -> str:
    """Category of an exact logged prompt match, else the fallback label."""
    for pid, text, cat in zip(matcher.prompt_ids, matcher.prompt_texts,
                              matcher.categories):
        if str(text).strip() == str(prompt_text).strip():
            return str(cat or fallback)
    return fallback


def seen_models(text: str, matcher, eng) -> set:
    """Models already rated for this exact prompt - active dataset plus the
    in-app log, so a tap takes effect on the next screen."""
    wanted = str(text).strip()
    pids = [str(u) for u, t in zip(matcher.prompt_ids, matcher.prompt_texts)
            if str(t).strip() == wanted]
    inter = eng.data.interactions
    out = set(inter[inter["user_id"].astype(str).isin(pids)]["item_id"].astype(str))
    mine = logged_rows()
    if not mine.empty:
        out |= set(mine[mine["prompt_text"].astype(str).str.strip() == wanted]
                   ["model_id"].astype(str))
    return out


def candidate_pool(eng, scope: str, hidden: list, seen: set) -> list | None:
    all_ids = [str(i) for i in eng.data.items["item_id"]]
    pool = all_ids
    if scope != "Any":
        pool = [str(r["item_id"]) for _, r in eng.data.items.iterrows()
                if str(r.get("modality")) == scope]
    pool = [i for i in pool if i not in set(hidden) | set(seen)]
    return None if pool == all_ids else pool


def tap(ctx: dict, card: dict, kind: str) -> None:
    """One-tap affinity feedback -> a real row in the user's own trial log."""
    value = TAP_RATING[kind]
    try:
        trial_log.append_trial(MY_DIR, ctx["text"], ctx["category"], card["item_id"],
                               value)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["logged_note"] = (
        f"Logged {value}/5 for {card['name']}. Row added to "
        "data/prompt_rec/uploaded/my_trials.csv; your source CSVs were not touched.")
    st.toast(f"{card['name']} rated {value}/5", icon="✅")


# ------------------------------------------------------------------ components
def feedback_row(col, card: dict, ctx: dict) -> None:
    key = f"{ctx['tag']}{ctx['key']}{card['item_id']}"
    b1, b2, b3 = col.columns([1, 1, 3], gap="small")
    if b1.button("👍", key=f"up{key}",
                 help=f"Good fit - logs this model at {TAP_RATING['good']}/5 on this "
                      "prompt as a real trial row."):
        tap(ctx, card, "good")
    if b2.button("👎", key=f"dn{key}",
                 help=f"Poor fit - logs this model at {TAP_RATING['poor']}/5 on this "
                      "prompt as a real trial row."):
        tap(ctx, card, "poor")
    if b3.button("Not now", key=f"hd{key}",
                 help="Hide this model on this screen for the rest of the session. "
                      "Writes no rating; Undo restores it."):
        if card["item_id"] not in st.session_state["hidden_models"]:
            st.session_state["hidden_models"].append(card["item_id"])
        st.rerun()


def render_card(col, card: dict, ctx: dict | None = None) -> None:
    with col, st.container(border=True):
        st.markdown(
            f"<div class='rankrow'><span class='rankbadge'>{card['rank']}</span>"
            f"<span class='rankname'>{html.escape(card['name'])}</span></div>",
            unsafe_allow_html=True)
        chips([(t, "") for t in (card["modality"], card["pricing"]) if t and t != "nan"])
        st.progress(card["share"], text=f"{card['score']:.2f} ranking score")
        chip(card["signal"], "blue")
        fineprint("signals · " + " · ".join(card["sources"]))
        with st.expander("Why this model", expanded=False):
            st.markdown(f"**Suits:** {html.escape(card['best_for'])}")
            st.write(card["explanation"])
            st.caption("The ranking score is normalized inside the results shown - it is "
                       "not a probability of a good output.")
            st.caption(card["limitations"])
        if ctx and ctx["text"].strip():
            feedback_row(col, card, ctx)


def render_ranking(rows: list[dict], columns: int = 2, ctx: dict | None = None) -> None:
    for chunk_start in range(0, len(rows), columns):
        cols = st.columns(columns, gap="medium")
        for col, card in zip(cols, rows[chunk_start:chunk_start + columns]):
            render_card(col, card, ctx)


def render_carousel(rows: list[dict], ctx: dict | None = None) -> None:
    """Horizontal browse panel: one card at a time, arrow navigation, no modal."""
    if not rows:
        fineprint("No ranked model for this prompt yet.")
        return
    n = len(rows)
    slot = int(st.session_state.get("car_slot", 0)) % n
    nav = st.columns([1, 7, 1], gap="small")
    if nav[0].button("◀", key="car_prev", help="Previous model"):
        st.session_state["car_slot"] = (slot - 1) % n
        st.rerun()
    render_card(nav[1], rows[slot], ctx)
    if nav[2].button("▶", key="car_next", help="Next model"):
        st.session_state["car_slot"] = (slot + 1) % n
        st.rerun()
    with nav[1]:
        dots = "".join(f"<span class='dot{' on' if j == slot else ''}'></span>"
                       for j in range(n))
        st.markdown(f"<div class='dotrow'>{dots}<span class='slotno'>{slot + 1} of {n}"
                    f"</span></div>", unsafe_allow_html=True)


def ctx_for(matcher, eng, prompt_text: str, tag: str, category: str | None = None) -> dict:
    return {"text": prompt_text, "tag": tag,
            "category": category or prompt_category(matcher, prompt_text, "unspecified"),
            "key": hashlib.md5(str(prompt_text).strip().encode()).hexdigest()[:8]}


def render_discovery(matcher, eng, sample: bool) -> None:
    """Front-page browse panel fed by the newest logged trial."""
    inter = eng.data.interactions
    section("Recommended for you", "swipe ◀ ▶ · tap 👍 👎 on any card")
    if inter.empty:
        fineprint("No usable trial rows yet - rate an output under My trial log, or "
                  "search above.")
        return
    latest = inter.sort_values("timestamp", kind="stable").iloc[-1]
    pid = str(latest["user_id"])
    ptext = matcher.prompt_texts[matcher.prompt_ids.index(pid)]
    st.markdown(f"<div class='feedline'>Because your most recent trial ran "
                f"<b>{html.escape(pid)}</b> · {html.escape(ptext[:88])}</div>",
                unsafe_allow_html=True)
    if st.button("Search this prompt", key="car_seed",
                 help="Load this logged prompt into the search box"):
        st.session_state["pending_prompt"] = ptext
        st.rerun()
    rows = bars(recommendations_to_df(
        eng.recommend(pid, top_k=4)["recommendations"]), eng,
        [{"prompt_id": pid, "similarity": 1.0, "prompt_text": ptext}], logged=True)
    if sample:
        rows = [dict(r, signal=r["signal"] + " · demo rating") for r in rows]
    render_carousel(rows, ctx=ctx_for(matcher, eng, ptext, "car"))


def switch_to_my_log(label: str, key: str, paths: dict) -> None:
    if st.button(label, type="primary", key=key):
        st.session_state.custom_paths = {
            "trials.csv": MY_DIR / trial_log.TRIALS_FILE,
            "prompts.csv": MY_DIR / trial_log.PROMPTS_FILE,
            "models.csv": paths.get("models.csv", SAMPLE_DIR / "models.csv")}
        st.cache_resource.clear()
        st.rerun()


def how_it_works(sample: bool, eligible: list[str]) -> None:
    with st.expander("How these rankings work"):
        st.write(
            "- Each card is produced by algorithms that ran on **your** logged trials; "
            "the small grey line names them.\n"
            "- The blue chip is the reason: a named trial row, or that model's own "
            "history. Where the data supports no reason, the card says so instead of "
            "inventing one.\n"
            "- **Model type** is a hard catalogue filter - it decides who may appear, "
            "not how they score. **Hide models I already rated** and **Not now** only "
            "change what this screen displays.\n"
            "- 👍 / 👎 write real trial rows (4/5 and 2/5 on the 1-5 rubric). No number "
            "in this app is a probability.")
        if sample:
            st.write(f"- {DEMO_CHIP}: the bundled ratings are generated placeholders, so "
                     "those reasons describe the mechanics, not any model's quality. "
                     "Rate your own outputs to replace them.")
        fineprint("Algorithms active on this dataset: " + ", ".join(eligible))


# ---------------------------------------------------------------- page config
st.set_page_config(page_title="Prompt to AI Model", page_icon="◆", layout="wide")
st.markdown("""<style>
.block-container {padding-top:1.3rem;padding-bottom:2.6rem;max-width:1120px;}
footer,#MainMenu {visibility:hidden;}
.hero h1 {font-size:1.8rem;font-weight:800;letter-spacing:-.03em;margin:0 0 .1rem;}
.hero p {color:#667085;font-size:.9rem;margin:0;}
.brand {font-size:1.02rem;font-weight:800;letter-spacing:-.02em;}
.section {display:flex;align-items:baseline;gap:.55rem;margin:1.4rem 0 .5rem;}
.sectionTitle {font-size:1.04rem;font-weight:750;letter-spacing:-.02em;}
.sectionNote {font-size:.76rem;color:#8a94a6;}
.feedline {font-size:.85rem;color:#475467;background:#f5f7fb;border:1px solid #eceff4;
  border-radius:12px;padding:.5rem .75rem;margin:.25rem 0 .55rem;}
.muted {font-size:.74rem;color:#8a94a6;margin:-.15rem 0 .35rem;}
.chip {display:inline-flex;align-items:center;background:#eef1f6;color:#3f4a5a;
  border:1px solid #e4e8ef;border-radius:999px;padding:.16rem .6rem;font-size:.73rem;
  font-weight:600;line-height:1.5;}
.chip.blue {background:#eaf1ff;border-color:#d5e3ff;color:#1d5bc7;}
.chip.green {background:#e7f6ed;border-color:#cfeadb;color:#1a6b3d;}
.chip.amber {background:#fff4e3;border-color:#ffe2bd;color:#8a5300;}
.chip.red {background:#fdeceb;border-color:#f8d7d5;color:#a32d28;}
.chip.grey {background:#f2f4f7;border-color:#e4e8ef;color:#667085;}
.tagrow {display:flex;gap:.3rem;flex-wrap:wrap;margin:-.1rem 0 .45rem;}
.rankrow {display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;}
.rankbadge {display:inline-flex;align-items:center;justify-content:center;
  width:1.6rem;height:1.6rem;border-radius:12px;color:#fff;font-size:.82rem;
  font-weight:800;background:linear-gradient(135deg,#2f6fed,#6da3ff);}
.rankname {font-size:1.02rem;font-weight:750;letter-spacing:-.015em;}
.dotrow {display:flex;align-items:center;gap:.32rem;margin-top:-.45rem;}
.dot {width:.4rem;height:.4rem;border-radius:50%;background:#d0d5dd;}
.dot.on {background:#2f6fed;}
.slotno {font-size:.75rem;color:#8a94a6;margin-left:.5rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:18px;}
div[data-testid="stMetric"] {background:#f7f8fa;border:1px solid #e6e9ef;
  border-radius:14px;padding:12px 14px;}
div[data-testid="stExpander"] details {border-radius:12px;border-color:#e4e8ef;}
.stButton > button {border-radius:12px;}
h1,h2,h3 {letter-spacing:-.02em;}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------- sidebar
if "pending_prompt" in st.session_state:
    st.session_state["prompt_seed"] = st.session_state.pop("pending_prompt")

paths = st.session_state.get("custom_paths", {})
matcher = get_matcher(paths.get("trials.csv", SAMPLE_DIR / "trials.csv"),
                      paths.get("prompts.csv", SAMPLE_DIR / "prompts.csv"),
                      paths.get("models.csv", SAMPLE_DIR / "models.csv"))
eng = matcher.engine
hidden = st.session_state.setdefault("hidden_models", [])
sample = not paths

with st.sidebar:
    st.markdown("<div class='brand'>Prompt → AI</div>", unsafe_allow_html=True)
    fineprint("which tool runs it best")
    st.divider()
    idx = st.radio("Workflow", STEPS, label_visibility="collapsed", key="step")
    st.divider()
    if hidden:
        chip(f"{len(hidden)} hidden this session", "grey")
    if sample:
        chip(DEMO_CHIP, "amber")
    fineprint(f"{len(eng.data.interactions)} trials · {eng.data.n_users} prompts · "
              f"{eng.data.n_items} models · source: "
              f"{'demo' if sample else 'your uploads'}")

# --------------------------------------------------------------- step 1: ask
if idx == STEPS[0]:
    st.markdown("<div class='hero'><h1>Which AI should run this prompt?</h1>"
                "<p>Ranked from trials you rated on prompts with similar wording.</p>"
                "</div>", unsafe_allow_html=True)

    with st.container(border=True):
        text = st.text_area(
            "Your prompt", height=96, key="prompt_seed", label_visibility="collapsed",
            placeholder="Paste a prompt - e.g. write a python script to clean survey "
                        "responses and summarize by region")
        row = st.columns([1, 1])
        with row[0]:
            run = st.button("Find my model", type="primary", width="stretch")
        with row[1]:
            if st.button("Refine ▾", width="stretch", key="open_filters",
                         help="Model type, number of results, hide what you already "
                              "rated, undo hides"):
                st.session_state["show_filters"] = not st.session_state.get(
                    "show_filters", False)
        picked = st.pills("Try an example", EXAMPLE_PROMPTS, selection_mode="single",
                          label_visibility="collapsed")
        if picked:
            st.session_state["prompt_seed"] = picked
    if run:
        st.session_state["asked"] = text.strip()
    asked = st.session_state.get("asked", "")

    scope, top_k, hide_seen = "Any", 4, False
    modalities = sorted({str(v) for v in eng.data.items.get("modality", [])}
                        - {"nan"}) if "modality" in eng.data.items.columns else []
    if st.session_state.get("show_filters"):
        with st.container(border=True):
            filters = st.columns([2, 1, 3, 1, 1])
            with filters[0]:
                scope = st.selectbox("Model type", ["Any", *modalities],
                                     help="Hard catalogue filter: restricts which models "
                                          "may appear. It does not re-score them.")
            with filters[1]:
                top_k = st.selectbox("Results", [3, 4, 5, 6, 8], index=1,
                                     help="How many models to rank")
            with filters[2]:
                hide_seen = st.checkbox("Hide models I already rated for this prompt",
                                        help="Checked against your logged trials for "
                                             "this exact prompt text.")
            with filters[3]:
                st.write("")
                if hidden and st.button(f"Undo {len(hidden)}", key="unhide"):
                    st.session_state["hidden_models"] = []
                    st.rerun()
            with filters[4]:
                st.write("")
                if asked and st.button("Browse", key="back_to_carousel",
                                       help="Leave this answer and browse again"):
                    st.session_state.pop("asked", None)
                    st.rerun()

    if asked:
        seen = seen_models(asked, matcher, eng) if hide_seen else set()
        candidates = candidate_pool(eng, scope, hidden, seen)
        res = matcher.recommend_for_prompt(asked, top_k=int(top_k),
                                          candidate_ids=candidates)
        neighbours = res["matched_prompts"]
        best_sim = neighbours[0]["similarity"] if neighbours else 0.0
        strength, tone = match_strength(best_sim)
        category = (neighbours[0]["task_category"] if neighbours and best_sim >= 0.45
                    else "unspecified")
        rows = bars(recommendations_to_df(res["recommendations"]), eng, neighbours)

        head = st.columns([3, 1])
        with head[0]:
            section("Best match for your prompt",
                    f"filter {scope} · {len(rows)} of {len(eng.data.items)} models")
            line(asked[:150])
        with head[1]:
            st.write("")
            chip(f"{strength} · sim {best_sim:.2f}", tone)
            if hidden or seen:
                chip(f"{len(set(hidden) | seen)} held back", "grey")
        for note in res["notes"]:
            line(note)
        if rows:
            render_ranking(rows, columns=2,
                           ctx=ctx_for(matcher, eng, asked, "grid", category))
        else:
            line("Nothing left to show - every candidate for this selection is hidden or "
                 "already rated. Undo, untick the filter, or widen the model type.")

        if note := st.session_state.pop("logged_note", None):
            line(note)
        mine = logged_rows()
        if not mine.empty:
            loop = st.columns([1, 2])
            with loop[0]:
                switch_to_my_log("Rank from my logged trials", "switch_answer", paths)
            with loop[1]:
                fineprint(f"your log: {len(mine)} trials over "
                          f"{mine['prompt_id'].nunique()} prompts - switching rebuilds "
                          f"eligibility, ranking and evaluation on those ratings")

        with st.expander("Evidence - logged prompts behind this ranking"):
            if neighbours:
                st.dataframe(pd.DataFrame(neighbours)[
                    ["prompt_id", "task_category", "similarity", "prompt_text"]],
                    width="stretch", hide_index=True)
            else:
                st.write("No logged prompt shares wording with this one, so the ranking "
                         "came from the overall popularity baseline only - a starting "
                         "point, not a personalized answer.")
        if rows:
            detail = recommendations_to_df(res["recommendations"]).copy()
            detail.insert(0, "ai_model", detail["item_id"].map(
                lambda m: model_label(eng, m)))
            detail["because"] = [r["signal"] for r in rows]
            st.download_button("Download suggestions as CSV",
                               detail.to_csv(index=False).encode(),
                               "ai_model_suggestions.csv", "text/csv")
        how_it_works(sample, eng.status.eligible)
    else:
        render_discovery(matcher, eng, sample=sample)
        how_it_works(sample, eng.status.eligible)

# ------------------------------------------------------------ step 2: the log
elif idx == STEPS[1]:
    st.header("My trial log")
    st.caption("Everything the suggestions are built from - and where you add ratings "
               "so the answers improve.")
    tab_rank, tab_models, tab_log, tab_data = st.tabs(
        ["Ranked prompts", "Model performance", "Log a trial", "Data & uploads"])

    with tab_rank:
        pid = st.selectbox("Logged prompt",
                           [f"{p} :: {t[:70]}" for p, t in
                            zip(matcher.prompt_ids, matcher.prompt_texts)])
        chosen = pid.split(" :: ")[0]
        chosen_text = matcher.prompt_texts[matcher.prompt_ids.index(chosen)]
        line(chosen_text)
        mine = eng.data.interactions[eng.data.interactions["user_id"] == chosen].copy()
        mine["ai_model"] = mine["item_id"].map(lambda m: model_label(eng, m))
        st.dataframe(mine[["ai_model", "item_id", "interaction_value", "timestamp"]]
                     .rename(columns={"interaction_value": "your rating"}),
                     width="stretch", hide_index=True)
        section("Engine ranking", "your history, re-ranked")
        render_ranking(bars(recommendations_to_df(
            eng.recommend(chosen, top_k=4)["recommendations"]), eng,
            [{"prompt_id": chosen, "similarity": 1.0, "prompt_text": chosen_text}],
            logged=True), columns=4,
            ctx=ctx_for(matcher, eng, chosen_text, "log"))
        fineprint("Models you already trialled stay in the ranking on purpose: reusing "
                  "one you know works is a valid answer for a new prompt.")

    with tab_models:
        table, by_cat = matcher.stats_table()
        board = table.reset_index().rename(columns={
            "item_id": "model_id", "count": "trials", "mean": "mean rating"})
        board.insert(0, "ai_model", board["model_id"].map(lambda m: model_label(eng, m)))
        board["modality"] = board["model_id"].map(lambda m: model_meta(eng, m).get("modality", ""))
        board["best_for"] = board["model_id"].map(lambda m: model_meta(eng, m).get("best_for", ""))
        board = board.sort_values(["mean rating", "trials"], ascending=False)
        st.dataframe(board[["ai_model", "modality", "trials", "mean rating", "best_for"]],
                     width="stretch", hide_index=True)
        if sample:
            fineprint("On the demo every mean is identical (placeholder pattern), so this "
                      "order reflects trial counts, not quality.")
        if by_cat is not None:
            fig, ax = plt.subplots(figsize=(9, 3.4))
            by_cat.plot(kind="bar", ax=ax, colormap="viridis")
            ax.set_title("Mean trial rating by task category", fontsize=11)
            ax.set_ylabel("mean quality_score")
            fig.tight_layout()
            st.pyplot(fig)
            with st.expander("Category pivot as a table"):
                st.dataframe(by_cat, width="stretch")
        cats = eng.data.users.get("task_category")
        if cats is not None:
            with st.expander("Task mix of your prompt set"):
                st.dataframe(desc.describe_categorical(cats), width="stretch",
                             hide_index=True)

    with tab_log:
        st.caption("Record what a tool actually produced for you. Rows append to new "
                   f"files (`uploaded/{trial_log.TRIALS_FILE}`, "
                   f"`uploaded/{trial_log.PROMPTS_FILE}`); your source CSVs are never "
                   "modified. The 👍 / 👎 taps on cards write here too.")
        with st.form("trial_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                new_prompt = st.text_area("Prompt you ran", height=90)
                category = st.selectbox("Task category", sorted(CATEGORY_LABELS.values()))
            with c2:
                labels = [f"{model_label(eng, m)} ({m})" for m in eng.data.items["item_id"]]
                model_choice = st.selectbox("Model you used", labels)
                rating = st.slider("Output quality, 1-5", 1, 5, 4,
                                   help="1 unusable · 3 usable after edits · 5 ship as is")
                st.caption("Same prompt text twice = one prompt, two models - that is "
                           "what comparison-based ranking needs.")
            saved = st.form_submit_button("Save trial", type="primary")
        if saved:
            try:
                row = trial_log.append_trial(
                    MY_DIR, new_prompt, [k for k, v in CATEGORY_LABELS.items()
                                         if v == category][0],
                    model_choice.rsplit(" (", 1)[1].rstrip(")"), int(rating))
                st.success(f"Saved {row['model_id']} on {row['prompt_id']} "
                           f"(rating {row['quality_score']}).")
            except ValueError as exc:
                st.error(str(exc))

        mine = logged_rows()
        if not mine.empty:
            st.markdown(f"###### Your log ({len(mine)} trials, "
                        f"{mine['prompt_id'].nunique()} prompts)")
            view = mine.copy()
            view["ai_model"] = view["model_id"].map(lambda m: model_label(eng, m))
            st.dataframe(view[["prompt_id", "ai_model", "model_id", "quality_score",
                               "timestamp", "task_category"]],
                         width="stretch", hide_index=True)
            b1, b2 = st.columns(2)
            with b1:
                switch_to_my_log("Switch the app onto my log", "switch_log", paths)
            with b2:
                st.download_button("Download my log", mine.to_csv(index=False).encode(),
                                   "my_trials_log.csv", "text/csv")
            st.caption("Switching reuses the same column mapping, so eligibility, "
                       "ranking and evaluation all run on your real ratings.")
        else:
            fineprint("No personal trials logged yet.")

    with tab_data:
        st.markdown("###### Use your own trials")
        st.caption("Upload all three files with the same headers "
                   "(data/prompt_rec/README_DATA_COLLECTION.md) to run every step on "
                   "your own log.")
        c1, c2, c3 = st.columns(3)
        with c1:
            up_t = st.file_uploader("trials.csv", type="csv")
        with c2:
            up_p = st.file_uploader("prompts.csv", type="csv")
        with c3:
            up_m = st.file_uploader("models.csv", type="csv")
        if up_t and up_p and up_m:
            MY_DIR.mkdir(parents=True, exist_ok=True)
            saved_paths = {}
            for f, name in ((up_t, "trials.csv"), (up_p, "prompts.csv"),
                            (up_m, "models.csv")):
                p = MY_DIR / name
                f.save(p)
                saved_paths[name] = p
            st.session_state.custom_paths = saved_paths
            st.cache_resource.clear()
            st.success("Saved - rebuilding on your trials.")
            st.rerun()
        elif any((up_t, up_p, up_m)):
            st.info("Upload all three CSVs together; until then the sample data stays active.")
        if not logged_rows().empty:
            st.caption(f"Your in-app log is at data/prompt_rec/uploaded/{trial_log.TRIALS_FILE} "
                       f"(+ {trial_log.PROMPTS_FILE}). Export the two files plus models.csv as "
                       "the three uploads to switch the app onto it.")
        st.markdown("---")
        for rep in eng.status.validation_reports:
            with st.expander(f"Validation - {getattr(rep, 'name', 'dataset')}"):
                st.code("\n".join(rep.summary_lines()))
        with st.expander("Raw trial rows (first 60)"):
            st.dataframe(eng.data.interactions.head(60), width="stretch", hide_index=True)
        with st.expander("AI model catalogue"):
            st.dataframe(eng.data.items.rename(columns={"item_id": "model_id"}),
                         width="stretch", hide_index=True)

# ---------------------------------------------------------- step 3: evidence
else:
    st.header("Report & checks")
    st.caption("The evidence section of your write-up: which approach actually predicts "
               "your best-rated model, how clean the data is, and what each column means.")
    tab_eval, tab_quality, tab_scores = st.tabs(
        ["Model comparison", "Data quality", "Score guide & config"])

    with tab_eval:
        st.caption("Per-prompt chronological last-trial-out split: each prompt's two "
                   "most recent trials are hidden from fitting and the last one is "
                   "scored once. No test trial influences any ranking.")
        if st.button("Run evaluation", type="primary"):
            st.session_state["last_eval"] = run_chronological_comparison(
                eng.data, top_k=4, do_tuning=False)
        out = st.session_state.get("last_eval")
        if out and "error" in out:
            st.error(out["error"])
        elif out:
            m = st.columns(3)
            m[0].metric("Prompts evaluated", out["n_eval_users"])
            m[1].metric("Methods compared", len(out["test_table"]))
            best = out["test_table"].iloc[0]
            m[2].metric("Top by NDCG@4", f"{best['method']} {best['ndcg@4']:.3f}")
            st.dataframe(out["test_table"], width="stretch", hide_index=True)
            st.caption(f"Saved for your report: "
                       f"{save_table(out['test_table'], 'prompt_eval_comparison')}")
            with st.expander("Split strategy and limitations"):
                st.write(out["strategy"])
                st.warning(out["limitations"])
        else:
            st.info("Select **Run evaluation** to compute the comparison.")

    with tab_quality:
        st.markdown("###### Ingestion validation")
        for rep in eng.status.validation_reports:
            st.code("\n".join(rep.summary_lines()))
        st.markdown("###### Data-quality report")
        q = {k: v for k, v in eng.status.quality.items() if not isinstance(v, (list, dict))}
        st.json(q)
        if sample:
            st.warning("Demo data: the bundled trial ratings are deterministic "
                       "placeholders. They prove the pipeline computes and ranks "
                       "correctly - they say nothing about any real AI model.")

    with tab_scores:
        st.markdown("###### What each number means")
        st.dataframe(pd.DataFrame([
            {"column": "ranking score", "kind": "normalized_ranking_score",
             "meaning": "weighted blend of per-source scores, 0-1 within the results "
                        "shown. Ordering is meaningful; the magnitude is not.",
             "probability?": "no"},
            {"column": "similarity", "kind": "similarity",
             "meaning": "TF-IDF cosine between prompt texts (Evidence table).",
             "probability?": "no"},
            {"column": "your rating", "kind": "interaction_value",
             "meaning": "the 1-5 score you logged for one model output.",
             "probability?": "no"},
            {"column": "👍 / 👎 tap", "kind": "interaction_value",
             "meaning": f"quick feedback stored as a real trial row: 👍 = "
                        f"{TAP_RATING['good']}/5, 👎 = {TAP_RATING['poor']}/5. Only as "
                        "accurate as the tap.",
             "probability?": "no"},
            {"column": "reason chip", "kind": "logged evidence",
             "meaning": "the trial row that produced this card, quoted with its "
                        "similarity. No supporting row, no reason claimed.",
             "probability?": "no"},
            {"column": "Not now", "kind": "display filter",
             "meaning": "hides that model on this screen for this session. It writes "
                        "nothing and re-scores nothing.",
             "probability?": "no"},
            {"column": "match strength", "kind": "derived label",
             "meaning": "Strong ≥ 0.45, Fair ≥ 0.25, Weak > 0 on closest prompt "
                        "similarity - a stated uncertainty, not a modelled confidence.",
             "probability?": "no"},
        ]), width="stretch", hide_index=True)
        st.markdown("###### Engine state")
        st.code(eng.status.describe())
        h = eng.config["hybrid"]
        st.json({"weights": h["weights"], "top_k": h["top_k"],
                 "exclude_seen": h["exclude_seen"], "seed": eng.config.get("random_seed"),
                 "thresholds": eng.config["thresholds"]})
        st.caption("Scores are uncalibrated ranking values. Nothing in this app is "
                   "presented as a probability of a good output.")
