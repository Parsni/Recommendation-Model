"""Universal Recommendation Engine - Streamlit application.

Launch:  streamlit run app.py
Runs fully locally with the bundled synthetic demo dataset; real CSVs can be
uploaded. All numbers shown come from actual engine computation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data.loader import load_dataset, read_csv
from src.data.schema import validate_interactions, apply_mapping
from src.engine import RecommendationEngine
from src.evaluation.compare import run_chronological_comparison
from src.features.quality import quality_report
from src.statistics import descriptive as desc
from src.statistics import inferential as inf
from src.statistics import models as mdl
from src.reporting.exporter import recommendations_to_df, save_table
from src.utils.helpers import PROJECT_ROOT, load_config

st.set_page_config(page_title="Universal Recommendation Engine", layout="wide")
DEMO = PROJECT_ROOT / "data" / "synthetic_demo"


@st.cache_resource(show_spinner="Preparing data and fitting eligible algorithms...")
def build_engine(inter_path, users_path, items_path, config):
    eng = RecommendationEngine(config).load(inter_path, users_path, items_path)
    eng.build()
    return eng


def get_config() -> dict:
    cfg = load_config()
    overrides = st.session_state.get("config_overrides")
    if overrides:
        cfg["hybrid"]["weights"].update(overrides.get("weights", {}))
        cfg["hybrid"]["top_k"] = overrides.get("top_k", cfg["hybrid"]["top_k"])
    return cfg


st.sidebar.title("Recommendation Engine")
page = st.sidebar.radio("Section", ["Executive overview", "Data & quality",
                                    "Recommendations", "Model comparison & evaluation",
                                    "Statistical analysis", "Config & fallback status"])
st.sidebar.caption("Domain-agnostic engine. Customize data, features and analysis "
                   "for the chosen problem later (see docs/customization_guide.md).")

if "data_source" not in st.session_state:
    st.session_state.data_source = "demo"

if page == "Data & quality":
    st.header("Dataset upload & validation")
    up_i = st.file_uploader("Interactions CSV (user_id, item_id, [interaction_value], [timestamp])",
                            type="csv", key="ui")
    up_u = st.file_uploader("Users CSV (optional)", type="csv", key="uu")
    up_it = st.file_uploader("Items CSV (optional)", type="csv", key="uii")
    if up_i is not None:
        folder = PROJECT_ROOT / "data" / "uploaded"
        folder.mkdir(parents=True, exist_ok=True)
        p_i = folder / "interactions.csv"
        up_i.save(p_i)
        p_u = folder / "users.csv"
        p_it = folder / "items.csv"
        up_u.save(p_u) if up_u else None
        up_it.save(p_it) if up_it else None
        st.session_state.data_source = "upload"
        st.session_state.upload_paths = (p_i,
                                         p_u if up_u else None,
                                         p_it if up_it else None)
        st.success("Uploaded. Switch to another section to build on this data.")
    if st.button("Use bundled synthetic demo dataset"):
        st.session_state.data_source = "demo"

    if st.session_state.data_source == "upload":
        p_i, p_u, p_it = st.session_state.upload_paths
        raw_df = read_csv(p_i)
        cfg = get_config()
        mapped = apply_mapping(raw_df, cfg["column_mapping"]["interactions"], "interactions")
        st.write("Detected columns after mapping:", list(mapped.columns))
        _, rep = validate_interactions(raw_df, cfg["column_mapping"]["interactions"])
        st.code("\n".join(rep.summary_lines()))
        st.dataframe(raw_df.head(50), width="stretch")
    else:
        st.caption("Using the bundled synthetic demo (placeholder data for functionality "
                   "testing only - results are not real-world evidence).")
        eng = build_engine(DEMO / "interactions.csv", DEMO / "users.csv", DEMO / "items.csv",
                           get_config())
        for rep in eng.status.validation_reports:
            st.code("\n".join(rep.summary_lines()))
        if eng.data.interactions is not None:
            st.dataframe(eng.data.interactions.head(50), width="stretch")

# ---------------------------------------------------------------------------
else:
    if st.session_state.data_source == "upload":
        p_i, p_u, p_it = st.session_state.upload_paths
        eng = build_engine(p_i, p_u, p_it, get_config())
    else:
        eng = build_engine(DEMO / "interactions.csv", DEMO / "users.csv", DEMO / "items.csv",
                           get_config())

    if page == "Executive overview":
        st.header("Executive overview")
        q = eng.status.quality
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Users", q.get("n_users", 0))
        c2.metric("Items (catalog)", q.get("n_items_in_catalog", 0))
        c3.metric("Interactions", q.get("n_interactions", 0))
        c4.metric("Matrix density", f"{q.get('matrix_density', 0):.3f}")
        st.markdown("**Engine status**")
        st.code(eng.status.describe())
        st.caption("Scores shown by this app are ranking scores / normalized blends / "
                   "similarities - never probabilities, unless a module explicitly says so.")

    elif page == "Recommendations":
        st.header("Generate recommendations")
        users = eng.data.user_ids + ["<new/cold-start user id>"]
        col1, col2 = st.columns([2, 1])
        with col1:
            uid = st.selectbox("User", users)
        with col2:
            top_k = st.slider("Top-K", 1, 20, 5)
        exclude_seen = st.checkbox("Exclude already-consumed items", value=True)
        diversify = st.checkbox("Diversify results", value=False)
        restrict = st.multiselect("Restrict candidates (empty = full catalog)",
                                  eng.data.item_ids)
        st.subheader("Eligible algorithms")
        st.write(sorted(eng.recommenders))
        if uid == "<new/cold-start user id>":
            uid = st.text_input("Enter a cold-start user id", "NEW_USER_1")
        if st.button("Recommend", type="primary"):
            res = eng.recommend(uid, top_k=top_k, exclude_seen=exclude_seen,
                                 candidate_ids=restrict or None, diversify=diversify)
            for note in res["notes"]:
                st.info(note)
            df = recommendations_to_df(res["recommendations"])
            if df.empty:
                st.warning("No recommendations were produced for this user/config - "
                           "see fallback status for details.")
            else:
                st.dataframe(df.drop(columns=["limitations"]), width="stretch")
                with st.expander("Limitations per item"):
                    st.dataframe(df[["item_id", "limitations"]], width="stretch")
                st.download_button("Download CSV", df.to_csv(index=False).encode(),
                                   f"recommendations_{uid}.csv", "text/csv")
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.bar(df["item_id"], df["score"], color="#3b6ea5")
                ax.set_title(f"Hybrid normalized ranking scores for {uid} (not probabilities)")
                st.pyplot(fig)

    elif page == "Model comparison & evaluation":
        st.header("Model comparison & evaluation")
        st.caption("Chronological per-user last-out split: training excludes each user's "
                   "two most recent interactions; validation selects hybrid weights; "
                   "test is consulted once. Synthetic-data demo of the methodology only.")
        if st.button("Run evaluation", type="primary"):
            with st.spinner("Evaluating all eligible methods..."):
                out = run_chronological_comparison(eng.data)
            if "error" in out:
                st.error(out["error"])
            else:
                st.code(out["strategy"])
                st.metric("Evaluated users", out["n_eval_users"])
                st.dataframe(out["test_table"], width="stretch")
                table = out["test_table"]
                fig, ax = plt.subplots(figsize=(7, 3.2))
                kcol = [c for c in table.columns if c.startswith("ndcg")][0]
                ax.barh(table["method"], table[kcol], color="#3b6ea5")
                ax.set_title(f"{kcol} by method (held-out test)")
                st.pyplot(fig)
                if out["weight_tuning"]:
                    st.subheader("Hybrid weight tuning (validation only)")
                    st.write("Best weights:", json.dumps(out["weight_tuning"]["best_weights"]))
                    st.dataframe(pd.DataFrame(out["weight_tuning"]["trials"]))
                st.warning(out["limitations"])
                path = save_table(table, "model_comparison")
                st.caption(f"Saved: {path}")

    elif page == "Statistical analysis":
        st.header("Statistical analysis (generic utilities)")
        st.caption("Modules here are reusable Business Analytics utilities; the tests "
                   "actually used for the final DA problem will be selected according to "
                   "the research question and variable types.")
        inter = eng.data.interactions
        tab1, tab2, tab3, tab4 = st.tabs(["Descriptives", "Associations", "Group tests",
                                          "Trend / forecast"])
        with tab1:
            if eng.data.has_values:
                s = desc.describe_numeric(inter["interaction_value"])
                st.json(s)
                st.dataframe(desc.missing_data_diagnostics(
                    eng.data.items if eng.data.items is not None else inter))
                o = pd.to_numeric(inter["interaction_value"], errors="coerce").dropna()
                counts = o.value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(counts.index.astype(str), counts.values, color="#3b6ea5")
                ax.set_title("Interaction value distribution")
                st.pyplot(fig)
            else:
                st.info("No numeric interaction_value column in this dataset.")
        with tab2:
            if eng.data.items is not None:
                nums = eng.data.item_numerical_fields
                cats = eng.data.item_categorical_fields
                if len(nums) >= 2:
                    st.dataframe(desc.correlation_table(eng.data.items, nums))
                if cats and nums:
                    a, b = st.selectbox("Categorical A", cats), st.selectbox("Numeric", nums)
                    if st.button("Group means by category"):
                        st.dataframe(eng.data.items.groupby(a)[b].agg(["count", "mean"]).round(3))
            else:
                st.info("Item metadata needed for association analysis.")
        with tab3:
            if eng.data.items is not None and len(eng.data.item_categorical_fields or []) >= 2:
                a, b = st.selectbox("Categorical A", eng.data.item_categorical_fields), \
                    st.selectbox("Categorical B", eng.data.item_categorical_fields)
                if st.button("Chi-square test"):
                    st.json(desc.categorical_association(eng.data.items, a, b))
            if eng.data.users is not None and eng.data.user_numerical_fields:
                num = st.selectbox("User numeric", eng.data.user_numerical_fields)
                grp = st.selectbox("Group by", eng.data.user_categorical_fields or [None])
                if grp and st.button("Two-group t-test"):
                    d = eng.data.users[[num, grp]].dropna()
                    levels = d[grp].unique()[:2]
                    if len(levels) == 2:
                        st.json(inf.two_sample_ttest(d.loc[d[grp] == levels[0], num],
                                                    d.loc[d[grp] == levels[1], num]))
                    else:
                        st.error("Need exactly two group levels for a t-test; found "
                                 f"{len(levels)}. Use ANOVA/regression later.")
        with tab4:
            if eng.data.has_timestamps:
                daily = inter.copy()
                daily["day"] = pd.to_datetime(daily["timestamp"]).dt.date
                counts = daily.groupby("day").size().to_numpy()
                st.json(mdl.forecast_series(counts, periods_ahead=3))
                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(counts, marker="o", ms=3, color="#3b6ea5")
                ax.set_title("Daily interaction counts (actual data)")
                st.pyplot(fig)
            else:
                st.info("No timestamps - time-series utilities unavailable for this data.")

    elif page == "Config & fallback status":
        st.header("Configuration & reproducibility")
        st.json(eng.config, expanded=False)
        st.subheader("Algorithm availability")
        rows = [{"algorithm": n, "status": "enabled"} for n in sorted(eng.recommenders)]
        rows += [{"algorithm": n, "status": f"excluded: {r}"}
                 for n, r in eng.status.excluded.items()]
        rows += [{"algorithm": n, "status": f"disabled after failure: {r}"}
                 for n, r in eng.status.disabled_after_failure.items()]
        st.dataframe(pd.DataFrame(rows))
        st.subheader("Fallback / failure log")
        entries = eng.failure_log.read_all()
        if entries:
            st.dataframe(pd.DataFrame(entries[-50:]), width="stretch")
        else:
            st.info("No failures recorded for this run.")
        if eng.data.items is not None and eng.data.item_categorical_fields:
            st.subheader("Hybrid weights (applied on rebuild)")
            w = st.number_input("weight per source (equal)", 0.05, 1.0, 0.25, 0.05)
            if st.button("Apply & rebuild engine"):
                names = sorted(eng.recommenders)
                st.session_state.config_overrides = {
                    "weights": {n: w for n in names}}
                st.cache_resource.clear()
                st.rerun()
