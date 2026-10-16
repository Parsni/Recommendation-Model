"""PromptMatcher: recommend AI models for a brand-new prompt text.

Domain adapter for the prompt->model use case. The generic engine treats
prompts as "users"; a new prompt has no trial history (cold start), so this
matcher:
  1. embeds all known prompt texts with TF-IDF,
  2. finds the most similar historical prompts to the new text,
  3. blends the engine's recommendations for those neighbours weighted by
     text similarity.
Explanations always name the real neighbour prompts that produced a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.engine import RecommendationEngine
from src.recommenders.base import Recommendation
from src.utils.helpers import load_config, PROJECT_ROOT

PROMPT_DIR = PROJECT_ROOT / "data" / "prompt_rec"


class PromptMatcher:
    def __init__(self, interactions_path=None, prompts_path=None, models_path=None,
                 config: dict | None = None):
        config = config or load_config(PROJECT_ROOT / "config" / "prompt_config.json")
        self.engine = (RecommendationEngine(config)
                       .load(interactions_path or PROMPT_DIR / "trials.csv",
                             prompts_path or PROMPT_DIR / "prompts.csv",
                             models_path or PROMPT_DIR / "models.csv")
                       .build())
        prompts_df = self.engine.data.users
        self.prompt_ids = prompts_df["user_id"].tolist()   # canonical name after mapping
        self.prompt_texts = prompts_df["prompt_text"].fillna("").astype(str).tolist()
        self.categories = (prompts_df["task_category"].astype(str).tolist()
                           if "task_category" in prompts_df else [None] * len(self.prompt_ids))
        self.vectorizer = TfidfVectorizer(stop_words="english", token_pattern=r"[a-z0-9]+",
                                          sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform([t.lower() for t in self.prompt_texts])

    # ------------------------------------------------------------------
    def nearest_prompts(self, prompt_text: str, n: int = 5) -> list[dict]:
        vec = self.vectorizer.transform([prompt_text.lower()])
        sims = cosine_similarity(vec, self.tfidf).ravel()
        order = np.argsort(-sims)[:n]
        return [{"prompt_id": self.prompt_ids[i], "similarity": float(sims[i]),
                 "task_category": self.categories[i], "prompt_text": self.prompt_texts[i]}
                for i in order if sims[i] > 0]

    def recommend_for_prompt(self, prompt_text: str, top_k: int = 4,
                             candidate_ids: list[str] | None = None) -> dict:
        """candidate_ids applies a hard metadata constraint (e.g. only image
        models); it filters the candidate pool, it does not re-weight scores."""
        neighbours = self.nearest_prompts(prompt_text, n=5)
        if not neighbours:
            res = self.engine.recommend("__cold__", top_k=top_k,
                                        candidate_ids=candidate_ids)   # popularity baseline
            return {"recommendations": res["recommendations"],
                    "matched_prompts": [],
                    "notes": ["No text overlap with any known prompt - showing overall "
                              "best-average-rated models (popularity baseline) only."]}
        # blend engine recommendations of neighbour prompts by similarity
        agg: dict[str, dict] = {}
        for nb in neighbours:
            recs = self.engine.recommend(nb["prompt_id"],
                                         top_k=self.engine.config["hybrid"]["top_k"],
                                         candidate_ids=candidate_ids)["recommendations"]
            for r in recs:
                entry = agg.setdefault(r.item_id, {"score": 0.0, "drivers": [], "sample": r})
                entry["score"] += nb["similarity"] * r.score
                if nb["similarity"] > 0.05:
                    entry["drivers"].append(
                        f"{nb['prompt_id']} (sim {nb['similarity']:.2f}: \"{nb['prompt_text'][:60]}...\")")
        ranked = sorted(agg.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
        if not ranked and candidate_ids:
            return {"recommendations": [], "matched_prompts": neighbours,
                    "notes": ["The model-type constraint left no scored candidates - "
                              "none of the similar logged prompts recommends a model of "
                              "that type."]}
        out = []
        for rank, (model, e) in enumerate(ranked[:top_k], start=1):
            sample = e["sample"]
            sample.item_id = model
            sample.score = e["score"]
            sample.rank = rank
            meta = self.engine.data.item_metadata(model)
            label = meta.get("provider_label", model)
            sample.explanation = (
                f"{label} scored well on prompts similar to yours: "
                + "; via ".join(dict.fromkeys(e["drivers"]))
                + f" | per-source reasoning: {sample.explanation}")
            sample.limitations = (
                "Similarity-blended score from neighbour prompts, not a probability; "
                "valid only to the extent your trial ratings represent your preferences. "
                + sample.limitations)
            out.append(sample)
        notes = []
        if neighbours[0]["similarity"] < 0.15:
            notes.append("Weakest-match warning: best text similarity is low "
                         f"({neighbours[0]['similarity']:.2f}); suggestions are "
                         "uncertain for this prompt.")
        return {"recommendations": out, "matched_prompts": neighbours, "notes": notes}

    def stats_table(self) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Mean trial score per model overall and per task category."""
        inter = self.engine.data.interactions
        table = inter.groupby("item_id")["interaction_value"].agg(["count", "mean"]).round(2)
        if "task_category" in self.engine.data.users.columns:
            joined = inter.merge(self.engine.data.users[["user_id", "task_category"]],
                                 on="user_id", how="left")
            by_cat = joined.pivot_table(index="task_category", columns="item_id",
                                        values="interaction_value", aggfunc="mean").round(2)
            return table, by_cat
        return table, None
