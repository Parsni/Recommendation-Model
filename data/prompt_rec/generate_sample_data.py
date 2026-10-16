"""Illustrative sample dataset for the prompt -> AI-model recommender.

*** THE quality_score VALUES HERE ARE PLACEHOLDER PATTERNS, NOT BENCHMARK
*** RESULTS OR REAL USER JUDGEMENTS. They exist so the pipeline can be
*** demonstrated end-to-end. Replace by collecting your own trials -
*** see README_DATA_COLLECTION.md next to this file.

Deterministic by construction (no RNG). Run:
    python data/prompt_rec/generate_sample_data.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
START = datetime(2025, 6, 1)

# prompt_id, task_category, language, prompt_text
PROMPTS = [
    ("P001", "coding", "en", "write a python function to parse a csv file and compute column averages"),
    ("P002", "coding", "en", "debug this react component that re-renders on every state change"),
    ("P003", "coding", "en", "explain the difference between a stack and a queue with code examples"),
    ("P004", "creative_writing", "en", "draft a short story opening about a lighthouse keeper in a storm"),
    ("P005", "creative_writing", "en", "rewrite this product description in a witty friendly tone"),
    ("P006", "creative_writing", "en", "write poem about monsoon season in mumbai for school assembly"),
    ("P007", "image_generation", "en", "generate a logo mockup minimal fox silhouette orange gradient flat design"),
    ("P008", "image_generation", "en", "create photorealistic interior design mockup scandinavian living room"),
    ("P009", "image_generation", "en", "draw a fantasy island map with mountains rivers and hidden caves"),
    ("P010", "reasoning_math", "en", "solve this probability puzzle three dice at least one six step by step"),
    ("P011", "reasoning_math", "en", "prove by induction sum of first n odd numbers equals n squared"),
    ("P012", "reasoning_math", "en", "two trains problem relative speed find meeting time show working"),
    ("P013", "translation", "hi", "translate this formal business letter from english to hindi maintain politeness levels"),
    ("P014", "translation", "en", "translate technical abstract from french to english keep terminology precise"),
    ("P015", "translation", "en", "localize this app UI string set for japanese market casual register"),
    ("P016", "data_analysis", "en", "analyze this sales csv find seasonality and summarize trends in table"),
    ("P017", "data_analysis", "en", "build regression model on housing dataset report r2 and residual diagnostics"),
    ("P018", "data_analysis", "en", "create pivot summary of survey responses by region and age band"),
    ("P019", "research_summary", "en", "summarize latest peer reviewed papers on recommendation system evaluation metrics"),
    ("P020", "research_summary", "en", "compare two cloud pricing pages and produce decision table with sources"),
    ("P021", "research_summary", "en", "find statistics on remote work productivity 2024 cite original studies"),
]

# models with short neutral descriptions (original wording, general knowledge)
MODELS = [
    ("M_gpt", "chat assistant (GPT-style)", "general chat, coding, writing, analysis, multimodal", "text+image", "freemium"),
    ("M_claude", "chat assistant (Claude-style)", "long form writing, careful reasoning, coding, document analysis", "text", "freemium"),
    ("M_gemini", "chat assistant (Gemini-style)", "multimodal questions, translation, search integrated answers", "text+image", "freemium"),
    ("M_deepseek", "open reasoning model", "math, coding, step by step logical solutions", "text", "free budget"),
    ("M_copilot", "ide assistant", "in editor code completion, refactoring, unit test scaffolding", "code", "subscription"),
    ("M_perplexity", "answer engine", "research questions needing fresh web sources and citations", "text", "freemium"),
    ("M_midjourney", "image generator", "stylized artistic images, concept art, logo mockups", "image", "subscription"),
    ("M_sdxl", "open image model", "photoreal images and local control with prompt weighting", "image", "open source"),
]

# which models were trialled per prompt (illustrative coverage pattern)
CAT_COVERAGE = {
    "coding": ["M_gpt", "M_claude", "M_deepseek", "M_copilot"],
    "creative_writing": ["M_claude", "M_gpt"],
    "image_generation": ["M_midjourney", "M_sdxl", "M_gpt"],
    "reasoning_math": ["M_deepseek", "M_claude", "M_gpt"],
    "translation": ["M_gemini", "M_gpt", "M_claude"],
    "data_analysis": ["M_gpt", "M_deepseek", "M_copilot"],
    "research_summary": ["M_perplexity", "M_gemini", "M_gpt"],
}


def placeholder_score(pi: int, mi: int) -> int:
    """Explicitly artificial deterministic pattern in 3..5 - a marker, not a measurement."""
    return 3 + ((pi + mi) % 3)


def build():
    inter_rows = []
    for pi, (pid, cat, lang, _txt) in enumerate(PROMPTS):
        coverage = CAT_COVERAGE[cat]
        for k, mid in enumerate(coverage):
            mi = [m[0] for m in MODELS].index(mid)
            inter_rows.append({
                "prompt_id": pid, "model_id": mid,
                "quality_score": placeholder_score(pi, mi),
                "timestamp": (START + timedelta(days=pi * 3 + k)).strftime("%Y-%m-%d %H:%M:%S"),
            })
    prompts = pd.DataFrame(PROMPTS, columns=["prompt_id", "task_category", "language", "prompt_text"])
    models = pd.DataFrame(MODELS, columns=["model_id", "provider_label", "best_for", "modality", "pricing"])
    models["description"] = models["best_for"]
    interactions = pd.DataFrame(inter_rows)
    return prompts, models, interactions


if __name__ == "__main__":
    prompts, models, interactions = build()
    prompts.to_csv(OUT / "prompts.csv", index=False)
    models.to_csv(OUT / "models.csv", index=False)
    interactions.to_csv(OUT / "trials.csv", index=False)
    print(f"prompts.csv: {len(prompts)} rows | models.csv: {len(models)} rows | trials.csv: {len(interactions)} rows")
    print("REMINDER: quality_score column is a deterministic placeholder pattern, not real results.")
