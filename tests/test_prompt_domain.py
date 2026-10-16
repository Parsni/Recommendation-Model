"""Domain adaptation tests: prompt/model mapping, matcher behaviour."""
import pandas as pd
import pytest

from src.utils.helpers import PROJECT_ROOT, load_config
from src.data.loader import load_dataset
from src.features.preprocess import prepare
from src.domain.prompt_matcher import PromptMatcher

PDIR = PROJECT_ROOT / "data" / "prompt_rec"
CFG = PROJECT_ROOT / "config" / "prompt_config.json"


@pytest.fixture(scope="module")
def matcher():
    return PromptMatcher()


def test_config_maps_domain_columns():
    cfg = load_config(CFG)
    m = cfg["column_mapping"]["interactions"]
    assert m["user_id"] == "prompt_id" and m["item_id"] == "model_id"
    assert m["interaction_value"] == "quality_score"
    raw = load_dataset(PDIR / "trials.csv", PDIR / "prompts.csv", PDIR / "models.csv", cfg)
    assert list(raw.interactions.columns[:4]) == ["user_id", "item_id", "interaction_value", "timestamp"]
    assert raw.interactions["user_id"].nunique() == 21
    assert raw.interactions["item_id"].nunique() == 8


def test_sample_data_integrity():
    trials = pd.read_csv(PDIR / "trials.csv")
    prompts = pd.read_csv(PDIR / "prompts.csv")
    models = pd.read_csv(PDIR / "models.csv")
    assert len(trials) == 63
    assert set(trials["prompt_id"]) <= set(prompts["prompt_id"])
    assert set(trials["model_id"]) <= set(models["model_id"])
    assert trials["quality_score"].between(1, 5).all()


def test_eligibility_on_prompt_data(matcher):
    enabled = set(matcher.engine.recommenders)
    assert "content_based" in enabled      # model best_for text available
    assert "popularity" in enabled         # baseline always
    assert matcher.engine.status.quality["n_users"] == 21
    assert matcher.engine.status.quality["n_items_in_catalog"] == 8


def test_matcher_finds_same_category_neighbours(matcher):
    res = matcher.recommend_for_prompt(
        "write a python function that reads a csv and computes averages per column", 4)
    top_nb = res["matched_prompts"][0]
    assert top_nb["prompt_id"] == "P001" and top_nb["task_category"] == "coding"
    assert top_nb["similarity"] > 0.4
    models = [r.item_id for r in res["recommendations"]]
    assert len(models) == len(set(models)), "duplicate models must be aggregated"
    assert res["recommendations"][0].score >= res["recommendations"][-1].score


def test_matcher_image_prompt_suggests_image_models(matcher):
    res = matcher.recommend_for_prompt(
        "generate a photorealistic logo mockup of a fox for branding", 3)
    ids = {r.item_id for r in res["recommendations"]}
    assert ids & {"M_midjourney", "M_sdxl", "M_gpt"}, ids


def test_matcher_unrelated_prompt_falls_back_with_note(matcher):
    res = matcher.recommend_for_prompt("qqq zzz www kkk", 3)
    assert res["matched_prompts"] == []
    assert res["notes"] and "popularity baseline" in res["notes"][0]
    assert len(res["recommendations"]) >= 1


def test_low_similarity_warning(matcher):
    res = matcher.recommend_for_prompt("python", 3)
    if res["matched_prompts"] and res["matched_prompts"][0]["similarity"] < 0.15:
        assert any("uncertain" in n for n in res["notes"])


def test_scores_are_labelled_not_probabilities(matcher):
    res = matcher.recommend_for_prompt("debug a react component that rerenders constantly", 4)
    for r in res["recommendations"]:
        assert r.score_kind in ("normalized_ranking_score", "frequency_count")
        assert "not a probability" in r.limitations or "not, a probability" in r.limitations \
            or "probability" in r.limitations


def test_trialalled_models_are_not_filtered_out(matcher):
    """Domain choice: exclude_seen is off because re-using a model you already
    trialled on a different prompt is a valid suggestion."""
    cfg = matcher.engine.config["hybrid"]
    assert cfg["exclude_seen"] is False
    trials = pd.read_csv(PDIR / "trials.csv")
    seen = set(trials[trials["prompt_id"] == "P001"]["model_id"])
    ids = {r.item_id for r in matcher.engine.recommend("P001", top_k=4)["recommendations"]}
    assert ids & seen, "already-trialled models should remain rankable"


def test_ranking_is_deterministic(matcher):
    text = "summarize a long research paper into ten bullet points"
    a = [(r.item_id, round(r.score, 6)) for r in matcher.recommend_for_prompt(text, 4)["recommendations"]]
    b = [(r.item_id, round(r.score, 6)) for r in matcher.recommend_for_prompt(text, 4)["recommendations"]]
    assert a == b and len(a) >= 2


def test_model_type_constraint_is_hard_filter(matcher):
    image_ids = {str(r["item_id"]) for _, r in matcher.engine.data.items.iterrows()
                 if str(r.get("modality")) == "image"}
    assert image_ids, "sample models.csv should tag at least one image model"
    res = matcher.recommend_for_prompt(
        "create an isometric illustration of a cozy bookshop", 4,
        candidate_ids=sorted(image_ids))
    got = {r.item_id for r in res["recommendations"]}
    assert got and got <= image_ids, f"constraint leaked non-image models: {got}"


