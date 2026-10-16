"""Headless smoke tests for the app-style prompt recommender.

Covers the three workflow steps, the discovery carousel, the comparison grid,
the reason chips a card is allowed to show, the collapsed filter strip, the
one-tap feedback that writes real trial rows, and the requirement that step 1
keeps its honesty caveats in chips and tooltips rather than warning boxes.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = "app_prompt.py"
STEP1, STEP2, STEP3 = ("🔍  Find a model", "📝  My trial log", "📊  Report & checks")
MY_DIR = Path("data") / "prompt_rec" / "uploaded"
QUERY = ("write a python script to clean survey responses in a csv and "
         "summarize by region")
DEMO_CHIP = "Demo data · placeholder ratings, not real benchmarks"


@pytest.fixture(autouse=True)
def clean_personal_log():
    """One-tap tests append real rows; never leave invented trials behind."""
    shutil.rmtree(MY_DIR, ignore_errors=True)
    yield
    shutil.rmtree(MY_DIR, ignore_errors=True)


def boot(step: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, f"app raised: {at.exception}"
    if step:
        at.sidebar.radio[0].set_value(step)
        at.run()
        assert not at.exception, f"{step} raised: {at.exception}"
    return at


def click(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception, f"clicking {label!r} raised: {at.exception}"
    return at


def refine(at: AppTest) -> AppTest:
    """Open the collapsed filter strip if its widgets are not on screen yet."""
    if not at.selectbox:
        return click(at, "Refine ▾")
    return at


def score_bars(at):
    """One st.progress per rendered card."""
    return at.get("progress")


def cards(at):
    return [m.value for m in at.markdown if "class='rankbadge'" in m.value]


def reasons(at):
    return [m.value for m in at.markdown if "chip blue" in m.value]


def signals(at):
    return [m.value for m in at.markdown if "class='muted'>signals" in m.value]


def feedlines(at):
    return [m.value for m in at.markdown if "class='feedline'" in m.value]


def chips(at):
    return [m.value for m in at.markdown if "class='chip" in m.value]


def sections(at):
    return [m.value for m in at.markdown if "sectionTitle" in m.value]


def frames(at, *columns):
    return [d for d in at.dataframe if set(columns) <= set(d.value.columns)]


def asked(at: AppTest, text: str = QUERY) -> AppTest:
    at.text_area[0].set_value(text)
    return click(at, "Find my model")


def test_step1_opens_on_a_search_plus_browse_front_page():
    at = boot()
    assert at.sidebar.radio[0].value == STEP1
    hero = "".join(m.value for m in at.markdown)
    assert "Which AI should run this prompt?" in hero
    assert any("class='chip amber'" in c and "placeholder ratings" in c for c in chips(at)), \
        "the demo-data caveat must still be visible"
    assert any("Recommended for you" in s for s in sections(at))
    labels = [b.label for b in at.button]
    assert "Find my model" in labels and "Refine ▾" in labels
    assert "◀" in labels and "▶" in labels, "horizontal browse controls missing"
    assert len(score_bars(at)) == 1, "carousel shows one card at a time"
    assert re.search(r">\s*1 of 4", "".join(m.value for m in at.markdown))
    assert reasons(at), "carousel card must state why it appears"


def test_step1_uses_no_callout_boxes():
    """The user asked for an app-style screen: caveats are chips and fineprint."""
    at = asked(boot())
    assert at.warning == [], f"step 1 must not render a warning box: {at.warning}"
    assert at.error == []


def test_carousel_arrow_changes_the_visible_model():
    at = boot()
    first = cards(at)[0]
    click(at, "▶")
    assert cards(at)[0] != first, "next arrow did not advance the carousel"


def test_search_this_prompt_loads_the_logged_text_into_the_box():
    at = boot()
    click(at, "Search this prompt")
    assert at.text_area[0].value.strip(), "seeded prompt did not reach the search box"
    assert not any("Best match" in s for s in sections(at)), \
        "seeding must not run the query by itself"


def test_browse_button_returns_to_the_discovery_panel():
    at = asked(refine(boot()))
    assert len(cards(at)) >= 3
    click(at, "Browse")
    assert any("Recommended for you" in s for s in sections(at))
    assert len(score_bars(at)) == 1


def test_query_renders_comparison_cards_with_real_reasons():
    at = asked(boot())
    names = "".join(cards(at))
    assert "chat assistant (GPT-style)" in names
    assert len(cards(at)) >= 3, "side-by-side grid expected"
    assert len(score_bars(at)) >= 2, "one score bar per suggested model"
    assert any("Best match for your prompt" in s for s in sections(at))
    assert signals(at) and "popularity" in signals(at)[0] \
        and "content_based" in signals(at)[0]
    assert any("You rated it" in r or "In your log:" in r for r in reasons(at)), \
        "every card needs a reason grounded in a trial row"
    assert re.search(r"(Strong|Fair|Weak|No wording) match · sim \d",
                     "".join(chips(at))), "match strength chip missing"
    assert frames(at, "prompt_id", "similarity"), "evidence table missing"
    assert any("Download suggestions as CSV" in b.label for b in at.get("download_button"))


def test_filters_are_collapsed_until_refine_is_tapped():
    at = boot()
    assert at.selectbox == [], "filter strip must start collapsed"
    at = refine(at)
    assert {s.label for s in at.selectbox} == {"Model type", "Results"}
    assert at.checkbox[0].label == "Hide models I already rated for this prompt"


def test_model_type_filter_is_applied():
    at = refine(boot())
    at.text_area[0].set_value("create an isometric illustration of a cozy bookshop")
    next(s for s in at.selectbox if s.label == "Model type").set_value("image")
    click(at, "Find my model")
    names = "".join(cards(at))
    assert "image generator" in names, names
    assert "chat assistant" not in names, f"filter leaked: {names}"


def test_unmatched_prompt_states_the_popularity_fallback():
    at = asked(boot(), "qqq zzz www kkk")
    assert any("popularity baseline" in f for f in feedlines(at))
    assert frames(at, "prompt_id", "similarity") == []
    assert any("No wording match · sim 0.00" in c for c in chips(at))
    assert any("In your log:" in r for r in reasons(at))


def test_not_now_hides_a_model_and_undo_restores_it():
    at = asked(refine(boot()))
    before = "".join(cards(at))
    assert "chat assistant" in before
    click(at, "Not now")
    assert at.session_state["hidden_models"], "hidden list should record the model"
    assert "".join(cards(at)) != before, "hiding did not change the grid"
    assert any(b.label == "Undo 1" for b in at.button)
    click(at, "Undo 1")
    assert "".join(cards(at)) == before, "undo did not restore the ranking"
    assert at.session_state["hidden_models"] == []


def test_quick_tap_writes_a_real_trial_row():
    at = asked(boot())
    click(at, "👍")
    assert any("Logged 4/5" in f for f in feedlines(at)), "no on-screen confirmation"
    trials = pd.read_csv(MY_DIR / "my_trials.csv")
    prompts = pd.read_csv(MY_DIR / "my_prompts.csv")
    assert list(trials["quality_score"]) == [4]
    assert prompts.iloc[0]["prompt_text"] == QUERY
    assert any("your log: 1 trials over 1 prompts" in m.value for m in at.markdown
               if "class='muted'" in m.value)
    assert any("Rank from my logged trials" in b.label for b in at.button)
    at = refine(at)
    at.checkbox[0].set_value(True)
    at.run()
    assert not at.exception
    assert any("1 held back" in c for c in chips(at)), \
        "hide-models-I-rated should drop the tapped model"


def test_hiding_every_candidate_states_recovery_and_undo_restores():
    """candidate_ids=[] legitimately means 'nothing may appear' - the screen
    must say so and offer the control that fixes it."""
    at = refine(boot())
    next(s for s in at.selectbox if s.label == "Model type").set_value("image")
    at = asked(at)
    for _ in range(6):
        hides = [b for b in at.button if b.label == "Not now"]
        if not hides:
            break
        hides[0].click()
        at.run()
        assert not at.exception
    assert at.session_state["hidden_models"], "every image model should be hidden"
    assert not cards(at), "grid must be empty once all candidates are hidden"
    assert any("Nothing left to show" in f for f in feedlines(at)), \
        "empty screen needs a stated recovery path"
    click(at, f"Undo {len(at.session_state['hidden_models'])}")
    assert at.session_state["hidden_models"] == []
    assert cards(at), "undo did not bring results back"


def test_step2_tabs_show_history_board_and_logger():
    at = boot(STEP2)
    assert at.header[0].value == "My trial log"
    assert [t.label for t in at.get("tab")] == [
        "Ranked prompts", "Model performance", "Log a trial", "Data & uploads"]
    assert frames(at, "ai_model", "your rating"), "trial table missing"
    assert len(score_bars(at)) >= 1, "per-prompt ranking did not render"
    assert any("You rated it" in r for r in reasons(at))
    assert frames(at, "ai_model", "mean rating", "best_for"), "leaderboard missing"
    assert any(b.label == "Save trial" for b in at.button), "trial logging form missing"


def test_step3_evaluation_runs_and_explains_scores():
    at = boot(STEP3)
    assert at.header[0].value == "Report & checks"
    click(at, "Run evaluation")
    assert at.metric[0].label == "Prompts evaluated"
    assert int(at.metric[0].value) > 0
    assert frames(at, "method", "ndcg@4"), "comparison table missing"
    guide = frames(at, "column", "probability?")
    assert guide and set(guide[0].value["probability?"]) == {"no"}
    rows = guide[0].value
    assert "ranking score" in set(rows["column"]), "scores must be named as ranking scores"
    tap = rows[rows["column"] == "👍 / 👎 tap"].iloc[0]
    assert "4/5" in tap["meaning"] and "2/5" in tap["meaning"], "tap mapping undocumented"
    assert "display filter" in set(rows["kind"]), "hide filter must be labelled as one"
