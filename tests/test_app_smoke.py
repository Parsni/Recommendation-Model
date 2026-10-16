"""Application startup + core interaction tests via streamlit.testing.AppTest.

Verifies the app launches, renders, builds the engine on the demo dataset,
and produces recommendations through the actual UI wiring.
"""
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture()
def app():
    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    return at


def test_app_launches_without_exceptions(app):
    assert not app.exception, f"app raised: {[e.value for e in app.exception]}"


def test_app_sidebar_navigation(app):
    at = app
    options = at.sidebar.radio[0].options
    for page in ["Executive overview", "Recommendations", "Config & fallback status"]:
        assert page in options
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"


def test_core_recommendation_flow(app):
    at = app
    at.sidebar.radio[0].set_value("Recommendations").run()
    assert not at.exception
    at.button[0].set_value(True).run()
    assert not at.exception, f"recommend flow raised: {[e.value for e in at.exception]}"
    # the ranked-results dataframe must be rendered somewhere on the page
    cols = {frozenset(d.value.columns) for d in at.dataframe}
    assert any({"item_id", "score", "rank"}.issubset(c) for c in cols), \
        f"no results dataframe among rendered tables: {[sorted(c) for c in cols]}"
    results = [d.value for d in at.dataframe
               if {"item_id", "score", "rank"}.issubset(set(d.value.columns))][0]
    assert len(results) >= 1


def test_evaluation_flow_runs(app):
    at = app
    at.sidebar.radio[0].set_value("Model comparison & evaluation").run()
    assert not at.exception
    at.button[0].set_value(True).run()
    assert not at.exception, f"evaluation flow raised: {[e.value for e in at.exception]}"
