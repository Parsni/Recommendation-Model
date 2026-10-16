"""Export and reporting artifact tests (files actually written)."""
import pandas as pd

from src.reporting.exporter import (recommendations_to_df, save_bar_chart,
                                    save_table, load_table, recommendations_to_df as _rd)
from src.recommenders.popularity import PopularityRecommender
from src.utils.helpers import TABLES_DIR, FIGURES_DIR


def test_recommendations_to_df_columns(demo_data):
    rec = PopularityRecommender(demo_data.config).fit(demo_data)
    df = recommendations_to_df(rec.recommend("U001", 3))
    assert list(df.columns) == ["item_id", "score", "score_kind", "rank",
                                "sources", "explanation", "limitations"]
    assert len(df) == 3


def test_save_and_reload_table():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = save_table(df, "unit_test_table")
    back = load_table("unit_test_table")
    assert back.equals(df)
    assert path.endswith("unit_test_table.csv")
    (TABLES_DIR / "unit_test_table.csv").unlink()


def test_save_bar_chart_creates_png():
    path = save_bar_chart({"a": 1.0, "b": 2.5, "c": 1.7},
                          "Unit test chart", "unit_test_chart")
    assert path != "table-only" and (FIGURES_DIR / "unit_test_chart.png").exists()
    (FIGURES_DIR / "unit_test_chart.png").unlink()
