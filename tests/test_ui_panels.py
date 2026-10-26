"""Unit tests for pure helpers in src/routing/ui_panels.py."""
from __future__ import annotations

import pytest

from src.routing import ui_panels as ui


class TestClassifyTier:
    def test_cheap_vs_median(self):
        assert ui.classify_tier(1.0, 4.0, lower_is_better=True) == "cheap"

    def test_fair_band(self):
        assert ui.classify_tier(4.0, 4.0, lower_is_better=True) == "fair"
        assert ui.classify_tier(5.9, 4.0, lower_is_better=True) == "fair"

    def test_pricy(self):
        assert ui.classify_tier(8.0, 4.0, lower_is_better=True) == "pricy"

    def test_fast_slow_labels(self):
        assert ui.classify_tier(100.0, 400.0, lower_is_better=False) == "fast"
        assert ui.classify_tier(400.0, 400.0, lower_is_better=False) == "average"
        assert ui.classify_tier(2000.0, 400.0, lower_is_better=False) == "slow"

    def test_zero_median_fallback(self):
        assert ui.classify_tier(1.0, 0.0) == "fair"


class TestCostContextLine:
    def test_free(self):
        assert "free" in ui.cost_context_line(0.0)

    def test_small_cost_mentions_questions_per_dollar(self):
        line = ui.cost_context_line(0.0002)
        assert "$0.0002" in line
        assert "5,000 questions for $1" in line

    def test_larger_cost_six_decimals_when_tiny(self):
        line = ui.cost_context_line(0.000015)
        assert "$0.000015" in line


class TestLatencyLine:
    def test_sub_second(self):
        assert "under a second" in ui.format_latency_line(450)

    def test_multi_second(self):
        assert "2.5 seconds" in ui.format_latency_line(2500)


class TestPresetCaption:
    def test_all_preset_keys_have_captions(self):
        presets = [
            "Balanced & General Purpose (Recommended)",
            "High-Precision Flagship (Quality Benchmark)",
            "Deep Mathematical & Logical Derivation",
            "Specialized Code & Architecture Engineering",
            "Ultra-Fast Throughput (Sub-Second Latency)",
            "Cost-Constrained / Free Tier (Budget Optimization)",
        ]
        for p in presets:
            cap = ui.preset_caption(p)
            assert cap, f"missing caption for {p}"

    def test_unknown_preset_returns_empty(self):
        assert ui.preset_caption("nope") == ""
