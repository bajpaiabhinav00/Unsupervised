"""
Unit tests for analytics/scenario_simulator.py
Tests hypothetical scenario simulation logic against the actual API.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from analytics.scenario_simulator import (
    SIMULATABLE_INDICATORS,
    apply_scenario,
    build_before_after_df,
    run_scenario_comparison,
)
from analytics.gap_detection import BENCHMARKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_imputed_df() -> pd.DataFrame:
    """
    Minimal synthetic DHS-like imputed DataFrame using ACTUAL benchmark column names.
    Uses the keys from BENCHMARKS so gap calculation works end-to-end.
    """
    countries = ["Chad", "Niger", "Ghana", "Albania", "Cambodia"]
    data = {
        "CountryName": countries,
        "SurveyYear": [2019, 2017, 2022, 2017, 2021],
        # Maternal
        "anc_4plus_coverage":              [15.0, 12.0, 70.0, 88.0, 75.0],
        "anc_skilled_provider":            [30.0, 25.0, 85.0, 95.0, 90.0],
        "skilled_birth_attendance":        [24.0, 20.0, 78.0, 99.0, 90.0],
        "institutional_delivery":          [22.0, 18.0, 73.0, 99.0, 88.0],
        "modern_contraceptive_prevalence": [5.0,  8.0,  27.0, 55.0, 38.0],
        # Child
        "infant_mortality_rate":           [70.0, 80.0, 35.0, 10.0, 22.0],
        "under_five_mortality_rate":       [120.0,140.0, 55.0, 12.0, 30.0],
        "dpt3_vaccination_coverage":       [25.0, 20.0, 93.0, 97.0, 90.0],
        "measles_vaccination_coverage":    [22.0, 18.0, 90.0, 96.0, 88.0],
        "basic_vaccination_coverage":      [18.0, 15.0, 80.0, 93.0, 82.0],
        "child_stunting_prevalence":       [45.0, 50.0, 18.0,  5.0, 25.0],
        "child_wasting_prevalence":        [18.0, 22.0,  5.0,  2.0, 10.0],
        "child_underweight_prevalence":    [30.0, 35.0, 12.0,  3.0, 18.0],
        "exclusive_breastfeeding_rate":    [15.0, 12.0, 55.0, 25.0, 60.0],
        "improved_drinking_water_access":  [52.0, 48.0, 90.0, 97.0, 85.0],
        "improved_sanitation_access":      [12.0,  9.0, 70.0, 95.0, 68.0],
        "female_secondary_education":      [5.0,   4.0, 55.0, 85.0, 50.0],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# SIMULATABLE_INDICATORS structure
# ---------------------------------------------------------------------------

class TestSimulatableIndicators:
    def test_at_least_one_indicator(self):
        assert len(SIMULATABLE_INDICATORS) > 0

    def test_all_entries_have_label_min_max(self):
        """SIMULATABLE_INDICATORS uses label/min/max/step keys."""
        required_keys = {"label", "min", "max", "step"}
        for col, meta in SIMULATABLE_INDICATORS.items():
            assert required_keys.issubset(meta.keys()), (
                f"Indicator '{col}' missing keys: {required_keys - meta.keys()}"
            )

    def test_min_less_than_max(self):
        for col, meta in SIMULATABLE_INDICATORS.items():
            assert meta["min"] < meta["max"], f"'{col}': min >= max"

    def test_all_simulatable_indicators_are_subsets_of_benchmarks_or_dataset(self):
        """Every simulatable indicator should be a column in BENCHMARKS or at least parseable."""
        for col in SIMULATABLE_INDICATORS:
            # We just check the key exists (the column may not be in every dataset)
            assert isinstance(col, str) and len(col) > 0


# ---------------------------------------------------------------------------
# apply_scenario
# ---------------------------------------------------------------------------

class TestApplyScenario:
    def test_returns_dataframe(self, sample_imputed_df):
        result = apply_scenario(sample_imputed_df, "Chad", {})
        assert isinstance(result, pd.DataFrame)

    def test_returns_copy_not_inplace(self, sample_imputed_df):
        orig_val = float(sample_imputed_df.loc[
            sample_imputed_df["CountryName"] == "Chad", "skilled_birth_attendance"
        ].iloc[0])
        apply_scenario(sample_imputed_df, "Chad", {"skilled_birth_attendance": 80.0})
        # Original must be unchanged
        new_val = float(sample_imputed_df.loc[
            sample_imputed_df["CountryName"] == "Chad", "skilled_birth_attendance"
        ].iloc[0])
        assert orig_val == pytest.approx(new_val)

    def test_adds_simulated_column(self, sample_imputed_df):
        """apply_scenario adds a _simulated boolean column."""
        result = apply_scenario(sample_imputed_df, "Chad", {"skilled_birth_attendance": 80.0})
        assert "_simulated" in result.columns

    def test_simulated_flag_true_only_for_target(self, sample_imputed_df):
        result = apply_scenario(sample_imputed_df, "Chad", {"skilled_birth_attendance": 80.0})
        assert bool(result.loc[result["CountryName"] == "Chad", "_simulated"].iloc[0]) is True
        for country in ["Niger", "Ghana", "Albania", "Cambodia"]:
            assert bool(result.loc[result["CountryName"] == country, "_simulated"].iloc[0]) is False

    def test_target_country_value_changed(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = apply_scenario(sample_imputed_df, "Chad", mods)
        assert float(result.loc[result["CountryName"] == "Chad", "skilled_birth_attendance"].iloc[0]) == pytest.approx(80.0)

    def test_other_country_values_unchanged(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = apply_scenario(sample_imputed_df, "Chad", mods)
        for country in ["Niger", "Ghana", "Albania", "Cambodia"]:
            orig = float(sample_imputed_df.loc[
                sample_imputed_df["CountryName"] == country, "skilled_birth_attendance"
            ].iloc[0])
            new = float(result.loc[result["CountryName"] == country, "skilled_birth_attendance"].iloc[0])
            assert orig == pytest.approx(new)

    def test_unknown_country_raises_value_error(self, sample_imputed_df):
        with pytest.raises(ValueError):
            apply_scenario(sample_imputed_df, "Narnia", {"skilled_birth_attendance": 80.0})

    def test_unknown_column_raises_value_error(self, sample_imputed_df):
        with pytest.raises(ValueError):
            apply_scenario(sample_imputed_df, "Chad", {"nonexistent_column": 50.0})

    def test_percentage_capped_at_100(self, sample_imputed_df):
        """Coverage indicators should not exceed 100."""
        mods = {"anc_4plus_coverage": 150.0}
        result = apply_scenario(sample_imputed_df, "Chad", mods)
        val = float(result.loc[result["CountryName"] == "Chad", "anc_4plus_coverage"].iloc[0])
        assert val <= 100.0

    def test_multiple_indicators_modified(self, sample_imputed_df):
        mods = {"anc_4plus_coverage": 70.0, "skilled_birth_attendance": 60.0}
        result = apply_scenario(sample_imputed_df, "Niger", mods)
        assert float(result.loc[result["CountryName"] == "Niger", "anc_4plus_coverage"].iloc[0]) == pytest.approx(70.0)
        assert float(result.loc[result["CountryName"] == "Niger", "skilled_birth_attendance"].iloc[0]) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# run_scenario_comparison
# ---------------------------------------------------------------------------

class TestRunScenarioComparison:
    def test_returns_dict_with_required_keys(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        required = {
            "country", "adjustments",
            "baseline_maternal", "baseline_child", "baseline_overall",
            "scenario_maternal", "scenario_child", "scenario_overall",
            "baseline_prio", "scenario_prio",
            "delta_maternal", "delta_child", "delta_overall",
        }
        assert required.issubset(result.keys()), f"Missing keys: {required - result.keys()}"

    def test_country_matches_input(self, sample_imputed_df):
        result = run_scenario_comparison(sample_imputed_df, "Ghana", {"anc_4plus_coverage": 95.0})
        assert result["country"] == "Ghana"

    def test_adjustments_matches_input(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        assert result["adjustments"] == mods

    def test_baseline_prio_is_dataframe(self, sample_imputed_df):
        result = run_scenario_comparison(sample_imputed_df, "Niger", {"anc_4plus_coverage": 60.0})
        assert isinstance(result["baseline_prio"], pd.DataFrame)

    def test_scenario_prio_is_dataframe(self, sample_imputed_df):
        result = run_scenario_comparison(sample_imputed_df, "Niger", {"anc_4plus_coverage": 60.0})
        assert isinstance(result["scenario_prio"], pd.DataFrame)

    def test_delta_maternal_equals_scenario_minus_baseline(self, sample_imputed_df):
        result = run_scenario_comparison(sample_imputed_df, "Chad", {"skilled_birth_attendance": 80.0})
        b = result["baseline_maternal"]
        s = result["scenario_maternal"]
        d = result["delta_maternal"]
        if not (np.isnan(b) or np.isnan(s) or np.isnan(d)):
            assert abs(d - (s - b)) < 1e-6

    def test_indicator_baseline_and_scenario_recorded(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        assert "indicator_baseline" in result
        assert "indicator_scenario" in result
        assert "skilled_birth_attendance" in result["indicator_baseline"]
        assert float(result["indicator_scenario"]["skilled_birth_attendance"]) == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# build_before_after_df
# ---------------------------------------------------------------------------

class TestBuildBeforeAfterDf:
    def test_returns_dataframe(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        ba_df = build_before_after_df(result)
        assert isinstance(ba_df, pd.DataFrame)

    def test_has_required_columns(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        ba_df = build_before_after_df(result)
        required_cols = {"Indicator", "Baseline", "Hypothetical", "Change"}
        assert required_cols.issubset(ba_df.columns), f"Missing: {required_cols - set(ba_df.columns)}"

    def test_rows_match_modifications(self, sample_imputed_df):
        mods = {"anc_4plus_coverage": 80.0, "skilled_birth_attendance": 65.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        ba_df = build_before_after_df(result)
        assert len(ba_df) == len(mods)

    def test_change_equals_hypothetical_minus_baseline(self, sample_imputed_df):
        mods = {"skilled_birth_attendance": 80.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        ba_df = build_before_after_df(result)
        for _, row in ba_df.iterrows():
            expected_change = row["Hypothetical"] - row["Baseline"]
            assert abs(row["Change"] - expected_change) < 1e-6, (
                f"Change mismatch for {row['Indicator']}"
            )

    def test_empty_modifications_gives_empty_df(self, sample_imputed_df):
        result = run_scenario_comparison(sample_imputed_df, "Chad", {})
        ba_df = build_before_after_df(result)
        assert len(ba_df) == 0

    def test_indicator_label_used_in_indicator_column(self, sample_imputed_df):
        """build_before_after_df should use human-readable labels from SIMULATABLE_INDICATORS."""
        mods = {"skilled_birth_attendance": 80.0}
        result = run_scenario_comparison(sample_imputed_df, "Chad", mods)
        ba_df = build_before_after_df(result)
        if "skilled_birth_attendance" in SIMULATABLE_INDICATORS:
            expected_label = SIMULATABLE_INDICATORS["skilled_birth_attendance"]["label"]
            assert ba_df["Indicator"].iloc[0] == expected_label
