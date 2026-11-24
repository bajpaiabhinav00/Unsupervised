"""
Extended pytest suite for the Analytics modules:
  - gap_detection.py
  - vulnerability.py
  - cluster_intelligence.py
  - prioritisation.py

Run with:
    python3 -m pytest tests/test_analytics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analytics.gap_detection import (
    BENCHMARKS,
    MATERNAL_INDICATORS,
    CHILD_INDICATORS,
    calculate_indicator_gap,
    calculate_region_gaps,
    calculate_maternal_gap_summary,
    calculate_child_gap_summary,
    generate_gap_heatmap_data,
    get_top_gap_indicators,
    get_benchmark_table,
)
from analytics.vulnerability import (
    compute_maternal_gap_index,
    compute_child_gap_index,
    compute_overall_index,
    build_vulnerability_dataframe,
    _normalise_weights,
)
from analytics.cluster_intelligence import (
    compute_cluster_profiles,
    generate_cluster_descriptions,
    generate_cluster_intelligence_report,
    compute_z_score_profiles,
    PROFILE_INDICATORS,
)
from analytics.prioritisation import (
    compute_priority_scores,
    categorise_priorities,
    run_scenario_analysis,
    compute_weight_sensitivity,
    build_priority_report,
    PREDEFINED_SCENARIOS,
    _validate_weights,
    _renormalise,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df():
    """Small representative DataFrame with 10 countries."""
    return pd.DataFrame({
        "CountryName": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon",
                        "Zeta", "Eta", "Theta", "Iota", "Kappa"],
        "DHS_CountryCode": ["AL", "BD", "CM", "EG", "GH",
                            "HN", "IN", "KE", "MG", "NG"],
        "SurveyYear": [2018, 2017, 2019, 2016, 2020,
                       2015, 2019, 2018, 2017, 2021],
        "anc_4plus_coverage": [80.0, 55.0, 40.0, 70.0, 90.0,
                               60.0, 50.0, 85.0, 30.0, 45.0],
        "anc_skilled_provider": [90.0, 60.0, 50.0, 80.0, 95.0,
                                 65.0, 55.0, 88.0, 35.0, 48.0],
        "skilled_birth_attendance": [95.0, 50.0, 40.0, 75.0, 90.0,
                                     55.0, 60.0, 80.0, 25.0, 42.0],
        "institutional_delivery": [95.0, 55.0, 42.0, 78.0, 92.0,
                                   58.0, 62.0, 82.0, 28.0, 44.0],
        "modern_contraceptive_prevalence": [40.0, 55.0, 20.0, 60.0, 35.0,
                                            45.0, 50.0, 55.0, 15.0, 25.0],
        "infant_mortality_rate": [10.0, 35.0, 55.0, 20.0, 40.0,
                                  30.0, 32.0, 25.0, 65.0, 50.0],
        "under_five_mortality_rate": [12.0, 50.0, 80.0, 28.0, 60.0,
                                      44.0, 46.0, 36.0, 95.0, 70.0],
        "dpt3_vaccination_coverage": [95.0, 70.0, 50.0, 85.0, 80.0,
                                      60.0, 75.0, 90.0, 40.0, 55.0],
        "measles_vaccination_coverage": [97.0, 65.0, 48.0, 83.0, 78.0,
                                         58.0, 72.0, 88.0, 38.0, 52.0],
        "basic_vaccination_coverage": [93.0, 62.0, 45.0, 80.0, 76.0,
                                       56.0, 70.0, 85.0, 36.0, 50.0],
        "child_stunting_prevalence": [5.0, 36.0, 42.0, 15.0, 20.0,
                                      25.0, 38.0, 18.0, 50.0, 40.0],
        "child_wasting_prevalence": [2.0, 15.0, 18.0, 6.0, 8.0,
                                     10.0, 20.0, 7.0, 22.0, 17.0],
        "child_underweight_prevalence": [3.0, 30.0, 35.0, 12.0, 15.0,
                                         18.0, 32.0, 14.0, 42.0, 35.0],
        "exclusive_breastfeeding_rate": [65.0, 55.0, 25.0, 30.0, 70.0,
                                         45.0, 50.0, 60.0, 20.0, 35.0],
        "improved_drinking_water_access": [98.0, 85.0, 60.0, 90.0, 80.0,
                                           70.0, 90.0, 75.0, 50.0, 65.0],
        "improved_sanitation_access": [95.0, 55.0, 35.0, 85.0, 70.0,
                                       60.0, 50.0, 65.0, 30.0, 40.0],
        "female_secondary_education": [70.0, 40.0, 20.0, 60.0, 50.0,
                                       35.0, 45.0, 55.0, 15.0, 25.0],
    })


@pytest.fixture(scope="module")
def gaps_df(sample_df):
    return calculate_region_gaps(sample_df)


@pytest.fixture(scope="module")
def vuln_df(gaps_df):
    return build_vulnerability_dataframe(gaps_df, alpha=0.5)


# ===========================================================================
# SECTION 1: Gap Detection
# ===========================================================================

class TestIndicatorGap:
    def test_higher_is_better_at_benchmark(self):
        """Gap should be 0 when observed == benchmark."""
        assert calculate_indicator_gap(80.0, 80.0, True) == pytest.approx(0.0)

    def test_higher_is_better_above_benchmark(self):
        """Gap should be 0 when observed > benchmark."""
        assert calculate_indicator_gap(90.0, 80.0, True) == pytest.approx(0.0)

    def test_higher_is_better_below_benchmark(self):
        """Gap should be 0.25 when observed is 25% below benchmark."""
        gap = calculate_indicator_gap(60.0, 80.0, True)
        assert gap == pytest.approx(0.25, rel=1e-4)

    def test_lower_is_better_at_benchmark(self):
        """Gap should be 0 when observed == benchmark."""
        assert calculate_indicator_gap(25.0, 25.0, False) == pytest.approx(0.0)

    def test_lower_is_better_below_benchmark(self):
        """Gap should be 0 when observed < benchmark."""
        assert calculate_indicator_gap(10.0, 25.0, False) == pytest.approx(0.0)

    def test_lower_is_better_above_benchmark(self):
        """Gap should be (observed - bm) / bm when observed > benchmark."""
        gap = calculate_indicator_gap(50.0, 25.0, False)
        assert gap == pytest.approx(1.0, rel=1e-4)

    def test_nan_observed(self):
        assert pd.isna(calculate_indicator_gap(float("nan"), 80.0, True))

    def test_nan_benchmark(self):
        assert pd.isna(calculate_indicator_gap(80.0, float("nan"), True))

    def test_zero_benchmark_returns_nan(self):
        assert pd.isna(calculate_indicator_gap(50.0, 0.0, True))

    def test_negative_observed_higher_is_better(self):
        """Negative (invalid) observed value for HIB indicator should produce a gap > 0."""
        gap = calculate_indicator_gap(-10.0, 80.0, True)
        assert gap > 0

    def test_gap_range(self):
        """Gap should always be in [0, inf) for lower-is-better when observed >> benchmark."""
        gap = calculate_indicator_gap(200.0, 25.0, False)
        assert gap > 0


class TestRegionGaps:
    def test_shape(self, sample_df, gaps_df):
        expected_gap_cols = sum(1 for ind in BENCHMARKS if ind in sample_df.columns)
        assert expected_gap_cols > 0
        assert len(gaps_df) == len(sample_df)

    def test_all_gap_values_non_negative(self, gaps_df):
        gap_cols = [c for c in gaps_df.columns if c.endswith("_gap")]
        vals = gaps_df[gap_cols].values.flatten()
        vals = vals[~np.isnan(vals)]
        assert (vals >= 0).all(), "All gap values must be non-negative."

    def test_country_names_preserved(self, sample_df, gaps_df):
        assert "CountryName" in gaps_df.columns
        assert set(gaps_df["CountryName"]) == set(sample_df["CountryName"])

    def test_missing_indicator_produces_nan_gap(self):
        """If an indicator column is missing, its gap column should be NaN."""
        mini = pd.DataFrame({
            "CountryName": ["X"],
            "anc_4plus_coverage": [70.0],
            # intentionally omit other indicators
        })
        g = calculate_region_gaps(mini)
        # Indicators not present in mini should have NaN gap
        assert g["skilled_birth_attendance_gap"].isna().all()

    def test_maternal_gap_summary_sorted(self, gaps_df):
        summary = calculate_maternal_gap_summary(gaps_df)
        assert "mean_maternal_gap" in summary.columns
        assert summary["mean_maternal_gap"].is_monotonic_decreasing

    def test_child_gap_summary_sorted(self, gaps_df):
        summary = calculate_child_gap_summary(gaps_df)
        assert "mean_child_gap" in summary.columns
        assert summary["mean_child_gap"].is_monotonic_decreasing

    def test_heatmap_data_no_meta_cols(self, gaps_df):
        hm = generate_gap_heatmap_data(gaps_df)
        assert "CountryName" not in hm.columns
        assert len(hm) > 0

    def test_top_gap_indicators_length(self, gaps_df):
        top = get_top_gap_indicators(gaps_df, n=5)
        assert len(top) == 5

    def test_benchmark_table_completeness(self):
        table = get_benchmark_table()
        assert len(table) == len(BENCHMARKS)
        assert "Reference Value" in table.columns
        assert "Direction" in table.columns

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame(columns=["CountryName"] + list(BENCHMARKS.keys()))
        g = calculate_region_gaps(empty)
        assert len(g) == 0


# ===========================================================================
# SECTION 2: Vulnerability Profiling
# ===========================================================================

class TestVulnerabilityIndexing:
    def test_mhgi_range(self, gaps_df):
        mhgi = compute_maternal_gap_index(gaps_df)
        valid = mhgi.dropna()
        assert (valid >= 0).all() and (valid <= 1.5).all()  # allow >1 for lower-is-better rescaling

    def test_chgi_range(self, gaps_df):
        chgi = compute_child_gap_index(gaps_df)
        valid = chgi.dropna()
        assert (valid >= 0).all()

    def test_overall_alpha_zero_equals_child(self, gaps_df):
        mhgi = compute_maternal_gap_index(gaps_df)
        chgi = compute_child_gap_index(gaps_df)
        overall_child = compute_overall_index(mhgi, chgi, alpha=0.0)
        pd.testing.assert_series_equal(overall_child, chgi.rename("overall_gap_index"), check_names=False)

    def test_overall_alpha_one_equals_maternal(self, gaps_df):
        mhgi = compute_maternal_gap_index(gaps_df)
        chgi = compute_child_gap_index(gaps_df)
        overall_maternal = compute_overall_index(mhgi, chgi, alpha=1.0)
        pd.testing.assert_series_equal(overall_maternal, mhgi.rename("overall_gap_index"), check_names=False)

    def test_invalid_alpha_raises(self, gaps_df):
        mhgi = compute_maternal_gap_index(gaps_df)
        chgi = compute_child_gap_index(gaps_df)
        with pytest.raises(ValueError, match="alpha"):
            compute_overall_index(mhgi, chgi, alpha=1.5)

    def test_vuln_df_sorted_descending(self, vuln_df):
        assert vuln_df["overall_gap_index"].is_monotonic_decreasing

    def test_vuln_df_has_required_columns(self, vuln_df):
        for col in ["CountryName", "maternal_gap_index", "child_gap_index", "overall_gap_index"]:
            assert col in vuln_df.columns, f"Missing column: {col}"

    def test_weight_normalisation(self):
        raw = {"a": 2.0, "b": 3.0, "c": 5.0}
        norm = _normalise_weights(raw, ["a", "b", "c"])
        assert abs(sum(norm.values()) - 1.0) < 1e-9

    def test_zero_weight_excluded(self):
        raw = {"a": 0.0, "b": 1.0, "c": 1.0}
        norm = _normalise_weights(raw, ["a", "b", "c"])
        assert "a" not in norm

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            _normalise_weights({"a": 0.0, "b": 0.0}, ["a", "b"])

    def test_missing_indicator_in_gaps(self):
        """Vulnerability should still compute when some indicators are missing."""
        mini_gaps = pd.DataFrame({
            "CountryName": ["X", "Y"],
            "anc_4plus_coverage_gap": [0.3, 0.1],
            "skilled_birth_attendance_gap": [0.4, 0.2],
        })
        mhgi = compute_maternal_gap_index(mini_gaps)
        assert len(mhgi) == 2
        assert not mhgi.isna().all()

    def test_single_country(self):
        """Single-row DataFrame should not raise errors."""
        single = pd.DataFrame({
            "CountryName": ["OnlyCountry"],
            "anc_4plus_coverage_gap": [0.5],
            "skilled_birth_attendance_gap": [0.3],
            "under_five_mortality_rate_gap": [0.8],
            "child_stunting_prevalence_gap": [0.4],
        })
        mhgi = compute_maternal_gap_index(single)
        chgi = compute_child_gap_index(single)
        assert len(mhgi) == 1
        assert len(chgi) == 1


# ===========================================================================
# SECTION 3: Cluster Intelligence
# ===========================================================================

class TestClusterIntelligence:
    @pytest.fixture(scope="class")
    def cluster_data(self, sample_df):
        """Provide sample_df + simple 3-cluster labels."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        return sample_df, labels

    def test_profile_shape(self, cluster_data, gaps_df):
        df, labels = cluster_data
        profile = compute_cluster_profiles(df, labels, gaps_df=gaps_df)
        assert len(profile) == 3  # 3 clusters

    def test_profile_counts_sum_to_n(self, cluster_data):
        df, labels = cluster_data
        profile = compute_cluster_profiles(df, labels)
        assert profile["n_regions"].sum() == len(df)

    def test_descriptions_are_non_empty(self, cluster_data, sample_df):
        df, labels = cluster_data
        profile = compute_cluster_profiles(df, labels)
        descriptions = generate_cluster_descriptions(profile, sample_df)
        for cid, desc in descriptions.items():
            assert isinstance(desc, str) and len(desc) > 20, f"Cluster {cid} description too short."

    def test_descriptions_contain_cluster_id(self, cluster_data, sample_df):
        df, labels = cluster_data
        profile = compute_cluster_profiles(df, labels)
        descriptions = generate_cluster_descriptions(profile, sample_df)
        for cid, desc in descriptions.items():
            assert str(cid) in desc

    def test_intelligence_report_keys(self, cluster_data, gaps_df, sample_df):
        df, labels = cluster_data
        report = generate_cluster_intelligence_report(df, labels, gaps_df=gaps_df)
        for key in ["profiles", "descriptions", "n_clusters", "cluster_sizes", "country_membership"]:
            assert key in report

    def test_z_score_profiles_shape(self, cluster_data, sample_df):
        df, labels = cluster_data
        z = compute_z_score_profiles(df, labels)
        assert len(z) == 3  # 3 clusters
        assert z.shape[1] > 0

    def test_all_countries_appear_in_membership(self, cluster_data):
        df, labels = cluster_data
        profile = compute_cluster_profiles(df, labels)
        if "countries" in profile.columns:
            all_countries = [c for sub in profile["countries"] for c in sub]
            assert set(all_countries) == set(df["CountryName"].tolist())


# ===========================================================================
# SECTION 4: Prioritisation Engine
# ===========================================================================

class TestPrioritisation:
    def test_priority_scores_range(self, vuln_df):
        scored = compute_priority_scores(vuln_df)
        assert (scored["priority_score"] >= 0).all()
        # Scores are scaled by 100 from indices; child gap indices can exceed 1
        # due to lower-is-better indicators (mortality) being far above benchmark,
        # so we only assert non-negative and finite.
        assert scored["priority_score"].isna().sum() == 0

    def test_priority_ranks_unique(self, vuln_df):
        scored = compute_priority_scores(vuln_df)
        # All ranks should be unique (no ties expected in test data)
        assert scored["priority_rank"].nunique() == len(scored)

    def test_priority_sorted_descending(self, vuln_df):
        scored = compute_priority_scores(vuln_df)
        assert scored["priority_score"].is_monotonic_decreasing

    def test_weight_sum_validation(self, vuln_df):
        bad_weights = {"maternal_gap": 0.6, "child_gap": 0.6}  # sum = 1.2
        with pytest.raises(ValueError, match="sum"):
            compute_priority_scores(vuln_df, weights=bad_weights)

    def test_negative_weight_validation(self, vuln_df):
        bad_weights = {"maternal_gap": -0.5, "child_gap": 1.5}
        with pytest.raises(ValueError, match="negative"):
            compute_priority_scores(vuln_df, weights=bad_weights)

    def test_categorise_three_tiers(self, vuln_df):
        scored = compute_priority_scores(vuln_df)
        tiers = categorise_priorities(scored["priority_score"], n_tiers=3)
        assert set(tiers.unique()).issubset({"Higher Relative Gap", "Moderate Relative Gap", "Lower Relative Gap"})

    def test_categorise_two_tiers(self, vuln_df):
        scored = compute_priority_scores(vuln_df)
        tiers = categorise_priorities(scored["priority_score"], n_tiers=2)
        assert set(tiers.unique()).issubset({"Higher Relative Gap", "Lower Relative Gap"})

    def test_invalid_n_tiers(self, vuln_df):
        scored = compute_priority_scores(vuln_df)
        with pytest.raises(ValueError):
            categorise_priorities(scored["priority_score"], n_tiers=5)

    def test_scenario_analysis_returns_all_scenarios(self, vuln_df):
        result = run_scenario_analysis(vuln_df, PREDEFINED_SCENARIOS)
        for s in PREDEFINED_SCENARIOS:
            assert s["name"] in result.columns

    def test_scenario_analysis_rank_stability_column(self, vuln_df):
        result = run_scenario_analysis(vuln_df, PREDEFINED_SCENARIOS)
        assert "rank_stability" in result.columns
        assert (result["rank_stability"] >= 0).all()

    def test_weight_sensitivity_shape(self, vuln_df):
        sens = compute_weight_sensitivity(vuln_df, n_samples=10)
        assert len(sens) == len(vuln_df)
        assert "rank_mean" in sens.columns
        assert "rank_std" in sens.columns

    def test_full_priority_report_tiers_assigned(self, vuln_df):
        report = build_priority_report(vuln_df)
        assert "priority_tier" in report.columns
        assert report["priority_tier"].notna().all()

    def test_no_population_excludes_component(self, vuln_df):
        """With include_population=False, there must be no population_impact column."""
        scored = compute_priority_scores(vuln_df, include_population=False)
        assert "population_impact" not in scored.columns

    def test_renormalise(self):
        norm = _renormalise({"a": 1.0, "b": 1.0, "c": 2.0})
        assert abs(sum(norm.values()) - 1.0) < 1e-9
        assert norm["c"] == pytest.approx(0.5)

    def test_validate_weights_passes_ok(self):
        _validate_weights({"a": 0.5, "b": 0.5})  # should not raise

    def test_validate_weights_fails_on_sum(self):
        with pytest.raises(ValueError):
            _validate_weights({"a": 0.3, "b": 0.3})

    def test_empty_vuln_df(self):
        empty = pd.DataFrame(columns=["CountryName", "maternal_gap_index",
                                       "child_gap_index", "overall_gap_index"])
        scored = compute_priority_scores(empty)
        assert len(scored) == 0


# ===========================================================================
# SECTION 5: Edge Cases
# ===========================================================================

class TestEdgeCases:
    def test_all_at_benchmark_gaps_all_zero(self):
        """If every indicator exactly meets its benchmark, all gaps should be 0."""
        at_bm = pd.DataFrame({
            "CountryName": ["Perfect"],
            **{ind: [spec["value"]] for ind, spec in BENCHMARKS.items()},
        })
        g = calculate_region_gaps(at_bm)
        gap_cols = [c for c in g.columns if c.endswith("_gap")]
        vals = g[gap_cols].values.flatten()
        vals = vals[~np.isnan(vals)]
        assert (vals == 0).all()

    def test_all_worst_case_gives_max_gaps(self):
        """Worst-case values should produce maximal (>0) gaps for all indicators."""
        worst = {}
        for ind, spec in BENCHMARKS.items():
            if spec["higher_is_better"]:
                worst[ind] = 0.0
            else:
                worst[ind] = spec["value"] * 10  # 10× above benchmark
        worst_df = pd.DataFrame({"CountryName": ["Worst"], **{k: [v] for k, v in worst.items()}})
        g = calculate_region_gaps(worst_df)
        gap_cols = [c for c in g.columns if c.endswith("_gap")]
        vals = g[gap_cols].values.flatten()
        vals = vals[~np.isnan(vals)]
        assert (vals > 0).all()

    def test_duplicate_countries_handled(self, sample_df, gaps_df):
        """Duplicate rows should not crash the pipeline."""
        dup_df = pd.concat([sample_df, sample_df]).reset_index(drop=True)
        g = calculate_region_gaps(dup_df)
        assert len(g) == len(dup_df)

    def test_impossible_percentage_above_100(self):
        """Values > 100% should not raise exceptions — gap calculation must be safe."""
        bad_df = pd.DataFrame({
            "CountryName": ["BadData"],
            "anc_4plus_coverage": [150.0],  # physically impossible
        })
        g = calculate_region_gaps(bad_df)
        # 150 > 80 benchmark → gap should be 0 (above benchmark)
        assert g["anc_4plus_coverage_gap"].iloc[0] == pytest.approx(0.0)

    def test_negative_mortality_treated_safely(self):
        """Negative mortality (data error) should not cause runtime errors."""
        bad_df = pd.DataFrame({
            "CountryName": ["BadData"],
            "under_five_mortality_rate": [-5.0],  # impossible
        })
        g = calculate_region_gaps(bad_df)
        # -5 < 25 benchmark → lower-is-better → gap = max(0, ...) = 0
        assert g["under_five_mortality_rate_gap"].iloc[0] == pytest.approx(0.0)
