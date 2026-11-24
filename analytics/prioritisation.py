"""
Resource Prioritisation Engine.

Provides configurable priority scoring, ranking, tier categorisation, and
scenario / sensitivity analysis for geographical regions based on their
computed maternal and child healthcare gap indices.

DISCLAIMER:
  Priority categories are descriptive analytical tiers derived from relative
  gap scores within the analytical sample.  They are NOT official government
  or WHO classifications.  Financial allocation or clinical intervention
  decisions must not be based solely on this module's output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default scoring weights (must sum to 1.0)
# Population impact is included only when population data is available.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS_NO_POPULATION: Dict[str, float] = {
    "maternal_gap": 0.5,
    "child_gap": 0.5,
}
DEFAULT_WEIGHTS_WITH_POPULATION: Dict[str, float] = {
    "maternal_gap": 0.4,
    "child_gap": 0.4,
    "population_impact": 0.2,
}

TIER_LABELS = {
    "high": "Higher Relative Gap",
    "moderate": "Moderate Relative Gap",
    "low": "Lower Relative Gap",
}


def _validate_weights(weights: Dict[str, float], tolerance: float = 1e-6) -> None:
    """Raise ValueError if any weight is negative or if they don't sum to ~1.0."""
    for k, v in weights.items():
        if v < 0:
            raise ValueError(f"Weight for '{k}' is negative ({v}). All weights must be ≥ 0.")
    total = sum(weights.values())
    if abs(total - 1.0) > tolerance:
        raise ValueError(
            f"Weights must sum to 1.0 (sum={total:.6f}). "
            "Re-normalise the weights before passing them to this function."
        )


def _renormalise(weights: Dict[str, float]) -> Dict[str, float]:
    """Renormalise a weight dict to sum to 1.0 (in-place safe)."""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("All weights are zero; cannot renormalise.")
    return {k: v / total for k, v in weights.items()}


def compute_priority_scores(
    vulnerability_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    include_population: bool = False,
    population_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute a 0–100 priority score for each region.

    Priority_Score = 100 * Σ (w_i * normalised_component_i)

    Components:
      - maternal_gap   : Maternal Healthcare Gap Index (0–1)
      - child_gap      : Child Healthcare Gap Index (0–1)
      - population_impact (optional): log-normalised population size

    Args:
        vulnerability_df:  Output of ``build_vulnerability_dataframe()``.
                           Required columns: maternal_gap_index, child_gap_index.
        weights:           Component weights.  If None, defaults are used based on
                           ``include_population``.
        include_population: If True, include population impact in the score.
        population_series:  pd.Series indexed by CountryName with population values.
                            Required when include_population=True.

    Returns:
        DataFrame with columns: CountryName, maternal_gap_index, child_gap_index,
        [population_impact,] priority_score, priority_rank, [SurveyYear, ...]
    """
    required = {"maternal_gap_index", "child_gap_index"}
    missing = required - set(vulnerability_df.columns)
    if missing:
        raise ValueError(f"vulnerability_df is missing required columns: {missing}")

    if weights is None:
        if include_population:
            weights = DEFAULT_WEIGHTS_WITH_POPULATION.copy()
        else:
            weights = DEFAULT_WEIGHTS_NO_POPULATION.copy()
    else:
        weights = dict(weights)  # make a copy

    # If population is not requested, drop population_impact from weights
    if not include_population and "population_impact" in weights:
        del weights["population_impact"]
        weights = _renormalise(weights)

    _validate_weights(weights)

    result = vulnerability_df.copy()

    # Population component
    if include_population:
        if population_series is None:
            logger.warning(
                "include_population=True but no population_series provided. "
                "Excluding population component and renormalising weights."
            )
            if "population_impact" in weights:
                del weights["population_impact"]
            weights = _renormalise(weights)
        else:
            # Log-normalise population
            pop = population_series.reindex(result["CountryName"])
            pop_log = np.log1p(pop.values.astype(float))
            pop_norm = (pop_log - pop_log.min()) / (pop_log.max() - pop_log.min() + 1e-9)
            result["population_impact"] = pop_norm

    # Build priority score
    score = pd.Series(0.0, index=result.index)
    for component, wt in weights.items():
        col_map = {
            "maternal_gap": "maternal_gap_index",
            "child_gap": "child_gap_index",
            "population_impact": "population_impact",
        }
        col = col_map.get(component, component)
        if col in result.columns:
            score += wt * result[col].fillna(0.0)
        else:
            logger.warning(f"Weight component '{component}' not found in DataFrame. Skipped.")

    result["priority_score"] = (score * 100).round(2)
    result["priority_rank"] = result["priority_score"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("priority_score", ascending=False).reset_index(drop=True)

    logger.info(
        f"Priority scoring: {len(result)} regions ranked. "
        f"Weights: {weights}. Population included: {include_population}."
    )
    return result


def categorise_priorities(
    scores: pd.Series,
    n_tiers: int = 3,
    tier_labels: Optional[Dict[str, str]] = None,
) -> pd.Series:
    """
    Assign tier labels to priority scores using quantile-based thresholds.

    For n_tiers=3:
      - Top third → "Higher Relative Gap"
      - Middle third → "Moderate Relative Gap"
      - Bottom third → "Lower Relative Gap"

    Args:
        scores:      Priority score Series.
        n_tiers:     Number of tiers (2, 3, or 4).
        tier_labels: Custom label dict for 'high', 'moderate', 'low', 'very_low'.

    Returns:
        pd.Series of tier label strings.
    """
    if n_tiers not in {2, 3, 4}:
        raise ValueError("n_tiers must be 2, 3, or 4.")

    tl = tier_labels or TIER_LABELS
    quantile_step = 1.0 / n_tiers
    quantiles = [scores.quantile(1.0 - quantile_step * i) for i in range(n_tiers)]

    def assign(val: float) -> str:
        if pd.isna(val):
            return "Unknown"
        if n_tiers == 2:
            return tl["high"] if val >= quantiles[1] else tl["low"]
        elif n_tiers == 3:
            if val >= quantiles[1]:
                return tl["high"]
            elif val >= quantiles[2]:
                return tl.get("moderate", "Moderate Relative Gap")
            else:
                return tl["low"]
        else:  # 4 tiers
            if val >= quantiles[1]:
                return tl["high"]
            elif val >= quantiles[2]:
                return tl.get("moderate", "Moderate Relative Gap")
            elif val >= quantiles[3]:
                return tl.get("low", "Lower Relative Gap")
            else:
                return tl.get("very_low", "Lowest Relative Gap")

    return scores.map(assign)


def run_scenario_analysis(
    vulnerability_df: pd.DataFrame,
    scenarios: List[Dict],
) -> pd.DataFrame:
    """
    Run multiple weighting scenarios and return a comparison DataFrame.

    Each scenario dict must have:
      - name (str): scenario label
      - maternal_gap (float): weight for maternal gap
      - child_gap (float): weight for child gap
      - population_impact (float, optional): weight for population
      - include_population (bool, optional)

    Returns:
        DataFrame with CountryName and one priority_rank column per scenario.
    """
    comparison_frames: Dict[str, pd.Series] = {}

    for scenario in scenarios:
        name = scenario.get("name", "unnamed")
        include_pop = scenario.get("include_population", False)
        wt_dict = {k: v for k, v in scenario.items() if k not in {"name", "include_population"}}
        wt_dict = _renormalise({k: wt_dict[k] for k in wt_dict if isinstance(wt_dict[k], (int, float))})

        try:
            ranked = compute_priority_scores(
                vulnerability_df,
                weights=wt_dict,
                include_population=include_pop,
            )
            comparison_frames[name] = ranked.set_index("CountryName")["priority_rank"]
        except Exception as e:
            logger.error(f"Scenario '{name}' failed: {e}")

    if not comparison_frames:
        raise RuntimeError("All scenarios failed. Check scenario definitions.")

    comparison = pd.DataFrame(comparison_frames)
    # Rank stability: max minus min rank change
    comparison["rank_stability"] = comparison.max(axis=1) - comparison.min(axis=1)
    return comparison.reset_index()


def compute_weight_sensitivity(
    vulnerability_df: pd.DataFrame,
    n_samples: int = 200,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Monte Carlo sensitivity analysis: randomly sample weight vectors and
    compute the resulting rank distribution for each country.

    Returns:
        DataFrame indexed by CountryName with columns:
        rank_mean, rank_std, rank_min, rank_max, rank_iqr.
    """
    rng = np.random.default_rng(seed)
    rank_records: List[np.ndarray] = []
    countries = vulnerability_df["CountryName"].tolist()

    for _ in range(n_samples):
        raw = rng.dirichlet([1.0, 1.0])  # 2-component Dirichlet
        wt = {"maternal_gap": float(raw[0]), "child_gap": float(raw[1])}
        scored = compute_priority_scores(vulnerability_df, weights=wt, include_population=False)
        rank_records.append(scored.set_index("CountryName")["priority_rank"].reindex(countries).values)

    rank_array = np.array(rank_records)  # shape (n_samples, n_countries)

    result = pd.DataFrame({
        "CountryName": countries,
        "rank_mean": rank_array.mean(axis=0),
        "rank_std": rank_array.std(axis=0),
        "rank_min": rank_array.min(axis=0),
        "rank_max": rank_array.max(axis=0),
        "rank_iqr": np.percentile(rank_array, 75, axis=0) - np.percentile(rank_array, 25, axis=0),
    }).sort_values("rank_mean").reset_index(drop=True)

    return result


def build_priority_report(
    vulnerability_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    n_tiers: int = 3,
) -> pd.DataFrame:
    """
    Full priority report: scores, ranks, tier labels, and all gap indices.

    Returns a DataFrame ready for display or export.
    """
    scored = compute_priority_scores(vulnerability_df, weights=weights, include_population=False)
    scored["priority_tier"] = categorise_priorities(scored["priority_score"], n_tiers=n_tiers)
    # Round indices for readability
    for col in ["maternal_gap_index", "child_gap_index", "overall_gap_index"]:
        if col in scored.columns:
            scored[col] = scored[col].round(4)
    return scored


PREDEFINED_SCENARIOS = [
    {
        "name": "Equal Weight (Baseline)",
        "maternal_gap": 0.5,
        "child_gap": 0.5,
        "include_population": False,
    },
    {
        "name": "Maternal Priority",
        "maternal_gap": 0.7,
        "child_gap": 0.3,
        "include_population": False,
    },
    {
        "name": "Child Priority",
        "maternal_gap": 0.3,
        "child_gap": 0.7,
        "include_population": False,
    },
    {
        "name": "Maternal Only",
        "maternal_gap": 1.0,
        "child_gap": 0.0,
        "include_population": False,
    },
    {
        "name": "Child Only",
        "maternal_gap": 0.0,
        "child_gap": 1.0,
        "include_population": False,
    },
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset
    from analytics.gap_detection import calculate_region_gaps
    from analytics.vulnerability import build_vulnerability_dataframe

    df, _ = get_latest_cross_sectional_dataset(min_year=2000)
    gaps = calculate_region_gaps(df)
    vuln = build_vulnerability_dataframe(gaps, alpha=0.5)

    report = build_priority_report(vuln)
    print(f"Priority report: {report.shape}")
    print(report[["CountryName", "priority_score", "priority_rank", "priority_tier"]].head(15))

    scenario_df = run_scenario_analysis(vuln, PREDEFINED_SCENARIOS)
    print(f"\nScenario analysis: {scenario_df.shape}")
    print(scenario_df.head(10))

    sens = compute_weight_sensitivity(vuln, n_samples=50)
    print(f"\nSensitivity: {sens.shape}")
    print(sens[["CountryName", "rank_mean", "rank_std"]].head(10))
