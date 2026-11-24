"""
Regional Cluster Intelligence Engine.

Generates data-driven per-cluster profiles, descriptions, and comparison tables
for the Maternal & Child Health Gap Intelligence system.

All descriptions are auto-generated from actual statistics — no labels or
narratives are hardcoded.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analytics.gap_detection import BENCHMARKS, MATERNAL_INDICATORS, CHILD_INDICATORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Indicators to include in cluster profiles
PROFILE_INDICATORS = [
    "anc_4plus_coverage",
    "skilled_birth_attendance",
    "institutional_delivery",
    "modern_contraceptive_prevalence",
    "infant_mortality_rate",
    "under_five_mortality_rate",
    "dpt3_vaccination_coverage",
    "measles_vaccination_coverage",
    "basic_vaccination_coverage",
    "child_stunting_prevalence",
    "child_wasting_prevalence",
    "child_underweight_prevalence",
    "exclusive_breastfeeding_rate",
    "improved_drinking_water_access",
    "improved_sanitation_access",
    "female_secondary_education",
]


def _direction_text(indicator: str, cluster_mean: float, global_mean: float) -> str:
    """Returns 'above', 'below', or 'similar to' based on the numeric difference."""
    diff = cluster_mean - global_mean
    threshold = abs(global_mean) * 0.05  # 5 % relative threshold
    if diff > threshold:
        return "above"
    elif diff < -threshold:
        return "below"
    else:
        return "similar to"


def _qualitative_magnitude(relative_diff: float) -> str:
    """Returns a qualitative magnitude word for a relative difference."""
    abs_diff = abs(relative_diff)
    if abs_diff < 0.05:
        return "marginally"
    elif abs_diff < 0.15:
        return "moderately"
    elif abs_diff < 0.30:
        return "notably"
    else:
        return "substantially"


def compute_cluster_profiles(
    imputed_df: pd.DataFrame,
    labels: np.ndarray,
    gaps_df: Optional[pd.DataFrame] = None,
    indicators: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute mean and median indicator values per cluster.

    Args:
        imputed_df: Wide-format DataFrame with indicator columns (imputed, pre-standardisation).
        labels:     Array of cluster assignments (same length as imputed_df).
        gaps_df:    Optional gap DataFrame for per-cluster gap summaries.
        indicators: Subset of indicator columns to profile. Defaults to PROFILE_INDICATORS.

    Returns:
        DataFrame indexed by cluster label with mean columns and optional gap means.
    """
    ind_cols = [c for c in (indicators or PROFILE_INDICATORS) if c in imputed_df.columns]
    df = imputed_df.copy()
    df["_cluster"] = labels

    # Means and medians
    cluster_means = df.groupby("_cluster")[ind_cols].mean()
    cluster_medians = df.groupby("_cluster")[ind_cols].median()
    cluster_counts = df.groupby("_cluster").size().rename("n_regions")

    profile = cluster_means.copy()
    profile.index.name = "cluster"
    profile = profile.join(cluster_counts)

    # Country membership
    country_col = "CountryName" if "CountryName" in df.columns else None
    if country_col:
        membership = (
            df.groupby("_cluster")[country_col]
            .apply(lambda x: sorted(x.tolist()))
        )
        profile["countries"] = membership

    # Gap score summaries if gaps_df provided
    if gaps_df is not None:
        from analytics.gap_detection import MATERNAL_INDICATORS, CHILD_INDICATORS
        m_gap_cols = [f"{i}_gap" for i in MATERNAL_INDICATORS if f"{i}_gap" in gaps_df.columns]
        c_gap_cols = [f"{i}_gap" for i in CHILD_INDICATORS if f"{i}_gap" in gaps_df.columns]

        gap_aug = gaps_df.copy()
        gap_aug["_cluster"] = labels
        if m_gap_cols:
            profile["mean_maternal_gap"] = gap_aug.groupby("_cluster")[m_gap_cols].mean().mean(axis=1)
        if c_gap_cols:
            profile["mean_child_gap"] = gap_aug.groupby("_cluster")[c_gap_cols].mean().mean(axis=1)

    return profile


def generate_cluster_descriptions(
    profile: pd.DataFrame,
    imputed_df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
) -> Dict[int, str]:
    """
    Auto-generate data-driven textual descriptions for each cluster.

    Descriptions are built from actual statistics, not hardcoded text.
    Key indicators are highlighted where the cluster mean deviates ≥10% from
    the global sample mean.

    Args:
        profile:    Output of ``compute_cluster_profiles()``.
        imputed_df: Original imputed dataset (for global means).
        indicators: Indicators to consider for description generation.

    Returns:
        Dict mapping cluster_id → description string.
    """
    ind_cols = [c for c in (indicators or PROFILE_INDICATORS) if c in imputed_df.columns]
    global_means = imputed_df[ind_cols].mean()

    # Indicator direction (higher_is_better) lookup
    direction_map = {
        v["short_name"]: v["higher_is_better"]
        for v in config.INDICATORS_CONFIG.values()
    }

    descriptions: Dict[int, str] = {}

    for cluster_id in profile.index:
        row = profile.loc[cluster_id]
        n = int(row.get("n_regions", "?"))
        countries_list = row.get("countries", [])
        country_examples = ", ".join(countries_list[:4])
        if len(countries_list) > 4:
            country_examples += f", and {len(countries_list) - 4} others"

        # Identify the most distinctive indicators
        notable_above: List[str] = []  # cluster above global — positive for HIB, negative for LIB
        notable_below: List[str] = []  # cluster below global — negative for HIB, positive for LIB

        for ind in ind_cols:
            if ind not in row.index or pd.isna(row[ind]):
                continue
            g_mean = global_means[ind]
            c_mean = row[ind]
            if g_mean == 0:
                continue
            rel_diff = (c_mean - g_mean) / abs(g_mean)
            hib = direction_map.get(ind, True)

            if abs(rel_diff) < 0.08:
                continue  # Not notable

            label = config.SHORT_TO_LABEL.get(ind, ind.replace("_", " ").title())
            magnitude = _qualitative_magnitude(rel_diff)

            if hib:
                # Higher is better: above global = positive
                if rel_diff > 0:
                    notable_above.append(
                        f"{label} ({c_mean:.1f}%, {magnitude} above the sample mean of {g_mean:.1f}%)"
                    )
                else:
                    notable_below.append(
                        f"{label} ({c_mean:.1f}%, {magnitude} below the sample mean of {g_mean:.1f}%)"
                    )
            else:
                # Lower is better: below global = positive (better performance)
                unit = "per 1,000" if "mortality" in ind else "%"
                if rel_diff > 0:  # cluster is higher — worse
                    notable_below.append(
                        f"{label} ({c_mean:.1f}{unit}, {magnitude} higher than sample mean of {g_mean:.1f}{unit})"
                    )
                else:  # cluster is lower — better
                    notable_above.append(
                        f"{label} ({c_mean:.1f}{unit}, {magnitude} lower than sample mean of {g_mean:.1f}{unit})"
                    )

        # Build description
        parts = [
            f"Cluster {cluster_id} comprises {n} region(s) including {country_examples}."
        ]

        if "mean_maternal_gap" in row and not pd.isna(row["mean_maternal_gap"]):
            mgap = row["mean_maternal_gap"]
            parts.append(
                f"The mean maternal healthcare gap index is {mgap:.3f} "
                f"({'above' if mgap > 0.25 else 'below'} the analytical sample threshold of 0.25)."
            )
        if "mean_child_gap" in row and not pd.isna(row["mean_child_gap"]):
            cgap = row["mean_child_gap"]
            parts.append(
                f"The mean child healthcare gap index is {cgap:.3f}."
            )

        if notable_above:
            parts.append(
                "Relative strengths or lower-burden indicators in this cluster: "
                + "; ".join(notable_above[:3]) + "."
            )
        if notable_below:
            parts.append(
                "Areas with notable gaps relative to the full sample: "
                + "; ".join(notable_below[:3]) + "."
            )

        if not notable_above and not notable_below:
            parts.append(
                "This cluster shows indicator profiles broadly similar to the full analytical sample."
            )

        descriptions[cluster_id] = " ".join(parts)

    return descriptions


def generate_cluster_intelligence_report(
    imputed_df: pd.DataFrame,
    labels: np.ndarray,
    gaps_df: Optional[pd.DataFrame] = None,
    indicators: Optional[List[str]] = None,
) -> Dict:
    """
    Master function: computes cluster profiles, descriptions, and returns a
    structured intelligence report dictionary.

    Returns:
        {
          "profiles": pd.DataFrame,
          "descriptions": Dict[int, str],
          "n_clusters": int,
          "cluster_sizes": Dict[int, int],
          "country_membership": Dict[int, List[str]],
        }
    """
    profile = compute_cluster_profiles(imputed_df, labels, gaps_df=gaps_df, indicators=indicators)
    descriptions = generate_cluster_descriptions(profile, imputed_df, indicators=indicators)

    cluster_sizes = {int(k): int(v) for k, v in profile["n_regions"].items()}
    country_membership: Dict[int, List[str]] = {}
    if "countries" in profile.columns:
        country_membership = {
            int(k): list(v) for k, v in profile["countries"].items()
        }

    report = {
        "profiles": profile,
        "descriptions": descriptions,
        "n_clusters": len(profile),
        "cluster_sizes": cluster_sizes,
        "country_membership": country_membership,
    }
    logger.info(
        f"Cluster intelligence report: {len(profile)} clusters, "
        f"sizes: {cluster_sizes}"
    )
    return report


def compute_z_score_profiles(
    imputed_df: pd.DataFrame,
    labels: np.ndarray,
    indicators: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Returns cluster means expressed as Z-scores relative to the global sample.
    Useful for radar charts and heatmaps showing which clusters deviate from the mean.
    """
    ind_cols = [c for c in (indicators or PROFILE_INDICATORS) if c in imputed_df.columns]
    df = imputed_df.copy()
    df["_cluster"] = labels

    global_mean = df[ind_cols].mean()
    global_std = df[ind_cols].std().replace(0, np.nan)

    cluster_means = df.groupby("_cluster")[ind_cols].mean()
    z_profiles = (cluster_means - global_mean) / global_std
    z_profiles.index.name = "cluster"
    return z_profiles


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.cluster import KMeans
    from src.data_loader import get_latest_cross_sectional_dataset
    from src.preprocessing import HealthDataPreprocessor
    from analytics.gap_detection import calculate_region_gaps

    df, _ = get_latest_cross_sectional_dataset(min_year=2000)
    prep = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
    _, imputed, standardized = prep.fit_transform_pipeline(df, save_artifacts=False)

    X = standardized[config.FEATURE_COLUMNS].values
    km = KMeans(n_clusters=4, random_state=config.RANDOM_SEED, n_init=config.N_INIT)
    labels = km.fit_predict(X)

    gaps = calculate_region_gaps(imputed)

    report = generate_cluster_intelligence_report(imputed, labels, gaps_df=gaps)
    print(f"Clusters: {report['n_clusters']}, sizes: {report['cluster_sizes']}")
    for cid, desc in report["descriptions"].items():
        print(f"\n--- Cluster {cid} ---")
        print(desc)

    z_profiles = compute_z_score_profiles(imputed, labels)
    print("\nZ-score profiles (first 5 indicators):")
    print(z_profiles.iloc[:, :5])
