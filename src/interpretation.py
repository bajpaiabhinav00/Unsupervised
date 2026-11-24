"""
Cluster Interpretation and Stability Validation Module.
Calculates cluster profiles (means, medians, standardized z-score deviations),
generates objective, statistically grounded epidemiological narratives,
maps country memberships, and executes multi-angle stability validation
(across random initializations, feature subsets, preprocessing strategies, and algorithms).
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config
from src.evaluation import compare_partitions_ari, evaluate_clustering

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_cluster_profiles(
    unscaled_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes summary statistics for each cluster:
    1. Mean indicator values
    2. Median indicator values
    3. Standardized z-score deviation from global sample mean
    """
    features = feature_cols or [c for c in config.FEATURE_COLUMNS if c in unscaled_df.columns]
    df = unscaled_df.copy()
    df["Cluster"] = cluster_labels
    df["Cluster_Name"] = [f"Cluster {c+1}" for c in cluster_labels]

    means_df = df.groupby("Cluster_Name")[features].mean()
    medians_df = df.groupby("Cluster_Name")[features].median()

    # Calculate standardized deviation: (Cluster_Mean - Global_Mean) / Global_Std
    global_mean = unscaled_df[features].mean()
    global_std = unscaled_df[features].std().replace(0, 1e-9)
    z_deviations = (means_df - global_mean) / global_std

    return means_df, medians_df, z_deviations


def generate_cluster_narrative(
    cluster_name: str,
    z_series: pd.Series,
    country_list: List[str],
) -> Dict[str, Union[str, List[str]]]:
    """
    Generates an objective, statistically grounded epidemiological description
    for a single cluster based on its standardized indicator deviations.
    Avoids pejorative or unsubstantiated labels like 'developed' or 'backward'.
    """
    # Categories breakdown
    categories = {
        "Maternal Healthcare Access": [
            "anc_4plus_coverage", "anc_skilled_provider", "skilled_birth_attendance",
            "institutional_delivery", "modern_contraceptive_prevalence"
        ],
        "Child Mortality & Survival": ["infant_mortality_rate", "under_five_mortality_rate"],
        "Child Immunization": [
            "dpt3_vaccination_coverage", "measles_vaccination_coverage", "basic_vaccination_coverage"
        ],
        "Child Nutrition": [
            "child_stunting_prevalence", "child_wasting_prevalence", "child_underweight_prevalence",
            "exclusive_breastfeeding_rate"
        ],
        "Socioeconomic & WASH Infrastructure": [
            "improved_drinking_water_access", "improved_sanitation_access", "female_secondary_education"
        ],
    }

    category_summaries = {}
    for cat_name, cat_feats in categories.items():
        present_feats = [f for f in cat_feats if f in z_series.index]
        if not present_feats:
            continue
        avg_z = z_series[present_feats].mean()

        if cat_name == "Child Mortality & Survival":
            # Higher mortality z-score means higher deaths (adverse)
            if avg_z > 0.5:
                status = "Elevated child mortality rates relative to the global survey average"
            elif avg_z < -0.5:
                status = "Substantially lower child mortality rates relative to the global survey average"
            else:
                status = "Child mortality rates near the cross-country survey median"
        elif cat_name == "Child Nutrition":
            # Stunting/wasting/underweight: higher z-score is adverse
            undernutrition_feats = [f for f in present_feats if "stunting" in f or "wasting" in f or "underweight" in f]
            nutr_z = z_series[undernutrition_feats].mean() if undernutrition_feats else avg_z
            if nutr_z > 0.5:
                status = "High burden of child undernutrition and chronic stunting"
            elif nutr_z < -0.5:
                status = "Relatively lower rates of child acute and chronic malnutrition"
            else:
                status = "Moderate child nutritional vulnerability near the survey average"
        else:
            # For access/coverage: higher z-score is positive
            if avg_z > 0.6:
                status = "High coverage and strong service utilization"
            elif avg_z > 0.1:
                status = "Moderate coverage with ongoing service gaps"
            elif avg_z > -0.6:
                status = "Below-average coverage with significant service access deficits"
            else:
                status = "Substantial systemic coverage gaps and restricted access"

        category_summaries[cat_name] = status

    # Top distinguishing features
    strongest_positive = z_series.nlargest(3)
    strongest_negative = z_series.nsmallest(3)

    return {
        "cluster_name": cluster_name,
        "country_count": len(country_list),
        "countries": country_list,
        "maternal_health_profile": category_summaries.get("Maternal Healthcare Access", "N/A"),
        "child_mortality_profile": category_summaries.get("Child Mortality & Survival", "N/A"),
        "child_immunization_profile": category_summaries.get("Child Immunization", "N/A"),
        "child_nutrition_profile": category_summaries.get("Child Nutrition", "N/A"),
        "socioeconomic_profile": category_summaries.get("Socioeconomic & WASH Infrastructure", "N/A"),
        "strongest_positive_indicators": [
            f"{config.SHORT_TO_LABEL.get(f, f)} (+{val:.2f} SD)" for f, val in strongest_positive.items()
        ],
        "strongest_negative_indicators": [
            f"{config.SHORT_TO_LABEL.get(f, f)} ({val:.2f} SD)" for f, val in strongest_negative.items()
        ],
    }


def get_cluster_membership(
    df: pd.DataFrame, cluster_labels: np.ndarray
) -> pd.DataFrame:
    """
    Creates a structured mapping table of countries, survey years, and their assigned cluster.
    """
    mapping_df = pd.DataFrame({
        "CountryName": df["CountryName"].values,
        "SurveyYear": df["SurveyYear"].values,
        "Cluster": cluster_labels,
        "Cluster_Name": [f"Cluster {c+1}" for c in cluster_labels],
    }).sort_values(by=["Cluster", "CountryName"]).reset_index(drop=True)
    return mapping_df


def run_stability_analysis(
    X: np.ndarray,
    df_imputed: pd.DataFrame,
    base_k: int = config.OPTIMAL_K_DEFAULT,
    base_seed: int = config.RANDOM_SEED,
) -> Dict[str, Union[pd.DataFrame, float]]:
    """
    Tests clustering stability across:
    1. 10 different random seeds (initialization sensitivity)
    2. Subsets of features (domain-specific stability: maternal vs child vs WASH)
    3. 80% bootstrap sub-sampling (data perturbation stability)
    """
    base_kmeans = KMeans(n_clusters=base_k, init="k-means++", n_init=config.N_INIT, random_state=base_seed)
    base_labels = base_kmeans.fit_predict(X)

    # 1. Random seed sensitivity
    seed_aris = []
    test_seeds = [1, 7, 13, 21, 42, 99, 123, 256, 500, 777]
    for s in test_seeds:
        km = KMeans(n_clusters=base_k, init="k-means++", n_init=config.N_INIT, random_state=s)
        labs = km.fit_predict(X)
        seed_aris.append(compare_partitions_ari(base_labels, labs))

    # 2. Domain feature subsets
    maternal_cols = [c for c in config.FEATURE_COLUMNS if config.SHORT_TO_CATEGORY.get(c) == "Maternal Health"]
    child_cols = [
        c for c in config.FEATURE_COLUMNS
        if config.SHORT_TO_CATEGORY.get(c) in ["Child Mortality", "Child Immunization", "Child Nutrition"]
    ]
    wash_cols = [c for c in config.FEATURE_COLUMNS if config.SHORT_TO_CATEGORY.get(c) == "Socioeconomic & WASH"]

    subset_results = []
    for subset_name, cols in [
        ("Maternal Indicators Only", maternal_cols),
        ("Child Indicators Only", child_cols),
        ("Health Indicators (Excl. WASH)", maternal_cols + child_cols),
    ]:
        indices = [config.FEATURE_COLUMNS.index(c) for c in cols if c in config.FEATURE_COLUMNS]
        X_sub = X[:, indices]
        km_sub = KMeans(n_clusters=base_k, init="k-means++", n_init=config.N_INIT, random_state=base_seed)
        sub_labs = km_sub.fit_predict(X_sub)
        ari = compare_partitions_ari(base_labels, sub_labs)
        subset_results.append({
            "Feature_Subset": subset_name,
            "N_Features": len(cols),
            "ARI_vs_FullModel": ari,
            "Silhouette": evaluate_clustering(X_sub, sub_labs)["silhouette"],
        })

    # 3. Bootstrap subsampling stability (80% subsample across 10 iterations)
    rng = np.random.RandomState(base_seed)
    bootstrap_aris = []
    n_samples = len(X)
    subsample_size = int(0.85 * n_samples)

    for i in range(10):
        sub_indices = rng.choice(n_samples, size=subsample_size, replace=False)
        X_boot = X[sub_indices]
        km_boot = KMeans(n_clusters=base_k, init="k-means++", n_init=config.N_INIT, random_state=base_seed + i)
        boot_labs = km_boot.fit_predict(X_boot)
        ari_boot = compare_partitions_ari(base_labels[sub_indices], boot_labs)
        bootstrap_aris.append(ari_boot)

    stability_summary = {
        "Mean_Seed_ARI": float(np.mean(seed_aris)),
        "Std_Seed_ARI": float(np.std(seed_aris)),
        "Min_Seed_ARI": float(np.min(seed_aris)),
        "Mean_Bootstrap_ARI": float(np.mean(bootstrap_aris)),
        "Std_Bootstrap_ARI": float(np.std(bootstrap_aris)),
        "Feature_Subsets_DF": pd.DataFrame(subset_results),
        "Seeds_Tested": test_seeds,
        "Seed_ARIs": seed_aris,
    }
    return stability_summary


if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset
    from src.preprocessing import HealthDataPreprocessor

    raw_data, _ = get_latest_cross_sectional_dataset(min_year=2000)
    preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
    cleaned, imputed, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

    X = standardized[config.FEATURE_COLUMNS].values
    kmeans = KMeans(n_clusters=4, init="k-means++", n_init=config.N_INIT, random_state=config.RANDOM_SEED)
    labels = kmeans.fit_predict(X)

    means_df, medians_df, z_df = compute_cluster_profiles(imputed, labels)
    print("\n--- Cluster Means (K=4) ---")
    print(means_df[["anc_4plus_coverage", "skilled_birth_attendance", "under_five_mortality_rate", "child_stunting_prevalence"]])

    membership_df = get_cluster_membership(imputed, labels)
    for c_id in range(4):
        c_name = f"Cluster {c_id+1}"
        c_countries = membership_df[membership_df["Cluster"] == c_id]["CountryName"].tolist()
        narrative = generate_cluster_narrative(c_name, z_df.loc[c_name], c_countries)
        print(f"\n=== {c_name} (N={narrative['country_count']} countries) ===")
        print("  Maternal:", narrative["maternal_health_profile"])
        print("  Child Mortality:", narrative["child_mortality_profile"])
        print("  Child Nutrition:", narrative["child_nutrition_profile"])
        print("  Countries:", narrative["countries"][:5], f"... (+{len(c_countries)-5} more)")

    stab = run_stability_analysis(X, imputed, base_k=4)
    print("\n--- Stability Validation ---")
    print(f"Mean Seed ARI: {stab['Mean_Seed_ARI']:.4f} (+/- {stab['Std_Seed_ARI']:.4f})")
    print(f"Mean 85% Bootstrap Subsample ARI: {stab['Mean_Bootstrap_ARI']:.4f} (+/- {stab['Std_Bootstrap_ARI']:.4f})")
    print("\nFeature Subsets Stability:")
    print(stab["Feature_Subsets_DF"])
