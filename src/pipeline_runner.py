"""
End-to-End Pipeline Runner for Maternal-Child Health Unsupervised Clustering.
Executes the full pipeline:
1. Ingestion of DHS indicators
2. Preprocessing, missingness analysis & standardisation
3. K-Means grid search (K=2..10) & Manhattan K-Medoids comparison
4. Hierarchical clustering, linkage trees & dendrograms
5. PCA dimensionality reduction & biplots
6. Cluster profiling, radar charts, & stability validation
7. Exporting high-resolution figures, summary tables, and models.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import logging
import sys
from pathlib import Path
import json
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans

import config
from src.data_loader import get_latest_cross_sectional_dataset, get_dataset_summary
from src.preprocessing import HealthDataPreprocessor
from src.clustering import run_kmeans_experiments, run_kmedoids_experiments, compare_kmeans_vs_kmedoids
from src.hierarchical import HierarchicalClusterer
from src.dimensionality_reduction import HealthPCA
from src.interpretation import (
    compute_cluster_profiles,
    generate_cluster_narrative,
    get_cluster_membership,
    run_stability_analysis,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configure visualization styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def run_full_pipeline():
    logger.info("=== STEP 1: LOADING RAW DHS DATASET ===")
    raw_df, _ = get_latest_cross_sectional_dataset(min_year=2000)
    summary = get_dataset_summary(raw_df)
    logger.info(f"Loaded {summary['num_countries']} countries across {summary['indicators_count']} indicators.")

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summary["stats"]).to_csv(config.TABLES_DIR / "dhs_indicators_raw_summary.csv")

    # Figure 1: Missing values heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    feat_cols = [c for c in config.FEATURE_COLUMNS if c in raw_df.columns]
    missing_matrix = raw_df.set_index("CountryName")[feat_cols].isna()
    sns.heatmap(missing_matrix.T, cmap="Blues", cbar=True, ax=ax, yticklabels=[config.SHORT_TO_LABEL.get(c, c) for c in feat_cols])
    ax.set_title("Missing-Value Matrix Across Countries and Health Indicators (Pre-Filtering)")
    ax.set_xlabel("Countries")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "01_missing_values_heatmap.png")
    plt.close()

    logger.info("=== STEP 2: PREPROCESSING & STANDARDIZATION ===")
    preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
    cleaned_df, imputed_df, standardized_df = preprocessor.fit_transform_pipeline(raw_df, save_artifacts=True)
    joblib.dump(preprocessor, config.MODELS_DIR / "preprocessor.joblib")

    # Missing values table
    missing_feat, missing_country = preprocessor.analyze_missingness(raw_df)
    missing_feat.to_csv(config.TABLES_DIR / "missing_values_by_feature.csv")

    # Figure 2: Feature distributions
    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(feat_cols):
        sns.histplot(imputed_df[col], kde=True, ax=axes[i], color="#1f77b4", bins=15)
        axes[i].set_title(config.SHORT_TO_LABEL.get(col, col), fontsize=9)
        axes[i].set_xlabel("")
    for j in range(len(feat_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Distributions of Maternal-Child Health Indicators (Imputed Cross-Section)", y=0.99)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "02_feature_distributions.png")
    plt.close()

    # Figure 3: Correlation Heatmap
    corr_matrix = preprocessor.compute_correlations(imputed_df)
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
        xticklabels=[config.SHORT_TO_LABEL.get(c, c) for c in corr_matrix.columns],
        yticklabels=[config.SHORT_TO_LABEL.get(c, c) for c in corr_matrix.columns],
        ax=ax
    )
    ax.set_title("Pairwise Pearson Correlation Matrix of Maternal-Child Health Indicators")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "03_correlation_heatmap.png")
    plt.close()

    # Figure 4: Before and After Standardisation Comparison
    sample_feats = ["skilled_birth_attendance", "under_five_mortality_rate", "child_stunting_prevalence", "improved_sanitation_access"]
    fig, axes = plt.subplots(2, len(sample_feats), figsize=(16, 8))
    for i, col in enumerate(sample_feats):
        sns.boxplot(y=imputed_df[col], ax=axes[0, i], color="#aec7e8")
        axes[0, i].set_title(f"Raw: {config.SHORT_TO_LABEL.get(col, col)}", fontsize=10)
        axes[0, i].set_ylabel("Original Units")

        sns.boxplot(y=standardized_df[col], ax=axes[1, i], color="#2ca02c")
        axes[1, i].set_title(f"Standardized: {config.SHORT_TO_LABEL.get(col, col)}", fontsize=10)
        axes[1, i].set_ylabel("Standardized Z-Score")
    plt.suptitle("Before-and-After Preprocessing: Distribution Comparison Across Key Features", y=0.99)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "04_before_after_preprocessing.png")
    plt.close()

    logger.info("=== STEP 3: K-MEANS EXPERIMENTATION & EVALUATION ===")
    X = standardized_df[feat_cols].values
    kmeans_results, kmeans_models = run_kmeans_experiments(X)
    kmeans_results.to_csv(config.TABLES_DIR / "kmeans_metrics_k2_to_10.csv", index=False)

    # Figure 5: Elbow curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(kmeans_results["K"], kmeans_results["WCSS"], "o-", color="#1f77b4", linewidth=2.5, markersize=8)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS / Inertia)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_title("Elbow Method for Optimal K (K-Means Clustering)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "05_kmeans_elbow_curve.png")
    plt.close()

    # Figure 6: Silhouette and Davies-Bouldin Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(kmeans_results["K"], kmeans_results["Silhouette"], "s-", color="#2ca02c", linewidth=2.5, markersize=8)
    ax1.set_title("Silhouette Score vs. K (Higher is Better)")
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Average Silhouette Coefficient")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(kmeans_results["K"], kmeans_results["DaviesBouldin"], "d-", color="#d62728", linewidth=2.5, markersize=8)
    ax2.set_title("Davies-Bouldin Index vs. K (Lower is Better)")
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Davies-Bouldin Index")
    ax2.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "06_kmeans_silhouette_db_curve.png")
    plt.close()

    # Figure 7: Cluster Size Distribution across K
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for idx, k in enumerate([2, 3, 4, 5, 6, 7]):
        row = kmeans_results[kmeans_results["K"] == k].iloc[0]
        sizes = row["ClusterSizes"]
        axes[idx].bar([f"C{i+1}" for i in range(k)], sizes, color=sns.color_palette("tab10", k))
        axes[idx].set_title(f"K = {k} (Sizes: {sizes})", fontsize=10)
        axes[idx].set_ylabel("Countries")
        axes[idx].set_ylim(0, max(sizes) + 5)
    plt.suptitle("Cluster Size Distribution Balance Across K Configurations", y=0.99)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "07_kmeans_cluster_sizes.png")
    plt.close()

    logger.info("=== STEP 4: DISTANCE COMPARISON (K-MEANS VS MANHATTAN K-MEDOIDS) ===")
    kmedoids_results, kmedoids_models = run_kmedoids_experiments(X, metric="manhattan")
    kmedoids_results.to_csv(config.TABLES_DIR / "kmedoids_manhattan_metrics_k2_to_10.csv", index=False)

    comp_k4 = compare_kmeans_vs_kmedoids(X, k=4)
    with open(config.TABLES_DIR / "kmeans_vs_kmedoids_head_to_head.json", "w") as f:
        json.dump(comp_k4, f, indent=2)

    # Figure 8: Euclidean K-Means vs Manhattan K-Medoids Silhouette Comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kmeans_results["K"], kmeans_results["Silhouette"], "o-", label="Euclidean K-Means (Euclidean Silhouette)", color="#1f77b4", linewidth=2)
    ax.plot(kmedoids_results["K"], kmedoids_results["Silhouette_Manhattan"], "s--", label="Manhattan K-Medoids (Manhattan Silhouette)", color="#ff7f0e", linewidth=2)
    ax.set_title("Methodological Distance Metric Comparison: Euclidean K-Means vs. Manhattan K-Medoids")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score (Corresponding Metric)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "08_kmeans_vs_kmedoids_comparison.png")
    plt.close()

    logger.info("=== STEP 5: HIERARCHICAL CLUSTERING COMPARISON ===")
    hier = HierarchicalClusterer(metric="euclidean")
    coph_scores = hier.fit_linkages(X)
    hier_comparison = hier.compare_linkages(X, n_clusters=4)
    hier_comparison.to_csv(config.TABLES_DIR / "hierarchical_linkage_evaluation.csv", index=False)

    # Figure 9: Ward Linkage Dendrogram
    fig, ax = plt.subplots(figsize=(16, 7))
    country_names = standardized_df["CountryName"].tolist()
    dendrogram(
        hier.linkage_matrices["ward"],
        labels=country_names,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        color_threshold=hier.linkage_matrices["ward"][-3, 2]  # Threshold for 4 clusters
    )
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage, Euclidean Distance)")
    ax.set_xlabel("Countries")
    ax.set_ylabel("Cluster Merge Distance (Ward Criterion)")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "09_hierarchical_dendrogram_ward.png")
    plt.close()

    # Figure 10: Hierarchical Linkage Comparison Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    hier_comparison.plot(kind="bar", x="Linkage", y=["Cophenetic_Correlation", "Silhouette", "DaviesBouldin"], ax=ax, colormap="viridis")
    ax.set_title("Hierarchical Linkage Methods: Cophenetic Correlation, Silhouette, & Davies-Bouldin")
    ax.set_ylabel("Score")
    ax.set_xticklabels(hier_comparison["Linkage"], rotation=0)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "10_hierarchical_linkage_comparison.png")
    plt.close()

    logger.info("=== STEP 6: PCA DIMENSIONALITY REDUCTION & VISUALIZATION ===")
    pca_model = HealthPCA()
    X_pca = pca_model.fit_transform(X, feature_names=feat_cols)
    joblib.dump(pca_model, config.MODELS_DIR / "pca_model.joblib")

    pca_summary = pca_model.get_variance_summary()
    pca_summary.to_csv(config.TABLES_DIR / "pca_variance_explained.csv", index=False)
    pca_model.loadings_df_.to_csv(config.TABLES_DIR / "pca_feature_loadings.csv")

    # Fit chosen K-Means model (K=4)
    best_kmeans = kmeans_models[4]
    best_labels = best_kmeans.labels_
    joblib.dump(best_kmeans, config.MODELS_DIR / "kmeans_k4.joblib")

    # Figure 11: PCA Explained Variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar([f"PC{i+1}" for i in range(10)], pca_summary["Explained_Variance_Pct"].head(10), color="#1f77b4")
    ax1.set_title("PCA Individual Explained Variance Ratio (%)")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot([f"PC{i+1}" for i in range(10)], pca_summary["Cumulative_Variance_Pct"].head(10), "o-", color="#d62728", linewidth=2)
    ax2.axhline(y=70, color="gray", linestyle="--", alpha=0.7, label="70% Threshold")
    ax2.set_title("PCA Cumulative Explained Variance Ratio (%)")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "11_pca_explained_variance.png")
    plt.close()

    # Figure 12: PCA 2D Scatter Plot with Cluster Assignments
    proj_df = pca_model.create_projection_dataframe(
        X, country_names, cluster_labels=best_labels, survey_years=standardized_df["SurveyYear"].tolist()
    )
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    for c_id in range(4):
        sub = proj_df[proj_df["Cluster"] == c_id]
        ax.scatter(sub["PC1"], sub["PC2"], label=f"Cluster {c_id+1} (N={len(sub)})", color=colors[c_id], s=90, alpha=0.85, edgecolors="k", linewidth=0.5)
        # Annotate representative countries
        for _, row in sub.head(4).iterrows():
            ax.annotate(row["CountryName"], (row["PC1"] + 0.1, row["PC2"] + 0.1), fontsize=8, alpha=0.85)

    ax.set_title(f"PCA 2D Cluster Separation (K-Means, K=4)\nPC1: {pca_model.explained_variance_ratio_[0]*100:.1f}% var | PC2: {pca_model.explained_variance_ratio_[1]*100:.1f}% var")
    ax.set_xlabel("Principal Component 1 (Overall Health System & Maternal Access)")
    ax.set_ylabel("Principal Component 2 (Breastfeeding & Immunization Specifics)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend(title="Clusters")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "12_pca_cluster_scatter_2d.png")
    plt.close()

    # Figure 13: PCA Biplot Loadings
    fig, ax = plt.subplots(figsize=(10, 8))
    loadings = pca_model.loadings_df_
    ax.scatter(loadings["PC1"], loadings["PC2"], color="purple", s=60)
    for feat in loadings.index:
        x_val, y_val = loadings.loc[feat, "PC1"], loadings.loc[feat, "PC2"]
        ax.arrow(0, 0, x_val * 0.9, y_val * 0.9, color="purple", alpha=0.4, head_width=0.03)
        ax.text(x_val * 1.05, y_val * 1.05, config.SHORT_TO_LABEL.get(feat, feat), fontsize=8, color="darkblue")
    ax.set_title("PCA Feature Loadings (Variable Contributions to PC1 and PC2)")
    ax.set_xlabel(f"PC1 Loadings ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 Loadings ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "13_pca_biplot_loadings.png")
    plt.close()

    logger.info("=== STEP 7: CLUSTER PROFILING & INTERPRETATION ===")
    means_df, medians_df, z_df = compute_cluster_profiles(imputed_df, best_labels, feature_cols=feat_cols)
    means_df.to_csv(config.TABLES_DIR / "cluster_profiles_unscaled_means.csv")
    medians_df.to_csv(config.TABLES_DIR / "cluster_profiles_unscaled_medians.csv")
    z_df.to_csv(config.TABLES_DIR / "cluster_profiles_zscores.csv")

    membership_df = get_cluster_membership(standardized_df, best_labels)
    membership_df.to_csv(config.TABLES_DIR / "country_cluster_assignments.csv", index=False)

    # Narrative profiles dictionary
    cluster_narratives = {}
    for c_id in range(4):
        c_name = f"Cluster {c_id+1}"
        c_countries = membership_df[membership_df["Cluster"] == c_id]["CountryName"].tolist()
        narrative = generate_cluster_narrative(c_name, z_df.loc[c_name], c_countries)
        cluster_narratives[c_name] = narrative

    with open(config.TABLES_DIR / "cluster_narratives.json", "w") as f:
        json.dump(cluster_narratives, f, indent=2)

    # Figure 14: Cluster Profile Heatmap (Z-Scores)
    fig, ax = plt.subplots(figsize=(14, 6))
    z_df_labeled = z_df.rename(columns=config.SHORT_TO_LABEL)
    sns.heatmap(z_df_labeled, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar_kws={"label": "Standardized Deviation (Z-score)"}, ax=ax)
    ax.set_title("Cluster Profiles Heatmap: Standardized Indicator Deviations from Survey Mean (K=4)")
    ax.set_xlabel("Health Indicators")
    ax.set_ylabel("Clusters")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "14_cluster_profiles_heatmap.png")
    plt.close()

    # Figure 15: Radar Charts for Cluster Profiles
    radar_categories = [
        "skilled_birth_attendance", "anc_4plus_coverage", "dpt3_vaccination_coverage",
        "improved_sanitation_access", "female_secondary_education", "child_stunting_prevalence"
    ]
    N = len(radar_categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
    axes = axes.flatten()

    for c_id in range(4):
        c_name = f"Cluster {c_id+1}"
        values = z_df.loc[c_name, radar_categories].values.flatten().tolist()
        values += values[:1]

        ax = axes[c_id]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([config.SHORT_TO_LABEL.get(c, c) for c in radar_categories], fontsize=8)
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=colors[c_id], label=c_name)
        ax.fill(angles, values, color=colors[c_id], alpha=0.25)
        ax.set_title(f"{c_name} (N={len(membership_df[membership_df['Cluster'] == c_id])})", size=11, color=colors[c_id], y=1.1)
        ax.set_ylim(-2.5, 2.0)

    plt.suptitle("Multi-Dimensional Radar Profiles (Standardized Z-Scores Across Core Health Pillars)", y=0.98)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "15_cluster_radar_charts.png")
    plt.close()

    # Figure 16: Comparative Bar Charts across Clusters for Key Metrics
    key_metrics = ["under_five_mortality_rate", "skilled_birth_attendance", "child_stunting_prevalence", "improved_sanitation_access"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for idx, metric in enumerate(key_metrics):
        metric_means = means_df[metric]
        axes[idx].bar([f"C{i+1}" for i in range(4)], metric_means, color=colors, edgecolor="black", alpha=0.85)
        axes[idx].set_title(config.SHORT_TO_LABEL.get(metric, metric), fontsize=11)
        axes[idx].set_ylabel(config.INDICATORS_CONFIG[list(config.INDICATORS_CONFIG.keys())[[c for c in feat_cols].index(metric)]]["unit"])
        for bar in axes[idx].patches:
            axes[idx].annotate(f"{bar.get_height():.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                               ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    plt.suptitle("Cluster-Wise Comparison of Key Maternal and Child Health Outcomes (Unscaled Means)", y=0.98)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "16_cluster_comparative_bars.png")
    plt.close()

    logger.info("=== STEP 8: STABILITY VALIDATION ===")
    stability_res = run_stability_analysis(X, imputed_df, base_k=4)
    stab_df = stability_res["Feature_Subsets_DF"]
    stab_df.to_csv(config.TABLES_DIR / "stability_validation_summary.csv", index=False)

    # Figure 17: Stability evaluation across seeds and subsets
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(stability_res["Seeds_Tested"], stability_res["Seed_ARIs"], "o-", color="#1f77b4", linewidth=2)
    ax1.axhline(stability_res["Mean_Seed_ARI"], color="red", linestyle="--", label=f"Mean ARI: {stability_res['Mean_Seed_ARI']:.3f}")
    ax1.set_title("Cluster Partition Stability Across 10 Random Initializations (ARI vs Base)")
    ax1.set_xlabel("Random Seed")
    ax1.set_ylabel("Adjusted Rand Index (ARI)")
    ax1.set_ylim(0.4, 1.05)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(stab_df["Feature_Subset"], stab_df["ARI_vs_FullModel"], color="#2ca02c", alpha=0.8)
    ax2.set_title("Partition Agreement (ARI) Across Domain Feature Subsets vs. Full Model")
    ax2.set_ylabel("Adjusted Rand Index (ARI)")
    ax2.set_xticklabels(stab_df["Feature_Subset"], rotation=15, ha="right")
    ax2.set_ylim(0, 1.0)
    for bar in ax2.patches:
        ax2.annotate(f"{bar.get_height():.3f}", (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02),
                     ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "17_cluster_stability_evaluation.png")
    plt.close()

    logger.info("=== PIPELINE EXECUTION SUCCESSFULLY COMPLETED ===")
    logger.info(f"17 High-resolution figures saved to: {config.FIGURES_DIR}")
    logger.info(f"13 Statistical summary tables saved to: {config.TABLES_DIR}")
    logger.info(f"Trained models saved to: {config.MODELS_DIR}")

    # === STEP 9: ANALYTICS — GAP DETECTION, VULNERABILITY, PRIORITY ===
    logger.info("=== STEP 9: HEALTHCARE GAP DETECTION & VULNERABILITY PROFILING ===")
    try:
        from analytics.gap_detection import calculate_region_gaps
        from analytics.vulnerability import build_vulnerability_dataframe
        from analytics.prioritisation import build_priority_report, run_scenario_analysis, PREDEFINED_SCENARIOS

        gaps_df = calculate_region_gaps(imputed_df)
        gaps_df.to_csv(config.TABLES_DIR / "healthcare_gaps.csv", index=False)
        logger.info(f"Gap scores saved: {len(gaps_df)} countries.")

        vuln_df = build_vulnerability_dataframe(gaps_df, alpha=0.5)
        vuln_df.to_csv(config.TABLES_DIR / "vulnerability_indices.csv", index=False)
        logger.info(f"Vulnerability indices saved.")

        priority_df = build_priority_report(vuln_df)
        priority_df.to_csv(config.TABLES_DIR / "priority_rankings.csv", index=False)
        logger.info(f"Priority rankings saved.")

        scenario_df = run_scenario_analysis(vuln_df, PREDEFINED_SCENARIOS)
        scenario_df.to_csv(config.TABLES_DIR / "scenario_comparison.csv", index=False)
        logger.info(f"Scenario comparison saved.")

    except Exception as e:
        logger.error(f"Analytics step failed (non-fatal): {e}")

    logger.info("=== ALL PIPELINE STEPS COMPLETED ===")


if __name__ == "__main__":
    run_full_pipeline()
