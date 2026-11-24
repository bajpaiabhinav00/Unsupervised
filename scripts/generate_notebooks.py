"""
Programmatic generator for 6 comprehensive Jupyter Notebooks
corresponding to the analysis phases of the Maternal-Child Health project.
"""

import sys
from pathlib import Path
import nbformat as nbf

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_nb_01():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 01: Exploratory Data Analysis & DHS Indicator Inspection
**Project**: Maternal-Child Health Profiles Across Countries Through Unsupervised Clustering  
**Source**: Demographic and Health Surveys (DHS) Program API

### Objectives:
1. Load raw multi-country indicator records and survey metadata.
2. Filter for standard 5-year retrospective national estimates.
3. Construct the latest cross-sectional dataset per country ($SurveyYear \\ge 2000$).
4. Inspect dimensions, data types, and preliminary missingness patterns."""),
        nbf.v4.new_code_cell("""import sys
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
from src.data_loader import load_raw_data, select_standard_records, get_latest_cross_sectional_dataset, get_dataset_summary"""),
        nbf.v4.new_markdown_cell("### 1. Ingest Raw DHS Indicators & Metadata"),
        nbf.v4.new_code_cell("""raw_df, meta_df = load_raw_data(config.RAW_DATA_FILE, config.SURVEY_META_FILE)
print(f"Raw indicator records: {len(raw_df):,}")
print(f"Available surveys: {len(meta_df):,}")
raw_df.head(5)"""),
        nbf.v4.new_markdown_cell("### 2. Standardize Reference Windows & Construct Cross-Section"),
        nbf.v4.new_code_cell("""cross_df = get_latest_cross_sectional_dataset(min_year=2000)
summary = get_dataset_summary(cross_df)
print(f"Constructed cross-sectional matrix: {summary['num_countries']} countries x {summary['indicators_count']} indicators")
print(f"Overall missingness: {summary['missing_percentage']:.2f}%")
cross_df.head()"""),
        nbf.v4.new_markdown_cell("### 3. Missingness Matrix Across Countries"),
        nbf.v4.new_code_cell("""feat_cols = [c for c in config.FEATURE_COLUMNS if c in cross_df.columns]
missing_counts = cross_df[feat_cols].isna().sum()
missing_pct = (missing_counts / len(cross_df)) * 100

missing_summary = pd.DataFrame({"Missing Count": missing_counts, "Missing %": missing_pct})
print(missing_summary)

plt.figure(figsize=(12, 6))
sns.heatmap(cross_df.set_index("CountryName")[feat_cols].isna().T, cmap="Blues", cbar=True)
plt.title("Missingness Pattern Matrix across Countries and Health Indicators")
plt.xlabel("Countries")
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()
plt.show()"""),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 01_data_exploration.ipynb")


def create_nb_02():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 02: Data Preprocessing, Imputation & Scaling
**Project**: Maternal-Child Health Profiles Across Countries Through Unsupervised Clustering

### Objectives:
1. Filter countries with excessive indicator missingness ($>35\\%$).
2. Justify and apply robust median imputation.
3. Detect statistical outliers and understand their epidemiological meaning.
4. Apply StandardScaler (Z-score standardisation) to equalize feature variance."""),
        nbf.v4.new_code_cell("""import sys
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config
from src.data_loader import get_latest_cross_sectional_dataset
from src.preprocessing import HealthDataPreprocessor"""),
        nbf.v4.new_markdown_cell("### 1. Execute Preprocessing Pipeline"),
        nbf.v4.new_code_cell("""raw_data = get_latest_cross_sectional_dataset(min_year=2000)
preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
cleaned, imputed, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

print(f"Cleaned sample shape: {cleaned.shape}")
print(f"Imputed sample shape: {imputed.shape}")
print(f"Standardized sample shape: {standardized.shape}")"""),
        nbf.v4.new_markdown_cell("### 2. Outlier Analysis & Verification"),
        nbf.v4.new_code_cell("""outliers_df = preprocessor.detect_outliers(imputed)
print(outliers_df[["Label", "IQR_Outlier_Count", "ZScore_Outlier_Count", "Min", "Max"]])"""),
        nbf.v4.new_markdown_cell("### 3. Before and After Standardisation Distributions"),
        nbf.v4.new_code_cell("""sample_cols = ["skilled_birth_attendance", "under_five_mortality_rate", "child_stunting_prevalence", "improved_sanitation_access"]
fig, axes = plt.subplots(2, len(sample_cols), figsize=(16, 8))
for i, col in enumerate(sample_cols):
    sns.boxplot(y=imputed[col], ax=axes[0, i], color="#aec7e8")
    axes[0, i].set_title(f"Raw: {config.SHORT_TO_LABEL.get(col, col)}", fontsize=10)
    sns.boxplot(y=standardized[col], ax=axes[1, i], color="#2ca02c")
    axes[1, i].set_title(f"Standardized: {config.SHORT_TO_LABEL.get(col, col)}", fontsize=10)
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("### 4. Correlation Structure"),
        nbf.v4.new_code_cell("""corr = preprocessor.compute_correlations(imputed)
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Pearson Correlation Heatmap Across Health Indicators")
plt.tight_layout()
plt.show()"""),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "02_preprocessing.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 02_preprocessing.ipynb")


def create_nb_03():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 03: Partition-Based Clustering (K-Means & Manhattan K-Medoids)
**Project**: Maternal-Child Health Profiles Across Countries Through Unsupervised Clustering

### Objectives:
1. Benchmark K-Means across $K=2\\dots10$ using multiple initializations (`k-means++`, $n\\_init=20$).
2. Compute WCSS/Inertia, Silhouette Scores, Davies-Bouldin Index, and Calinski-Harabasz Index.
3. Compare Euclidean K-Means with Manhattan K-Medoids (PAM) and evaluate distance metric differences."""),
        nbf.v4.new_code_cell("""import sys
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import config
from src.data_loader import get_latest_cross_sectional_dataset
from src.preprocessing import HealthDataPreprocessor
from src.clustering import run_kmeans_experiments, run_kmedoids_experiments, compare_kmeans_vs_kmedoids"""),
        nbf.v4.new_markdown_cell("### 1. K-Means Grid Search across K=2..10"),
        nbf.v4.new_code_cell("""raw_data = get_latest_cross_sectional_dataset(min_year=2000)
preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
_, _, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

X = standardized[config.FEATURE_COLUMNS].values
kmeans_df, models = run_kmeans_experiments(X)
kmeans_df[["K", "WCSS", "Silhouette", "DaviesBouldin", "CalinskiHarabasz", "ClusterSizes"]]"""),
        nbf.v4.new_markdown_cell("### 2. Elbow & Silhouette Curves"),
        nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(kmeans_df["K"], kmeans_df["WCSS"], "o-", color="#1f77b4", linewidth=2.5)
ax1.set_title("Elbow Curve (WCSS / Inertia vs K)")
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("WCSS")
ax1.grid(True)

ax2.plot(kmeans_df["K"], kmeans_df["Silhouette"], "s-", color="#2ca02c", linewidth=2.5)
ax2.set_title("Silhouette Score vs K")
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Average Silhouette")
ax2.grid(True)
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("### 3. Euclidean K-Means vs Manhattan K-Medoids Comparison"),
        nbf.v4.new_code_cell("""kmed_df, _ = run_kmedoids_experiments(X, metric="manhattan")
comp_k4 = compare_kmeans_vs_kmedoids(X, k=4)
print(f"Agreement between Euclidean K-Means and Manhattan K-Medoids (ARI): {comp_k4['ARI_Agreement']:.4f}")
print("K-Means:", comp_k4["KMeans"])
print("K-Medoids:", comp_k4["KMedoids_Manhattan"])

plt.figure(figsize=(9, 5))
plt.plot(kmeans_df["K"], kmeans_df["Silhouette"], "o-", label="Euclidean K-Means")
plt.plot(kmed_df["K"], kmed_df["Silhouette_Manhattan"], "s--", label="Manhattan K-Medoids")
plt.title("Distance Metric Comparison: Euclidean vs Manhattan Silhouette")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)
plt.show()"""),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "03_kmeans_clustering.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 03_kmeans_clustering.ipynb")


def create_nb_04():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 04: Hierarchical Clustering & Linkage Comparison
**Project**: Maternal-Child Health Profiles Across Countries Through Unsupervised Clustering

### Objectives:
1. Implement Agglomerative Hierarchical Clustering across Ward, Average, and Complete linkages.
2. Calculate Cophenetic Correlation Coefficients to evaluate tree distortion.
3. Compare hierarchical cluster cuts with K-Means using the Adjusted Rand Index (ARI)."""),
        nbf.v4.new_code_cell("""import sys
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans

import config
from src.data_loader import get_latest_cross_sectional_dataset
from src.preprocessing import HealthDataPreprocessor
from src.hierarchical import HierarchicalClusterer"""),
        nbf.v4.new_markdown_cell("### 1. Fit Hierarchical Linkages & Cophenetic Scores"),
        nbf.v4.new_code_cell("""raw_data = get_latest_cross_sectional_dataset(min_year=2000)
preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
_, _, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

X = standardized[config.FEATURE_COLUMNS].values
hier = HierarchicalClusterer(metric="euclidean")
coph_scores = hier.fit_linkages(X)

print("Cophenetic Correlation Coefficients:")
for link, score in coph_scores.items():
    print(f"  {link.capitalize()}: {score:.4f}")"""),
        nbf.v4.new_markdown_cell("### 2. Ward Linkage Dendrogram"),
        nbf.v4.new_code_cell("""plt.figure(figsize=(16, 7))
country_names = standardized["CountryName"].tolist()
dendrogram(
    hier.linkage_matrices["ward"],
    labels=country_names,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=hier.linkage_matrices["ward"][-3, 2]
)
plt.title("Hierarchical Dendrogram (Ward Linkage, Euclidean Distance)")
plt.xlabel("Countries")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("### 3. Comparison with K-Means (K=4)"),
        nbf.v4.new_code_cell("""km = KMeans(n_clusters=4, random_state=config.RANDOM_SEED, n_init=config.N_INIT)
km_labels = km.fit_predict(X)

comparison_df = hier.compare_with_kmeans(X, km_labels, n_clusters=4)
print(comparison_df)"""),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "04_hierarchical_clustering.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 04_hierarchical_clustering.ipynb")


def create_nb_05():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 05: Principal Component Analysis (PCA) & Biplot Visualisation
**Project**: Maternal-Child Health Profiles Across Countries Through Unsupervised Clustering

### Objectives:
1. Apply PCA to standardized 17-dimensional maternal-child health data.
2. Evaluate individual and cumulative explained variance.
3. Compute feature loadings and examine variable contributions (biplots).
4. Project observations into 2D space with cluster overlays."""),
        nbf.v4.new_code_cell("""import sys
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import config
from src.data_loader import get_latest_cross_sectional_dataset
from src.preprocessing import HealthDataPreprocessor
from src.dimensionality_reduction import HealthPCA"""),
        nbf.v4.new_markdown_cell("### 1. Fit PCA & Variance Summary"),
        nbf.v4.new_code_cell("""raw_data = get_latest_cross_sectional_dataset(min_year=2000)
preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
_, _, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

X = standardized[config.FEATURE_COLUMNS].values
pca = HealthPCA()
X_pca = pca.fit_transform(X, feature_names=config.FEATURE_COLUMNS)

summary_var = pca.get_variance_summary()
summary_var.head(6)"""),
        nbf.v4.new_markdown_cell("### 2. Feature Loadings (PC1 and PC2)"),
        nbf.v4.new_code_cell("""print("Top Contributors to PC1:")
print(pca.get_top_contributors(component=1, top_n=5))

print("\\nTop Contributors to PC2:")
print(pca.get_top_contributors(component=2, top_n=5))"""),
        nbf.v4.new_markdown_cell("### 3. PCA 2D Cluster Projection"),
        nbf.v4.new_code_cell("""km = KMeans(n_clusters=4, random_state=config.RANDOM_SEED, n_init=config.N_INIT)
labels = km.fit_predict(X)

country_names = standardized["CountryName"].tolist()
proj_df = pca.create_projection_dataframe(X, country_names, cluster_labels=labels)

plt.figure(figsize=(12, 8))
colors = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
for c in range(4):
    sub = proj_df[proj_df["Cluster"] == c]
    plt.scatter(sub["PC1"], sub["PC2"], label=f"Cluster {c+1} (N={len(sub)})", color=colors[c], s=80, alpha=0.85)

plt.title(f"PCA 2D Cluster Separation (PC1: {pca.explained_variance_ratio_[0]*100:.1f}%, PC2: {pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.xlabel("PC1 (Maternal Care & Health Infrastructure)")
plt.ylabel("PC2 (Child Nutrition & Immunization)")
plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
plt.axvline(0, color="gray", linestyle=":", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()"""),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "05_pca_visualisation.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 05_pca_visualisation.ipynb")


def create_nb_06():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("""# Notebook 06: Cluster Profiling, Interpretation & Stability Validation
**Project**: Maternal-Child Health Profiles Across Countries Through Unsupervised Clustering

### Objectives:
1. Compute cluster-wise means, medians, and standardized z-score deviations.
2. Construct multi-dimensional radar charts across maternal, child, and WASH pillars.
3. Generate objective, statistically grounded epidemiological narratives.
4. Execute stability validation across 10 random seeds, bootstrap subsamples, and feature subsets."""),
        nbf.v4.new_code_cell("""import sys
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

import config
from src.data_loader import get_latest_cross_sectional_dataset
from src.preprocessing import HealthDataPreprocessor
from src.interpretation import (
    compute_cluster_profiles,
    generate_cluster_narrative,
    get_cluster_membership,
    run_stability_analysis,
)"""),
        nbf.v4.new_markdown_cell("### 1. Compute Cluster Health Profiles (K=4)"),
        nbf.v4.new_code_cell("""raw_data = get_latest_cross_sectional_dataset(min_year=2000)
preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
cleaned, imputed, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

X = standardized[config.FEATURE_COLUMNS].values
km = KMeans(n_clusters=4, random_state=config.RANDOM_SEED, n_init=config.N_INIT)
labels = km.fit_predict(X)

means_df, medians_df, z_df = compute_cluster_profiles(imputed, labels)
print("Cluster Means:")
print(means_df[["skilled_birth_attendance", "under_five_mortality_rate", "child_stunting_prevalence", "improved_sanitation_access"]])"""),
        nbf.v4.new_markdown_cell("### 2. Standardized Profile Heatmap"),
        nbf.v4.new_code_cell("""plt.figure(figsize=(14, 6))
sns.heatmap(z_df.rename(columns=config.SHORT_TO_LABEL), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Cluster Standardized Deviations (Z-Scores) Across Health Indicators")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
plt.show()"""),
        nbf.v4.new_markdown_cell("### 3. Country Membership & Epidemiological Narratives"),
        nbf.v4.new_code_cell("""members_df = get_cluster_membership(standardized, labels)
for c in range(4):
    c_name = f"Cluster {c+1}"
    c_countries = members_df[members_df["Cluster"] == c]["CountryName"].tolist()
    narr = generate_cluster_narrative(c_name, z_df.loc[c_name], c_countries)
    print(f"\\n=== {c_name} (N={narr['country_count']} countries) ===")
    print("  Maternal Care:", narr["maternal_health_profile"])
    print("  Child Mortality:", narr["child_mortality_profile"])
    print("  Child Nutrition:", narr["child_nutrition_profile"])
    print("  Countries:", narr["countries"])"""),
        nbf.v4.new_markdown_cell("### 4. Stability Validation Across Seeds and Subsets"),
        nbf.v4.new_code_cell("""stability = run_stability_analysis(X, imputed, base_k=4)
print(f"Mean Seed ARI across 10 initializations: {stability['Mean_Seed_ARI']:.4f} (+/- {stability['Std_Seed_ARI']:.4f})")
print(f"Mean 85% Bootstrap Subsample ARI: {stability['Mean_Bootstrap_ARI']:.4f} (+/- {stability['Std_Bootstrap_ARI']:.4f})")
print("\\nFeature Subsets Agreement:")
print(stability["Feature_Subsets_DF"])"""),
    ]
    nb.cells = cells
    with open(NOTEBOOKS_DIR / "06_cluster_interpretation.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Created 06_cluster_interpretation.ipynb")


if __name__ == "__main__":
    create_nb_01()
    create_nb_02()
    create_nb_03()
    create_nb_04()
    create_nb_05()
    create_nb_06()
    print("All 6 Jupyter Notebooks successfully generated!")
