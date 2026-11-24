"""
Unit and Integration Test Suite for Maternal-Child Health Clustering Pipeline.
Tests data loading, preprocessing transformations, clustering algorithms,
hierarchical linkages, PCA dimensionality reduction, and cluster profiling
using Python's standard unittest framework.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from src.data_loader import load_raw_data, select_standard_records, get_latest_cross_sectional_dataset
from src.preprocessing import HealthDataPreprocessor
from src.clustering import KMeans, ManhattanKMedoids, run_kmeans_experiments
from src.hierarchical import HierarchicalClusterer
from src.dimensionality_reduction import HealthPCA
from src.interpretation import compute_cluster_profiles, generate_cluster_narrative


class TestMaternalChildHealthPipeline(unittest.TestCase):

    def test_raw_data_loading(self):
        """Verifies that raw DHS API indicators and surveys metadata exist and load correctly."""
        raw_df, meta_df, _ = load_raw_data(config.RAW_DATA_FILE, config.SURVEY_META_FILE)
        self.assertFalse(raw_df.empty, "Raw indicators dataframe must not be empty")
        self.assertFalse(meta_df.empty, "Survey metadata dataframe must not be empty")
        self.assertIn("CountryName", raw_df.columns)
        self.assertIn("IndicatorId", raw_df.columns)
        self.assertIn("Value", raw_df.columns)

    def test_cross_sectional_creation(self):
        """Verifies that the cross-sectional dataset contains valid countries and core indicators."""
        cross_df, _ = get_latest_cross_sectional_dataset(min_year=2000)
        self.assertGreaterEqual(len(cross_df), 60, f"Expected at least 60 countries, got {len(cross_df)}")
        for feat in config.FEATURE_COLUMNS:
            self.assertIn(feat, cross_df.columns, f"Feature {feat} must be present in cross-sectional matrix")

    def test_preprocessing_pipeline(self):
        """Verifies imputation and standardization logic without data leakage."""
        raw_df, _ = get_latest_cross_sectional_dataset(min_year=2000)
        preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
        cleaned, imputed, standardized = preprocessor.fit_transform_pipeline(raw_df, save_artifacts=False)

        # Imputed dataset must have 0 missing values across features
        feat_cols = [c for c in config.FEATURE_COLUMNS if c in imputed.columns]
        self.assertEqual(int(imputed[feat_cols].isna().sum().sum()), 0, "Imputed dataset must contain no missing values")

        # Standardized features must have mean approx 0 and std approx 1
        means = standardized[feat_cols].mean()
        stds = standardized[feat_cols].std()
        self.assertTrue(np.allclose(means, 0.0, atol=1e-2), "Standardized features must have zero mean")
        self.assertTrue(np.allclose(stds, 1.0, atol=5e-2), "Standardized features must have unit variance")

    def test_kmeans_experiments(self):
        """Verifies K-Means grid search across K=2..5."""
        X = np.random.RandomState(42).randn(40, 6)
        results_df, models = run_kmeans_experiments(X, k_range=[2, 3, 4, 5], random_state=42)

        self.assertEqual(len(results_df), 4)
        self.assertIn("WCSS", results_df.columns)
        self.assertIn("Silhouette", results_df.columns)
        self.assertIn("DaviesBouldin", results_df.columns)
        # Silhouette score must be in valid mathematical bounds [-1, 1]
        self.assertTrue(results_df["Silhouette"].between(-1.0, 1.0).all())

    def test_manhattan_kmedoids(self):
        """Verifies PAM K-Medoids algorithm with Manhattan distance."""
        X = np.array([
            [1.0, 1.0], [1.2, 0.8], [0.9, 1.1],
            [8.0, 8.0], [8.1, 7.9], [7.8, 8.2],
        ])
        kmed = ManhattanKMedoids(n_clusters=2, random_state=42, metric="manhattan")
        labels = kmed.fit_predict(X)

        self.assertEqual(len(labels), 6)
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(kmed.cluster_centers_.shape, (2, 2))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[3], labels[4])
        self.assertEqual(labels[4], labels[5])
        self.assertNotEqual(labels[0], labels[3])

    def test_hierarchical_clustering(self):
        """Verifies hierarchical agglomerative linkage and cophenetic score computation."""
        X = np.random.RandomState(42).randn(30, 5)
        hier = HierarchicalClusterer(metric="euclidean")
        coph = hier.fit_linkages(X)

        self.assertIn("ward", coph)
        self.assertIn("average", coph)
        self.assertIn("complete", coph)
        for link, score in coph.items():
            self.assertTrue(0.0 <= score <= 1.0, f"Cophenetic score for {link} must be between 0 and 1")

        labels = hier.get_cluster_labels("ward", n_clusters=3)
        self.assertEqual(len(np.unique(labels)), 3)

    def test_pca_decomposition(self):
        """Verifies PCA explained variance and projection dimensions."""
        X = np.random.RandomState(42).randn(50, 8)
        pca = HealthPCA(n_components=4)
        coords = pca.fit_transform(X)

        self.assertEqual(coords.shape, (50, 4))
        self.assertTrue(pca.cumulative_variance_ratio_[-1] <= 1.0 + 1e-5)
        self.assertEqual(len(pca.get_top_contributors(component=1, top_n=3)), 3)

    def test_cluster_profiling_and_narratives(self):
        """Verifies cluster profile calculation and automated epidemiological narratives."""
        df_dummy = pd.DataFrame({
            "CountryName": ["Country A", "Country B", "Country C", "Country D"],
            "SurveyYear": [2020, 2021, 2020, 2019],
            "anc_4plus_coverage": [90.0, 88.0, 30.0, 35.0],
            "under_five_mortality_rate": [15.0, 18.0, 95.0, 90.0],
        })
        labels = np.array([0, 0, 1, 1])

        means_df, medians_df, z_df = compute_cluster_profiles(df_dummy, labels, feature_cols=["anc_4plus_coverage", "under_five_mortality_rate"])
        self.assertEqual(len(means_df), 2)
        self.assertGreater(means_df.loc["Cluster 1", "anc_4plus_coverage"], means_df.loc["Cluster 2", "anc_4plus_coverage"])

        narr = generate_cluster_narrative("Cluster 1", z_df.loc["Cluster 1"], ["Country A", "Country B"])
        self.assertEqual(narr["country_count"], 2)
        self.assertIn("maternal_health_profile", narr)


if __name__ == "__main__":
    unittest.main()
