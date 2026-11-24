"""
Hierarchical Clustering Module for Maternal-Child Health Analysis.
Implements Agglomerative Hierarchical Clustering across Ward, Average, and Complete
linkages, evaluates Cophenetic correlation coefficients, cuts dendrogram trees,
and compares partitions with K-Means using the Adjusted Rand Index (ARI).
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

import config
from src.evaluation import evaluate_clustering, compare_partitions_ari

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HierarchicalClusterer:
    """
    Performs agglomerative hierarchical clustering with multiple linkage strategies
    and quantitative tree validation via cophenetic correlation.
    """

    SUPPORTED_LINKAGES = ["ward", "average", "complete"]

    def __init__(self, metric: str = "euclidean"):
        self.metric = metric
        self.linkage_matrices: Dict[str, np.ndarray] = {}
        self.cophenetic_scores: Dict[str, float] = {}

    def fit_linkages(self, X: np.ndarray) -> Dict[str, float]:
        """
        Computes hierarchical linkage trees for Ward, Average, and Complete linkages
        and calculates their cophenetic correlation coefficients.
        """
        # Condensed pairwise distance matrix
        pdist_matrix = pdist(X, metric=self.metric)

        for method in self.SUPPORTED_LINKAGES:
            # Note: Ward linkage requires Euclidean distance
            if method == "ward" and self.metric != "euclidean":
                raise ValueError("Ward linkage strictly requires Euclidean distance.")

            Z = linkage(X, method=method, metric=self.metric)
            self.linkage_matrices[method] = Z

            # Cophenetic correlation coefficient
            coph_corr, _ = cophenet(Z, pdist_matrix)
            self.cophenetic_scores[method] = float(coph_corr)
            logger.info(f"Linkage: '{method}', Cophenetic Correlation: {coph_corr:.4f}")

        return self.cophenetic_scores

    def get_cluster_labels(self, method: str, n_clusters: int) -> np.ndarray:
        """
        Cuts the dendrogram tree at a specific number of clusters.
        """
        if method not in self.linkage_matrices:
            raise ValueError(f"Linkage '{method}' has not been fitted yet.")

        Z = self.linkage_matrices[method]
        # fcluster returns 1-based labels; adjust to 0-based for scikit-learn convention
        labels = fcluster(Z, t=n_clusters, criterion="maxclust") - 1
        return labels

    def compare_linkages(
        self, X: np.ndarray, n_clusters: int = config.OPTIMAL_K_DEFAULT
    ) -> pd.DataFrame:
        """
        Evaluates and compares Ward, Average, and Complete linkages at a fixed cluster count.
        """
        records = []
        for method in self.SUPPORTED_LINKAGES:
            labels = self.get_cluster_labels(method, n_clusters)
            eval_metrics = evaluate_clustering(X, labels, metric=self.metric)
            records.append({
                "Linkage": method.capitalize(),
                "Cophenetic_Correlation": self.cophenetic_scores.get(method, float("nan")),
                "Silhouette": eval_metrics["silhouette"],
                "DaviesBouldin": eval_metrics["davies_bouldin"],
                "CalinskiHarabasz": eval_metrics["calinski_harabasz"],
                "ClusterSizes": eval_metrics["cluster_sizes"],
            })
        return pd.DataFrame(records)

    def compare_with_kmeans(
        self,
        X: np.ndarray,
        kmeans_labels: np.ndarray,
        n_clusters: int = config.OPTIMAL_K_DEFAULT,
    ) -> pd.DataFrame:
        """
        Computes Adjusted Rand Index (ARI) comparing each hierarchical linkage with K-Means.
        """
        records = []
        for method in self.SUPPORTED_LINKAGES:
            hier_labels = self.get_cluster_labels(method, n_clusters)
            ari = compare_partitions_ari(kmeans_labels, hier_labels)
            records.append({
                "Linkage": method.capitalize(),
                "ARI_vs_KMeans": ari,
                "Hierarchical_Silhouette": evaluate_clustering(X, hier_labels, metric=self.metric)["silhouette"],
            })
        return pd.DataFrame(records)


if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset
    from src.preprocessing import HealthDataPreprocessor
    from sklearn.cluster import KMeans

    raw_data, _ = get_latest_cross_sectional_dataset(min_year=2000)
    preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
    _, _, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

    X = standardized[config.FEATURE_COLUMNS].values
    hier = HierarchicalClusterer()
    coph_scores = hier.fit_linkages(X)

    print("\n--- Cophenetic Correlation Scores ---")
    for m, score in coph_scores.items():
        print(f"  {m.capitalize()}: {score:.4f}")

    print("\n--- Linkage Comparison (K=4) ---")
    print(hier.compare_linkages(X, n_clusters=4))

    kmeans = KMeans(n_clusters=4, random_state=config.RANDOM_SEED, n_init=config.N_INIT)
    km_labels = kmeans.fit_predict(X)
    print("\n--- Agreement with K-Means (K=4) ---")
    print(hier.compare_with_kmeans(X, km_labels, n_clusters=4))
