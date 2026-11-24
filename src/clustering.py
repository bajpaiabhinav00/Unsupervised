"""
Clustering Algorithms Module for Maternal-Child Health Analysis.
Implements:
1. Euclidean K-Means with grid search across K=2..10, iteration tracking, and convergence metrics.
2. Manhattan-compatible K-Medoids (Partitioning Around Medoids - PAM) to compare L1 vs L2 distance geometry.
3. Methodological comparison between Euclidean centroid-based vs Manhattan medoid-based partitioning.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances

import config
from src.evaluation import evaluate_clustering, compare_partitions_ari

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ManhattanKMedoids:
    """
    Partitioning Around Medoids (PAM) algorithm with Manhattan (L1) distance metric.

    Methodological distinction:
    - Standard K-Means calculates cluster centers as arithmetic means, minimizing sum of squared
      Euclidean distances (L2 norm). It is mathematically incompatible with Manhattan distance.
    - K-Medoids chooses actual exemplar data points (medoids) as cluster centers, minimizing the
      sum of dissimilarities under any arbitrary metric, notably L1 (Manhattan) distance.
    - Medoids are inherently more robust to outliers and extreme values in health indicators.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        max_iter: int = 300,
        random_state: int = config.RANDOM_SEED,
        metric: str = "manhattan",
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.metric = metric

        # Fitted attributes
        self.medoid_indices_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0

    def _compute_distance_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.metric == "manhattan":
            return manhattan_distances(X, Y if Y is not None else X)
        elif self.metric == "euclidean":
            return euclidean_distances(X, Y if Y is not None else X)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def fit(self, X: np.ndarray) -> "ManhattanKMedoids":
        """
        Fits K-Medoids clustering on dataset X.
        """
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Precompute pairwise distance matrix for computational efficiency
        D = self._compute_distance_matrix(X)

        # Step 1: Initialize medoids (k-medoids++ or random exemplar selection)
        initial_medoid = rng.choice(n_samples)
        medoid_indices = [initial_medoid]

        for _ in range(1, self.n_clusters):
            # Compute distance of each point to its closest already selected medoid
            min_dist = np.min(D[:, medoid_indices], axis=1)
            probs = min_dist / np.sum(min_dist)
            next_medoid = rng.choice(n_samples, p=probs)
            medoid_indices.append(next_medoid)

        medoid_indices = np.array(medoid_indices)

        # Step 2: Alternate Assignment and Update (PAM heuristic)
        for iteration in range(self.max_iter):
            # Assignment phase: assign each sample to the nearest medoid
            dist_to_medoids = D[:, medoid_indices]
            labels = np.argmin(dist_to_medoids, axis=1)

            # Update phase: for each cluster, select the point that minimizes sum of intra-cluster distances
            new_medoids = np.copy(medoid_indices)
            for k in range(self.n_clusters):
                cluster_members = np.where(labels == k)[0]
                if len(cluster_members) == 0:
                    # Reassign empty cluster to a random non-medoid point
                    non_medoids = list(set(range(n_samples)) - set(new_medoids))
                    new_medoids[k] = rng.choice(non_medoids)
                    continue

                # Submatrix of distances among cluster members
                sub_D = D[np.ix_(cluster_members, cluster_members)]
                sum_dist = sub_D.sum(axis=1)
                best_member_idx = cluster_members[np.argmin(sum_dist)]
                new_medoids[k] = best_member_idx

            # Check convergence
            if np.array_equal(np.sort(medoid_indices), np.sort(new_medoids)):
                self.n_iter_ = iteration + 1
                break

            medoid_indices = new_medoids
        else:
            self.n_iter_ = self.max_iter

        self.medoid_indices_ = medoid_indices
        self.cluster_centers_ = X[medoid_indices]
        self.labels_ = np.argmin(D[:, medoid_indices], axis=1)

        # Compute total dissimilarity (inertia)
        self.inertia_ = float(np.sum(np.min(D[:, medoid_indices], axis=1)))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


def run_kmeans_experiments(
    X: np.ndarray,
    k_range: List[int] = config.K_RANGE,
    random_state: int = config.RANDOM_SEED,
    n_init: int = config.N_INIT,
    max_iter: int = config.MAX_ITER,
) -> Tuple[pd.DataFrame, Dict[int, KMeans]]:
    """
    Executes K-Means experiments across multiple values of K.
    Calculates WCSS/inertia, silhouette scores, Davies-Bouldin index,
    Calinski-Harabasz index, cluster size balance, and iteration counts.
    """
    records = []
    models = {}

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(X)
        models[k] = kmeans

        metrics = evaluate_clustering(X, labels, centers=kmeans.cluster_centers_, metric="euclidean")
        records.append({
            "K": k,
            "WCSS": metrics["wcss"],
            "Inertia": float(kmeans.inertia_),
            "Silhouette": metrics["silhouette"],
            "DaviesBouldin": metrics["davies_bouldin"],
            "CalinskiHarabasz": metrics["calinski_harabasz"],
            "ClusterSizes": metrics["cluster_sizes"],
            "MinClusterSize": min(metrics["cluster_sizes"]),
            "MaxClusterSize": max(metrics["cluster_sizes"]),
            "SizeRatio": max(metrics["cluster_sizes"]) / min(metrics["cluster_sizes"]),
            "N_Iter": int(kmeans.n_iter_),
        })

    results_df = pd.DataFrame(records)
    logger.info(f"Completed K-Means experiments across K={k_range}")
    return results_df, models


def run_kmedoids_experiments(
    X: np.ndarray,
    k_range: List[int] = config.K_RANGE,
    metric: str = "manhattan",
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[int, ManhattanKMedoids]]:
    """
    Executes K-Medoids experiments across multiple values of K using Manhattan distance.
    Evaluates silhouette score using the matching Manhattan distance metric.
    """
    records = []
    models = {}

    for k in k_range:
        kmedoids = ManhattanKMedoids(
            n_clusters=k,
            max_iter=config.MAX_ITER,
            random_state=random_state,
            metric=metric,
        )
        labels = kmedoids.fit_predict(X)
        models[k] = kmedoids

        metrics = evaluate_clustering(
            X, labels, centers=kmedoids.cluster_centers_, metric=metric
        )
        records.append({
            "K": k,
            "TotalDissimilarity": kmedoids.inertia_,
            "Silhouette_Manhattan": metrics["silhouette"],
            "DaviesBouldin": metrics["davies_bouldin"],
            "ClusterSizes": metrics["cluster_sizes"],
            "MinClusterSize": min(metrics["cluster_sizes"]),
            "MaxClusterSize": max(metrics["cluster_sizes"]),
            "N_Iter": kmedoids.n_iter_,
        })

    results_df = pd.DataFrame(records)
    logger.info(f"Completed {metric.capitalize()} K-Medoids experiments across K={k_range}")
    return results_df, models


def compare_kmeans_vs_kmedoids(
    X: np.ndarray,
    k: int = 4,
    random_state: int = config.RANDOM_SEED,
) -> Dict[str, Union[float, Dict]]:
    """
    Performs head-to-head comparison between Euclidean K-Means and Manhattan K-Medoids.
    """
    # Euclidean K-Means
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=config.N_INIT, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_eval_euc = evaluate_clustering(X, kmeans_labels, centers=kmeans.cluster_centers_, metric="euclidean")
    kmeans_eval_man = evaluate_clustering(X, kmeans_labels, centers=kmeans.cluster_centers_, metric="manhattan")

    # Manhattan K-Medoids
    kmedoids = ManhattanKMedoids(n_clusters=k, random_state=random_state, metric="manhattan")
    kmedoids_labels = kmedoids.fit_predict(X)
    kmedoids_eval_man = evaluate_clustering(X, kmedoids_labels, centers=kmedoids.cluster_centers_, metric="manhattan")
    kmedoids_eval_euc = evaluate_clustering(X, kmedoids_labels, centers=kmedoids.cluster_centers_, metric="euclidean")

    # Partition similarity (Adjusted Rand Index)
    ari = compare_partitions_ari(kmeans_labels, kmedoids_labels)

    comparison = {
        "K": k,
        "ARI_Agreement": ari,
        "KMeans": {
            "Silhouette_Euclidean": kmeans_eval_euc["silhouette"],
            "Silhouette_Manhattan": kmeans_eval_man["silhouette"],
            "DaviesBouldin": kmeans_eval_euc["davies_bouldin"],
            "ClusterSizes": kmeans_eval_euc["cluster_sizes"],
            "Iterations": int(kmeans.n_iter_),
        },
        "KMedoids_Manhattan": {
            "Silhouette_Manhattan": kmedoids_eval_man["silhouette"],
            "Silhouette_Euclidean": kmedoids_eval_euc["silhouette"],
            "DaviesBouldin": kmedoids_eval_man["davies_bouldin"],
            "ClusterSizes": kmedoids_eval_man["cluster_sizes"],
            "Iterations": int(kmedoids.n_iter_),
        },
    }
    return comparison


if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset
    from src.preprocessing import HealthDataPreprocessor

    raw_data, _ = get_latest_cross_sectional_dataset(min_year=2000)
    preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
    _, _, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

    X = standardized[config.FEATURE_COLUMNS].values
    results_km, _ = run_kmeans_experiments(X)
    print("\n--- K-Means Results Across K=2..10 ---")
    print(results_km[["K", "WCSS", "Silhouette", "DaviesBouldin", "ClusterSizes"]])

    results_med, _ = run_kmedoids_experiments(X, metric="manhattan")
    print("\n--- Manhattan K-Medoids Results Across K=2..10 ---")
    print(results_med[["K", "TotalDissimilarity", "Silhouette_Manhattan", "DaviesBouldin", "ClusterSizes"]])

    comp = compare_kmeans_vs_kmedoids(X, k=4)
    print("\n--- Head-to-Head Comparison (K=4) ---")
    print(f"Partition Agreement (ARI): {comp['ARI_Agreement']:.4f}")
    print("K-Means:", comp["KMeans"])
    print("K-Medoids:", comp["KMedoids_Manhattan"])
