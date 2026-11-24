"""
Clustering Evaluation and Validation Metrics Module.
Calculates internal validation indices (WCSS, Silhouette, Davies-Bouldin, Calinski-Harabasz)
and external agreement indices (Adjusted Rand Index, Normalized Mutual Information)
with support for both Euclidean and Manhattan distance metrics.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances


def calculate_wcss(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Computes the Within-Cluster Sum of Squares (WCSS) / Inertia:
    WCSS = sum_{k=1}^K sum_{i in C_k} ||x_i - mu_k||^2
    """
    wcss = 0.0
    for k in np.unique(labels):
        cluster_points = X[labels == k]
        center = centers[k]
        wcss += np.sum((cluster_points - center) ** 2)
    return float(wcss)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    centers: Optional[np.ndarray] = None,
    metric: str = "euclidean",
) -> Dict[str, Union[float, int, List[int]]]:
    """
    Evaluates a single clustering solution across standard internal validation metrics.

    Args:
        X: Standardized feature matrix (n_samples, n_features).
        labels: Cluster assignment vector (n_samples,).
        centers: Cluster centroids or medoids (k, n_features), optional.
        metric: 'euclidean' or 'manhattan' distance for silhouette calculation.

    Returns:
        Dict of evaluation metrics: wcss, silhouette_avg, davies_bouldin,
        calinski_harabasz, cluster_sizes, n_clusters.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Cluster sizes
    cluster_sizes = [int(np.sum(labels == k)) for k in unique_labels]

    # Silhouette score (undefined if n_clusters <= 1 or n_clusters >= n_samples)
    if 1 < n_clusters < len(X):
        sil = float(silhouette_score(X, labels, metric=metric))
        sil_samples = silhouette_samples(X, labels, metric=metric)
    else:
        sil = float("nan")
        sil_samples = np.array([])

    # Davies-Bouldin index (lower is better)
    if 1 < n_clusters < len(X):
        db = float(davies_bouldin_score(X, labels))
    else:
        db = float("nan")

    # Calinski-Harabasz score (higher is better)
    if 1 < n_clusters < len(X):
        ch = float(calinski_harabasz_score(X, labels))
    else:
        ch = float("nan")

    # WCSS / Inertia
    if centers is not None:
        wcss = calculate_wcss(X, labels, centers)
    else:
        # Approximate centers from cluster means
        approx_centers = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
        wcss = calculate_wcss(X, labels, approx_centers)

    return {
        "n_clusters": n_clusters,
        "wcss": wcss,
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "cluster_sizes": cluster_sizes,
        "metric": metric,
    }


def compare_partitions_ari(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Computes the Adjusted Rand Index (ARI) to quantify agreement between two clustering solutions:
    ARI = (Index - ExpectedIndex) / (MaxIndex - ExpectedIndex)
    Ranges from -1 to 1 (1 = perfect agreement, 0 = random agreement).
    """
    return float(adjusted_rand_score(labels_a, labels_b))


def compare_partitions_nmi(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Computes the Normalized Mutual Information (NMI) between two cluster assignments.
    Ranges from 0 to 1 (1 = identical information).
    """
    return float(normalized_mutual_info_score(labels_a, labels_b))


if __name__ == "__main__":
    X_test = np.random.randn(50, 4)
    labels_test = np.random.randint(0, 3, size=50)
    res = evaluate_clustering(X_test, labels_test, metric="euclidean")
    print("Evaluation test output:", res)
