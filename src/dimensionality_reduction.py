"""
Dimensionality Reduction Module for Maternal-Child Health Indicators.
Implements Principal Component Analysis (PCA) on standardized data:
1. Explained variance and cumulative explained variance.
2. 2D and 3D projection of country observations.
3. Feature loadings and variable contribution analysis (biplots).
4. Academic caveat documentation: PCA serves as linear dimensionality reduction
   and visualization, not as definitive proof of cluster validity.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HealthPCA:
    """
    Performs Principal Component Analysis on standardized maternal-child health indicators.
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None
        self.loadings_df_: Optional[pd.DataFrame] = None
        self.feature_names_: List[str] = []

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> "HealthPCA":
        """
        Fits PCA model on standardized feature matrix X.
        """
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif X.shape[1] == len(config.FEATURE_COLUMNS):
            self.feature_names_ = config.FEATURE_COLUMNS.copy()
        else:
            self.feature_names_ = [f"Feature_{i+1}" for i in range(X.shape[1])]

        n_comp = self.n_components or min(X.shape[0], X.shape[1])

        self.pca = PCA(n_components=n_comp, random_state=config.RANDOM_SEED)
        self.pca.fit(X)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # Calculate loadings: eigenvectors scaled by the square root of eigenvalues
        # Loadings represent the Pearson correlation between original variables and principal components
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        comp_labels = [f"PC{i+1}" for i in range(loadings.shape[1])]

        self.loadings_df_ = pd.DataFrame(
            loadings,
            index=self.feature_names_,
            columns=comp_labels,
        )

        logger.info(
            f"Fitted PCA ({n_comp} components). PC1 explains {self.explained_variance_ratio_[0]*100:.1f}%, "
            f"PC2 explains {self.explained_variance_ratio_[1]*100:.1f}%, "
            f"Cumulative (PC1+PC2): {self.cumulative_variance_ratio_[1]*100:.1f}%"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects standardized feature matrix into principal component coordinates.
        """
        if self.pca is None:
            raise ValueError("PCA model has not been fitted yet.")
        return self.pca.transform(X)

    def fit_transform(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        self.fit(X, feature_names=feature_names)
        return self.transform(X)

    def get_variance_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing explained variance per component.
        """
        if self.pca is None:
            raise ValueError("PCA model has not been fitted yet.")

        n_comp = len(self.explained_variance_ratio_)
        summary = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(n_comp)],
            "Eigenvalue": self.pca.explained_variance_,
            "Explained_Variance_Ratio": self.explained_variance_ratio_,
            "Cumulative_Variance_Ratio": self.cumulative_variance_ratio_,
            "Explained_Variance_Pct": self.explained_variance_ratio_ * 100,
            "Cumulative_Variance_Pct": self.cumulative_variance_ratio_ * 100,
        })
        return summary

    def get_top_contributors(self, component: int = 1, top_n: int = 5) -> pd.DataFrame:
        """
        Identifies features with highest absolute loadings on a given principal component.
        """
        if self.loadings_df_ is None:
            raise ValueError("PCA model has not been fitted yet.")

        col_name = f"PC{component}"
        series = self.loadings_df_[col_name]
        sorted_df = pd.DataFrame({
            "Feature": series.index,
            "Label": [config.SHORT_TO_LABEL.get(f, f) for f in series.index],
            "Category": [config.SHORT_TO_CATEGORY.get(f, "Other") for f in series.index],
            "Loading": series.values,
            "Absolute_Loading": np.abs(series.values),
        }).sort_values(by="Absolute_Loading", ascending=False).reset_index(drop=True)

        return sorted_df.head(top_n)

    def create_projection_dataframe(
        self,
        X: np.ndarray,
        country_names: List[str],
        cluster_labels: Optional[np.ndarray] = None,
        survey_years: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Creates a structured DataFrame containing PC coordinates along with country metadata and cluster labels.
        """
        X_pca = self.transform(X)
        df = pd.DataFrame({
            "CountryName": country_names,
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "PC3": X_pca[:, 2] if X_pca.shape[1] >= 3 else 0.0,
        })

        if cluster_labels is not None:
            df["Cluster"] = cluster_labels
            df["Cluster_Label"] = [f"Cluster {c+1}" for c in cluster_labels]

        if survey_years is not None:
            df["SurveyYear"] = survey_years

        return df


if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset
    from src.preprocessing import HealthDataPreprocessor
    from sklearn.cluster import KMeans

    raw_data, _ = get_latest_cross_sectional_dataset(min_year=2000)
    preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
    cleaned, imputed, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=False)

    X = standardized[config.FEATURE_COLUMNS].values
    pca_model = HealthPCA()
    X_pca = pca_model.fit_transform(X, feature_names=config.FEATURE_COLUMNS)

    print("\n--- PCA Variance Summary (First 5 PCs) ---")
    print(pca_model.get_variance_summary().head(5))

    print("\n--- Top Contributors to PC1 ---")
    print(pca_model.get_top_contributors(component=1, top_n=5))

    print("\n--- Top Contributors to PC2 ---")
    print(pca_model.get_top_contributors(component=2, top_n=5))
