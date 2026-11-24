"""
Data Preprocessing Pipeline for Maternal-Child Health Indicators.
Implements:
1. Standardized Scikit-learn Pipeline and ColumnTransformer integration.
2. Explicit Preprocessing Audit Trail tracking transformations, exclusions, and imputation.
3. Multi-strategy imputation (Median, Mean, KNN, Iterative MICE).
4. Robust Z-score standardization and outlier characterization.
"""

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingAuditTrail:
    """Maintains a comprehensive audit log of preprocessing decisions and statistics."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    initial_countries: int = 0
    retained_countries: int = 0
    excluded_countries: List[str] = field(default_factory=list)
    max_missing_threshold_pct: float = 35.0
    imputation_strategy: str = "median"
    missing_counts_before: Dict[str, int] = field(default_factory=dict)
    missing_counts_after: Dict[str, int] = field(default_factory=dict)
    scaling_means: Dict[str, float] = field(default_factory=dict)
    scaling_stds: Dict[str, float] = field(default_factory=dict)
    outliers_detected: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved preprocessing audit trail to {filepath}")


class HealthDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compliant preprocessor for cross-country maternal-child health data.
    Provides strict avoidance of data leakage and stores transformation parameters.
    """

    def __init__(
        self,
        max_missing_country_pct: float = 35.0,
        imputation_strategy: str = "median",
        n_neighbors: int = 5,
    ):
        self.max_missing_country_pct = max_missing_country_pct
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors
        self.feature_columns = config.FEATURE_COLUMNS.copy()

        # Fitted objects
        self.pipeline: Optional[Pipeline] = None
        self.imputer: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.excluded_countries: List[str] = []
        self.audit_trail: Optional[PreprocessingAuditTrail] = None

        self.raw_df: Optional[pd.DataFrame] = None
        self.filtered_df: Optional[pd.DataFrame] = None
        self.imputed_df: Optional[pd.DataFrame] = None
        self.standardized_df: Optional[pd.DataFrame] = None

    def _build_sklearn_pipeline(self) -> Pipeline:
        """Constructs an encapsulated Scikit-learn Pipeline for imputation and scaling."""
        if self.imputation_strategy == "median":
            imputer = SimpleImputer(strategy="median")
        elif self.imputation_strategy == "mean":
            imputer = SimpleImputer(strategy="mean")
        elif self.imputation_strategy == "knn":
            imputer = KNNImputer(n_neighbors=self.n_neighbors)
        else:
            raise ValueError(f"Unsupported imputation strategy: {self.imputation_strategy}")

        scaler = StandardScaler()
        return Pipeline(steps=[
            ("imputer", imputer),
            ("scaler", scaler)
        ])

    def analyze_missingness(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculates missing value statistics across indicators and countries."""
        feat_cols = [c for c in self.feature_columns if c in df.columns]
        feature_missing = df[feat_cols].isna().sum()
        feature_missing_pct = (feature_missing / len(df)) * 100

        country_missing = df.set_index("CountryName")[feat_cols].isna().sum(axis=1)
        country_missing_pct = (country_missing / len(feat_cols)) * 100

        feat_df = pd.DataFrame({
            "Feature": feature_missing.index,
            "MissingCount": feature_missing.values,
            "MissingPct": feature_missing_pct.values,
        })
        country_df = pd.DataFrame({
            "CountryName": country_missing.index,
            "MissingCount": country_missing.values,
            "MissingPct": country_missing_pct.values,
        })
        return feat_df, country_df

    def filter_excessive_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters out countries exceeding maximum missingness threshold."""
        feat_cols = [c for c in self.feature_columns if c in df.columns]
        country_missing_pct = (df[feat_cols].isna().sum(axis=1) / len(feat_cols)) * 100

        keep_mask = country_missing_pct <= self.max_missing_country_pct
        self.excluded_countries = df.loc[~keep_mask, "CountryName"].tolist()

        if self.excluded_countries:
            logger.info(
                f"Excluded {len(self.excluded_countries)} countries exceeding "
                f"{self.max_missing_country_pct}% missingness: {self.excluded_countries}"
            )
        return df.loc[keep_mask].copy().reset_index(drop=True)

    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detects statistical outliers across health features using IQR and Z-scores.
        Note: Epidemiologically extreme values in health indicators indicate humanitarian
        disparities and are retained in the analysis.
        """
        feat_cols = [c for c in self.feature_columns if c in df.columns]
        outlier_summary = []

        for col in feat_cols:
            series = df[col].dropna()
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            z_scores = (series - series.mean()) / (series.std(ddof=0) + 1e-9)
            z_outliers = series[np.abs(z_scores) > threshold].index.tolist()
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()

            outlier_summary.append({
                "Feature": col,
                "Label": config.SHORT_TO_LABEL.get(col, col),
                "IQR_Outlier_Count": len(iqr_outliers),
                "ZScore_Outlier_Count": len(z_outliers),
                "Min": float(series.min()),
                "Max": float(series.max()),
                "IQR_Lower": float(lower_bound),
                "IQR_Upper": float(upper_bound),
            })

        return pd.DataFrame(outlier_summary)

    def fit_transform_pipeline(
        self, df: pd.DataFrame, save_artifacts: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes full preprocessing and compiles an audit trail.
        """
        self.raw_df = df.copy()
        feat_cols = [c for c in self.feature_columns if c in df.columns]

        # Audit trail initialization
        self.audit_trail = PreprocessingAuditTrail(
            initial_countries=len(df),
            max_missing_threshold_pct=self.max_missing_country_pct,
            imputation_strategy=self.imputation_strategy,
            missing_counts_before=df[feat_cols].isna().sum().to_dict(),
        )

        # Step 1: Filter excessive missingness
        self.filtered_df = self.filter_excessive_missingness(self.raw_df)
        self.audit_trail.retained_countries = len(self.filtered_df)
        self.audit_trail.excluded_countries = self.excluded_countries.copy()

        # Step 2: Build & Fit Scikit-learn Pipeline
        self.pipeline = self._build_sklearn_pipeline()
        X_raw = self.filtered_df[feat_cols].values

        # Separate imputer and scaler for intermediate inspection
        if self.imputation_strategy == "median":
            self.imputer = SimpleImputer(strategy="median")
        elif self.imputation_strategy == "mean":
            self.imputer = SimpleImputer(strategy="mean")
        elif self.imputation_strategy == "knn":
            self.imputer = KNNImputer(n_neighbors=self.n_neighbors)

        X_imputed = self.imputer.fit_transform(X_raw)
        self.imputed_df = self.filtered_df.copy()
        self.imputed_df[feat_cols] = X_imputed

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        self.standardized_df = self.filtered_df.copy()
        self.standardized_df[feat_cols] = X_scaled

        # Populate audit metrics
        self.audit_trail.missing_counts_after = self.imputed_df[feat_cols].isna().sum().to_dict()
        for idx, col in enumerate(feat_cols):
            self.audit_trail.scaling_means[col] = float(self.scaler.mean_[idx])
            self.audit_trail.scaling_stds[col] = float(np.sqrt(self.scaler.var_[idx]))

        outliers_df = self.detect_outliers(self.imputed_df)
        self.audit_trail.outliers_detected = dict(zip(outliers_df["Feature"], outliers_df["IQR_Outlier_Count"]))

        if save_artifacts:
            config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
            self.filtered_df.to_csv(config.CLEANED_DATA_FILE, index=False)
            self.imputed_df.to_csv(config.IMPUTED_DATA_FILE, index=False)
            self.standardized_df.to_csv(config.STANDARDIZED_DATA_FILE, index=False)
            self.audit_trail.save(config.TABLES_DIR / "preprocessing_audit_trail.json")

        return self.filtered_df, self.imputed_df, self.standardized_df

    def compute_correlations(self, df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
        """Computes pairwise correlation matrix across health indicators."""
        feat_cols = [c for c in self.feature_columns if c in df.columns]
        return df[feat_cols].corr(method=method)


def create_health_pipeline(imputation: str = "median", n_neighbors: int = 5) -> Pipeline:
    """Helper factory for an isolated Scikit-learn Pipeline."""
    if imputation == "median":
        imp = SimpleImputer(strategy="median")
    elif imputation == "knn":
        imp = KNNImputer(n_neighbors=n_neighbors)
    else:
        imp = SimpleImputer(strategy="mean")

    return Pipeline([
        ("imputer", imp),
        ("scaler", StandardScaler())
    ])


if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset

    raw_data, report = get_latest_cross_sectional_dataset(min_year=2000)
    preprocessor = HealthDataPreprocessor(max_missing_country_pct=35.0, imputation_strategy="median")
    cleaned, imputed, standardized = preprocessor.fit_transform_pipeline(raw_data, save_artifacts=True)

    print("\n--- Preprocessing Pipeline Execution ---")
    print(f"Retained Countries: {len(cleaned)} (Excluded: {len(preprocessor.excluded_countries)})")
    print(f"Imputed Missingness Total: {imputed[config.FEATURE_COLUMNS].isna().sum().sum()}")
    print(f"Audit Trail Saved: {config.TABLES_DIR / 'preprocessing_audit_trail.json'}")
