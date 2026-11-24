"""
Data Ingestion and Engineering Module for Maternal-Child Health Indicators.
Supports multi-format data ingestion (CSV, Excel, JSON, Parquet),
automated schema validation, DHS survey-weight preservation, and
methodologically sound cross-sectional harmonization.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config
from src.schema import DataSchemaValidator, ValidationReport

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def read_data_file(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Centralized file reader supporting CSV, Excel (.xlsx, .xls), JSON, and Parquet.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Requested dataset file does not exist: {path}")

    ext = path.suffix.lower()
    logger.info(f"Ingesting dataset from '{path.name}' (format: {ext})")

    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext == ".parquet":
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: .csv, .xlsx, .xls, .json, .parquet")


def load_raw_data(
    raw_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    validate_schema: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[ValidationReport]]:
    """
    Loads raw DHS indicator records and survey metadata from disk with automated schema checks.
    """
    raw_file = raw_path or config.RAW_DATA_FILE
    meta_file = metadata_path or config.SURVEY_META_FILE

    raw_df = read_data_file(raw_file)
    meta_df = read_data_file(meta_file)

    report = None
    if validate_schema:
        report = DataSchemaValidator.validate_raw_dataset(raw_df)
        if not report.is_valid:
            logger.warning(f"Raw dataset validation failed with {len(report.errors)} errors: {report.errors}")
        else:
            logger.info("Raw DHS dataset passed schema validation.")

    return raw_df, meta_df, report


def select_standard_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters raw DHS indicator records to retain standard national reporting definitions.

    Methodological filtering rules:
    1. Maternal health indicators (ANC, delivery): 5-year reference period preceding survey.
    2. Child immunization indicators (DPT, Measles, Basic): 12-23 months cohort, either source.
    3. Child mortality: 5-year retrospective period.
    4. Child nutrition & WASH: Total national population.
    """
    df = raw_df.copy()
    df["SurveyYear"] = pd.to_numeric(df["SurveyYear"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    # Drop records with invalid value or year
    df = df.dropna(subset=["Value", "SurveyYear"])

    standard_rows = []
    groupby_cols = ["CountryName", "SurveyYear", "IndicatorId"]

    for (country, year, ind_id), group in df.groupby(groupby_cols):
        if len(group) == 1:
            standard_rows.append(group.iloc[0])
            continue

        # Priority 1: Five-year reference window (maternal indicators)
        subset = group[group["ByVariableLabel"].str.contains("Five years", na=False, case=False)]
        if len(subset) > 0:
            standard_rows.append(subset.iloc[0])
            continue

        # Priority 2: 12-23 months cohort (vaccinations)
        subset = group[group["ByVariableLabel"].str.contains("12-23", na=False)]
        if len(subset) > 0:
            standard_rows.append(subset.iloc[0])
            continue

        # Priority 3: Total population characteristic
        subset = group[group["CharacteristicLabel"].str.contains("Total", na=False, case=False)]
        if len(subset) > 0:
            standard_rows.append(subset.iloc[0])
            continue

        # Priority 4: Either source (vaccination card or maternal recall)
        subset = group[group["CharacteristicLabel"].str.contains("Either source", na=False, case=False)]
        if len(subset) > 0:
            standard_rows.append(subset.iloc[0])
            continue

        # Fallback to first row
        standard_rows.append(group.iloc[0])

    result_df = pd.DataFrame(standard_rows).reset_index(drop=True)
    logger.info(f"Filtered {len(df)} raw rows down to {len(result_df)} standardized records")
    return result_df


def compute_weighted_country_aggregates(
    microdata_df: pd.DataFrame,
    value_column: str,
    weight_column: str,
    strata_columns: Optional[List[str]] = None,
) -> Tuple[float, float, float]:
    """
    Computes survey-weighted mean, effective sample size, and standard error
    accounting for DHS sampling weights (w_i = sampling_weight / 1,000,000).

    Returns:
        (weighted_mean, effective_sample_size, standard_error)
    """
    df = microdata_df.dropna(subset=[value_column, weight_column]).copy()
    if df.empty:
        return np.nan, 0.0, np.nan

    weights = df[weight_column].values.astype(float)
    values = df[value_column].values.astype(float)

    total_weight = np.sum(weights)
    if total_weight <= 0:
        return np.nan, 0.0, np.nan

    # Survey-weighted mean
    weighted_mean = float(np.sum(weights * values) / total_weight)

    # Kish's Design Effect / Effective Sample Size: n_eff = (sum w)^2 / sum(w^2)
    sum_w_sq = np.sum(weights ** 2)
    n_eff = float((total_weight ** 2) / (sum_w_sq + 1e-12))

    # Weighted sample variance
    weighted_var = float(np.sum(weights * (values - weighted_mean) ** 2) / total_weight)
    std_error = float(np.sqrt(weighted_var / (n_eff + 1e-12)))

    return weighted_mean, n_eff, std_error


def get_latest_cross_sectional_dataset(
    min_year: int = 2000,
    raw_path: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> Tuple[pd.DataFrame, Optional[ValidationReport]]:
    """
    Constructs a cross-sectional dataset containing the most recent standard DHS survey
    for each country within [min_year, present], with complete schema validation.
    """
    raw_df, meta_df, _ = load_raw_data(raw_path=raw_path, validate_schema=validate)
    std_df = select_standard_records(raw_df)

    # Filter to target indicator IDs
    target_indicators = list(config.INDICATORS_CONFIG.keys())
    std_df = std_df[std_df["IndicatorId"].isin(target_indicators)].copy()

    # Determine latest survey year per country
    latest_years = (
        std_df.groupby("CountryName")["SurveyYear"]
        .max()
        .reset_index()
        .rename(columns={"SurveyYear": "LatestSurveyYear"})
    )

    # Merge and filter
    merged = std_df.merge(
        latest_years,
        left_on=["CountryName", "SurveyYear"],
        right_on=["CountryName", "LatestSurveyYear"]
    )
    merged = merged[merged["SurveyYear"] >= min_year].copy()

    # Pivot into wide format
    pivoted = merged.pivot_table(
        index=["CountryName", "DHS_CountryCode", "SurveyYear"],
        columns="IndicatorId",
        values="Value",
        aggfunc="mean"
    ).reset_index()

    # Rename indicator IDs to clean feature names
    pivoted = pivoted.rename(columns=config.ID_TO_SHORT)

    # Validate resulting cross-section
    report = None
    if validate:
        report = DataSchemaValidator.validate_cross_sectional_dataset(pivoted)
        if not report.is_valid:
            logger.warning(f"Cross-sectional validation warnings/errors: {report.errors}")
        else:
            logger.info(f"Cross-sectional dataset passed validation ({len(pivoted)} countries).")

    return pivoted, report


def export_multi_format_samples(raw_df: pd.DataFrame, processed_df: pd.DataFrame):
    """
    Exports datasets into Excel and JSON formats to support multi-format ingestion testing.
    """
    excel_path = config.RAW_DATA_DIR / "dhs_maternal_child_indicators_raw.xlsx"
    if not excel_path.exists():
        raw_df.head(1000).to_excel(excel_path, index=False)
        logger.info(f"Exported sample Excel dataset to {excel_path}")


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Computes summary metadata, dimensions, missingness, and descriptive statistics.
    """
    feature_cols = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    summary = {
        "num_countries": len(df),
        "survey_year_min": int(df["SurveyYear"].min()) if "SurveyYear" in df.columns else None,
        "survey_year_max": int(df["SurveyYear"].max()) if "SurveyYear" in df.columns else None,
        "survey_year_median": float(df["SurveyYear"].median()) if "SurveyYear" in df.columns else None,
        "indicators_count": len(feature_cols),
        "total_cells": len(df) * len(feature_cols),
        "total_missing": int(df[feature_cols].isna().sum().sum()),
        "missing_percentage": float((df[feature_cols].isna().sum().sum() / (len(df) * len(feature_cols))) * 100),
        "missing_by_feature": df[feature_cols].isna().sum().to_dict(),
        "stats": df[feature_cols].describe().to_dict(),
    }
    return summary


if __name__ == "__main__":
    df, report = get_latest_cross_sectional_dataset(min_year=2000)
    print("\n--- Cross-Sectional Dataset Ingestion Report ---")
    print(f"Countries: {len(df)}")
    if report:
        print(f"Valid: {report.is_valid}, Errors: {len(report.errors)}, Warnings: {len(report.warnings)}")
