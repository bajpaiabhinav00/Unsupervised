"""
Data Schema and Quality Validation Engine for Maternal-Child Health Indicators.
Defines required fields, expected ranges, type contracts, and automated validation rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

import config


@dataclass
class ValidationReport:
    """Encapsulates results from schema, type, and range validation checks."""
    is_valid: bool
    total_records: int
    passed_records: int
    duplicate_records: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    out_of_range_details: Dict[str, int] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)


# Defined Indicator Permissible Ranges
INDICATOR_RANGES = {
    # Percentages [0, 100]
    "anc_4plus_coverage": (0.0, 100.0),
    "anc_skilled_provider": (0.0, 100.0),
    "skilled_birth_attendance": (0.0, 100.0),
    "institutional_delivery": (0.0, 100.0),
    "modern_contraceptive_prevalence": (0.0, 100.0),
    "dpt3_vaccination_coverage": (0.0, 100.0),
    "measles_vaccination_coverage": (0.0, 100.0),
    "basic_vaccination_coverage": (0.0, 100.0),
    "child_stunting_prevalence": (0.0, 100.0),
    "child_wasting_prevalence": (0.0, 100.0),
    "child_underweight_prevalence": (0.0, 100.0),
    "exclusive_breastfeeding_rate": (0.0, 100.0),
    "improved_drinking_water_access": (0.0, 100.0),
    "improved_sanitation_access": (0.0, 100.0),
    "female_secondary_education": (0.0, 100.0),
    # Rates per 1,000 live births [0, 500]
    "infant_mortality_rate": (0.0, 500.0),
    "under_five_mortality_rate": (0.0, 500.0),
}


class DataSchemaValidator:
    """
    Validates DHS datasets against structural schemas, bounds, and consistency constraints.
    """

    REQUIRED_RAW_COLUMNS = ["CountryName", "SurveyYear", "IndicatorId", "Value"]
    REQUIRED_CROSS_SECTION_COLUMNS = ["CountryName", "SurveyYear"]

    @classmethod
    def validate_raw_dataset(cls, df: pd.DataFrame) -> ValidationReport:
        """
        Validates raw long-format DHS indicator records before aggregation.
        """
        errors = []
        warnings = []
        out_of_range = {}

        if df is None or df.empty:
            return ValidationReport(
                is_valid=False,
                total_records=0,
                passed_records=0,
                duplicate_records=0,
                errors=["Dataset is empty or None."],
            )

        # 1. Required columns check
        missing_cols = [c for c in cls.REQUIRED_RAW_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns in raw dataset: {missing_cols}")

        # 2. Check for null identifiers
        null_countries = int(df["CountryName"].isna().sum()) if "CountryName" in df.columns else 0
        if null_countries > 0:
            errors.append(f"Found {null_countries} records with missing CountryName.")

        # 3. Check survey year validity
        if "SurveyYear" in df.columns:
            years = pd.to_numeric(df["SurveyYear"], errors="coerce")
            invalid_years = int(((years < 1980) | (years > 2030) | years.isna()).sum()) if not years.empty else 0
            if invalid_years > 0:
                warnings.append(f"Found {invalid_years} records with anomalous or missing SurveyYear (<1980 or >2030).")

        # 4. Check indicator values
        if "Value" in df.columns:
            vals = pd.to_numeric(df["Value"], errors="coerce")
            nan_vals = int(vals.isna().sum())
            if nan_vals > 0:
                warnings.append(f"Found {nan_vals} records with non-numeric or missing indicator values.")

            neg_vals = int((vals < 0).sum())
            if neg_vals > 0:
                errors.append(f"Found {neg_vals} negative indicator values (statistically impossible for health rates).")

        # 5. Duplicate check
        dup_subset = ["CountryName", "SurveyYear", "IndicatorId", "ByVariableLabel", "CharacteristicLabel"]
        present_subset = [c for c in dup_subset if c in df.columns]
        dup_count = int(df.duplicated(subset=present_subset).sum())
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate survey records on keys {present_subset}.")

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            total_records=len(df),
            passed_records=len(df) - len(errors),
            duplicate_records=dup_count,
            errors=errors,
            warnings=warnings,
            out_of_range_details=out_of_range,
        )

    @classmethod
    def validate_cross_sectional_dataset(cls, df: pd.DataFrame) -> ValidationReport:
        """
        Validates wide-format cross-sectional dataset before machine learning modeling.
        """
        errors = []
        warnings = []
        out_of_range = {}

        if df is None or df.empty:
            return ValidationReport(
                is_valid=False,
                total_records=0,
                passed_records=0,
                duplicate_records=0,
                errors=["Cross-sectional dataset is empty or None."],
            )

        # 1. Required identifier columns
        missing_id_cols = [c for c in cls.REQUIRED_CROSS_SECTION_COLUMNS if c not in df.columns]
        if missing_id_cols:
            errors.append(f"Missing required identifier columns: {missing_id_cols}")

        # 2. Check duplicate countries
        if "CountryName" in df.columns:
            dup_countries = int(df["CountryName"].duplicated().sum())
            if dup_countries > 0:
                errors.append(f"Found {dup_countries} duplicate country entries in cross-sectional matrix.")

        # 3. Check indicator ranges
        for ind_name, (min_val, max_val) in INDICATOR_RANGES.items():
            if ind_name in df.columns:
                series = pd.to_numeric(df[ind_name], errors="coerce").dropna()
                invalid_mask = (series < min_val) | (series > max_val)
                violating_count = int(invalid_mask.sum())
                if violating_count > 0:
                    out_of_range[ind_name] = violating_count
                    errors.append(
                        f"Indicator '{ind_name}' has {violating_count} values outside allowable range [{min_val}, {max_val}]."
                    )

        # 4. Check for constant-valued features (zero variance)
        for col in config.FEATURE_COLUMNS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) > 1 and series.std(ddof=0) == 0.0:
                    warnings.append(f"Feature '{col}' exhibits zero variance (constant value).")

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            total_records=len(df),
            passed_records=len(df),
            duplicate_records=df["CountryName"].duplicated().sum() if "CountryName" in df.columns else 0,
            errors=errors,
            warnings=warnings,
            out_of_range_details=out_of_range,
            summary_stats={
                "countries_count": len(df),
                "features_present": [c for c in config.FEATURE_COLUMNS if c in df.columns],
            },
        )
