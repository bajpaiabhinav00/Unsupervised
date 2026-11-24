"""
Healthcare Gap Detection Engine.

Automatically identifies maternal and child healthcare indicators that deviate from
documented WHO/UNICEF reference benchmarks. Produces normalised gap scores, region-level
summaries, and structured heatmap data for the Streamlit dashboard.

Benchmarks are sourced from:
 - WHO Global Health Observatory (https://www.who.int/data/gho)
 - UNICEF State of the World's Children 2023 (https://www.unicef.org/sowc)
 - SDG targets 3.1, 3.2, 3.8 (UN 2030 Agenda)
 - WHO Global Immunization Vision and Strategy (GIVS)
 - WHO/UNICEF Joint Monitoring Programme (JMP) for Water and Sanitation

NOTE: These are reference benchmarks for analytical comparison only.
Deviation from a benchmark does not automatically imply a confirmed adverse outcome;
it flags a potential gap warranting further investigation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralised benchmark registry
# Every entry must document its source, year and unit.
# ---------------------------------------------------------------------------
BENCHMARKS: Dict[str, Dict] = {
    # ---- Maternal Health ------------------------------------------------
    "anc_4plus_coverage": {
        "value": 80.0,
        "unit": "% of women (15–49)",
        "higher_is_better": True,
        "source": "WHO ANC Recommendation (2016); SDG 3.1 alignment",
        "year": "2016+",
        "explanation": (
            "WHO recommends a minimum of 8 antenatal care contacts; 4+ ANC visits is the "
            "widely reported proxy. The 80% coverage target aligns with SDG 3.8 UHC goals."
        ),
        "category": "Maternal Health",
        "domain": "maternal",
    },
    "anc_skilled_provider": {
        "value": 90.0,
        "unit": "% of women (15–49)",
        "higher_is_better": True,
        "source": "WHO Skilled Birth Attendant Framework (2004); SDG 3.1",
        "year": "2015+",
        "explanation": (
            "WHO guidelines recommend that all pregnant women receive ANC from a skilled "
            "health worker. 90% coverage is widely used as a high-performance threshold."
        ),
        "category": "Maternal Health",
        "domain": "maternal",
    },
    "skilled_birth_attendance": {
        "value": 90.0,
        "unit": "% of live births",
        "higher_is_better": True,
        "source": "WHO Skilled Birth Attendant Framework (2004); SDG 3.1",
        "year": "2015+",
        "explanation": (
            "Skilled attendance at birth is a key tracer for maternal health system coverage. "
            "The 90% threshold is used by WHO and UNICEF in progress assessments."
        ),
        "category": "Maternal Health",
        "domain": "maternal",
    },
    "institutional_delivery": {
        "value": 90.0,
        "unit": "% of live births",
        "higher_is_better": True,
        "source": "WHO Making Every Baby Count (2012); SDG 3.1",
        "year": "2015+",
        "explanation": (
            "Facility-based delivery is closely associated with reduced maternal and neonatal "
            "mortality. 90% is the high-coverage benchmark used in UNICEF annual reports."
        ),
        "category": "Maternal Health",
        "domain": "maternal",
    },
    "modern_contraceptive_prevalence": {
        "value": 75.0,
        "unit": "% married women (15–49)",
        "higher_is_better": True,
        "source": "SDG 3.7; FP2030 Global Partnership Target",
        "year": "2015+",
        "explanation": (
            "FP2030 and SDG 3.7 emphasise universal access to sexual and reproductive health "
            "services. 75% modern CPR is the aspirational target used in FP impact analyses."
        ),
        "category": "Maternal Health",
        "domain": "maternal",
    },
    # ---- Child Mortality ------------------------------------------------
    "infant_mortality_rate": {
        "value": 12.0,
        "unit": "deaths per 1,000 live births",
        "higher_is_better": False,
        "source": "SDG 3.2: End preventable deaths of newborns & children under 5",
        "year": "2030 target",
        "explanation": (
            "SDG 3.2 calls for all countries to reduce neonatal mortality to ≤12/1,000 and "
            "under-5 mortality to ≤25/1,000 by 2030. IMR ≤12 is used as the infant benchmark."
        ),
        "category": "Child Mortality",
        "domain": "child",
    },
    "under_five_mortality_rate": {
        "value": 25.0,
        "unit": "deaths per 1,000 live births",
        "higher_is_better": False,
        "source": "SDG 3.2 (UN 2030 Agenda); UNICEF SOWC 2023",
        "year": "2030 target",
        "explanation": (
            "SDG 3.2 target is U5MR ≤25/1,000 live births by 2030. Countries above this "
            "threshold have a measurable gap relative to the global sustainability goal."
        ),
        "category": "Child Mortality",
        "domain": "child",
    },
    # ---- Child Immunization --------------------------------------------
    "dpt3_vaccination_coverage": {
        "value": 90.0,
        "unit": "% children (12–23 months)",
        "higher_is_better": True,
        "source": "WHO GIVS; Immunization Agenda 2030 (IA2030)",
        "year": "2021+",
        "explanation": (
            "Immunization Agenda 2030 targets ≥90% national DPT3 coverage and ≥80% "
            "in every district. 90% is the global reference threshold."
        ),
        "category": "Child Immunization",
        "domain": "child",
    },
    "measles_vaccination_coverage": {
        "value": 95.0,
        "unit": "% children (12–23 months)",
        "higher_is_better": True,
        "source": "WHO Measles Elimination Strategy; IA2030",
        "year": "2021+",
        "explanation": (
            "Achieving herd immunity against measles requires ≥95% coverage. "
            "Countries below this threshold are analytically flagged as at-risk."
        ),
        "category": "Child Immunization",
        "domain": "child",
    },
    "basic_vaccination_coverage": {
        "value": 90.0,
        "unit": "% children (12–23 months)",
        "higher_is_better": True,
        "source": "WHO GIVS; IA2030 coverage targets",
        "year": "2021+",
        "explanation": (
            "Full basic immunization (BCG + measles + 3× DPT + 3× polio) coverage of 90% "
            "is the IA2030 global ambition for national-level performance."
        ),
        "category": "Child Immunization",
        "domain": "child",
    },
    # ---- Child Nutrition -----------------------------------------------
    "child_stunting_prevalence": {
        "value": 10.0,
        "unit": "% children under 5",
        "higher_is_better": False,
        "source": "WHO Global Nutrition Targets 2025; WHA Resolution 65.6",
        "year": "2025 target",
        "explanation": (
            "WHO WHA 2012 set a target to reduce stunting prevalence to <40% of 2012 "
            "levels by 2025. <10% is used as the threshold associated with low stunting "
            "burden in comparative analyses."
        ),
        "category": "Child Nutrition",
        "domain": "child",
    },
    "child_wasting_prevalence": {
        "value": 5.0,
        "unit": "% children under 5",
        "higher_is_better": False,
        "source": "WHO Global Nutrition Targets 2025; UNICEF–WHO–WBG Joint Estimates",
        "year": "2025 target",
        "explanation": (
            "WHO considers wasting prevalence ≥5% a public-health concern. "
            "The 5% threshold is widely used as the reference boundary in nutrition surveillance."
        ),
        "category": "Child Nutrition",
        "domain": "child",
    },
    "child_underweight_prevalence": {
        "value": 10.0,
        "unit": "% children under 5",
        "higher_is_better": False,
        "source": "WHO Global Nutrition Targets 2025; WHA Resolution 65.6",
        "year": "2025 target",
        "explanation": (
            "WHO underweight reduction target aligns with WHA 2012 goals. "
            "<10% prevalence is a marker of low burden in cross-country comparisons."
        ),
        "category": "Child Nutrition",
        "domain": "child",
    },
    "exclusive_breastfeeding_rate": {
        "value": 70.0,
        "unit": "% infants under 6 months",
        "higher_is_better": True,
        "source": "WHO Global Nutrition Targets 2025; WHA Resolution 65.6",
        "year": "2025 target",
        "explanation": (
            "WHA 2012 set a target to increase exclusive breastfeeding to ≥50% by 2025. "
            "WHO advocates for ≥70% as an ambitious high-performance reference."
        ),
        "category": "Child Nutrition",
        "domain": "child",
    },
    # ---- Socioeconomic & WASH -----------------------------------------
    "improved_drinking_water_access": {
        "value": 95.0,
        "unit": "% households",
        "higher_is_better": True,
        "source": "SDG 6.1; WHO/UNICEF JMP 2023",
        "year": "2030 target",
        "explanation": (
            "SDG 6.1 calls for universal and equitable access to safe drinking water. "
            "95% household coverage is used as the near-universal benchmark."
        ),
        "category": "Socioeconomic & WASH",
        "domain": "child",
    },
    "improved_sanitation_access": {
        "value": 90.0,
        "unit": "% households",
        "higher_is_better": True,
        "source": "SDG 6.2; WHO/UNICEF JMP 2023",
        "year": "2030 target",
        "explanation": (
            "SDG 6.2 targets universal sanitation access. 90% improved-facility coverage "
            "is the high-performance benchmark used in JMP regional progress reports."
        ),
        "category": "Socioeconomic & WASH",
        "domain": "child",
    },
    "female_secondary_education": {
        "value": 80.0,
        "unit": "% women (15–49)",
        "higher_is_better": True,
        "source": "SDG 4.1; UNESCO Education for All Framework",
        "year": "2015+",
        "explanation": (
            "Female secondary education is a strong upstream determinant of maternal "
            "and child health. 80% attainment is used as a high-coverage reference "
            "reflecting SDG 4.1 universal secondary completion aspirations."
        ),
        "category": "Socioeconomic & WASH",
        "domain": "maternal",
    },
}

# Domain membership lists
MATERNAL_INDICATORS = [k for k, v in BENCHMARKS.items() if v["domain"] == "maternal"]
CHILD_INDICATORS = [k for k, v in BENCHMARKS.items() if v["domain"] == "child"]
ALL_INDICATORS = list(BENCHMARKS.keys())


# ---------------------------------------------------------------------------
# Core gap calculation functions
# ---------------------------------------------------------------------------

def calculate_indicator_gap(
    observed: float,
    benchmark: float,
    higher_is_better: bool,
) -> float:
    """
    Compute the normalised gap between an observed value and its benchmark.

    For higher-is-better indicators (e.g. vaccination coverage):
        gap = max(0, (benchmark - observed) / benchmark)

    For lower-is-better indicators (e.g. child mortality):
        gap = max(0, (observed - benchmark) / benchmark)

    Returns a value in [0, 1] where 0 means at or beyond the benchmark
    and 1 means the indicator is at zero performance relative to the benchmark.

    Args:
        observed: Observed country/region value.
        benchmark: Reference benchmark value.
        higher_is_better: Direction of the indicator.

    Returns:
        Normalised gap score in [0, 1]. NaN if inputs are invalid.
    """
    if pd.isna(observed) or pd.isna(benchmark):
        return float("nan")
    if benchmark == 0:
        logger.warning("Zero benchmark encountered — returning NaN to avoid division by zero.")
        return float("nan")

    if higher_is_better:
        gap = max(0.0, (benchmark - observed) / benchmark)
    else:
        gap = max(0.0, (observed - benchmark) / benchmark)

    return round(float(gap), 6)


def calculate_region_gaps(
    df: pd.DataFrame,
    benchmarks: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Compute indicator-level gap scores for every country/region in the dataframe.

    Args:
        df: Wide-format dataframe with one row per country and indicator columns.
        benchmarks: Optional custom benchmark dict (defaults to module-level BENCHMARKS).

    Returns:
        DataFrame with the same index as `df`, columns are indicator gap scores.
        Metadata columns (CountryName, DHS_CountryCode, SurveyYear) are preserved.
    """
    bm = benchmarks or BENCHMARKS
    meta_cols = [c for c in ["CountryName", "DHS_CountryCode", "SurveyYear"] if c in df.columns]

    rows = []
    for _, row in df.iterrows():
        gap_row: Dict = {c: row[c] for c in meta_cols}
        for indicator, spec in bm.items():
            if indicator in df.columns:
                gap_row[f"{indicator}_gap"] = calculate_indicator_gap(
                    observed=row[indicator],
                    benchmark=spec["value"],
                    higher_is_better=spec["higher_is_better"],
                )
            else:
                gap_row[f"{indicator}_gap"] = float("nan")
        rows.append(gap_row)

    gaps_df = pd.DataFrame(rows)
    if meta_cols and len(gaps_df) > 0:
        gaps_df = gaps_df.set_index(meta_cols[0]) if "CountryName" in meta_cols else gaps_df
        gaps_df = gaps_df.reset_index()

    logger.info(
        f"Gap detection: computed gaps for {len(gaps_df)} regions across "
        f"{len(bm)} benchmarked indicators."
    )
    return gaps_df


def calculate_maternal_gap_summary(
    gaps_df: pd.DataFrame,
    benchmarks: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Returns a region-level summary of maternal indicator gaps.

    Columns: CountryName + one gap column per maternal indicator + mean_maternal_gap.
    """
    bm = benchmarks or BENCHMARKS
    maternal_gap_cols = [f"{ind}_gap" for ind in MATERNAL_INDICATORS if f"{ind}_gap" in gaps_df.columns]
    meta_cols = [c for c in ["CountryName", "DHS_CountryCode", "SurveyYear"] if c in gaps_df.columns]

    summary = gaps_df[meta_cols + maternal_gap_cols].copy()
    summary["mean_maternal_gap"] = summary[maternal_gap_cols].mean(axis=1, skipna=True)
    return summary.sort_values("mean_maternal_gap", ascending=False).reset_index(drop=True)


def calculate_child_gap_summary(
    gaps_df: pd.DataFrame,
    benchmarks: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Returns a region-level summary of child indicator gaps.

    Columns: CountryName + one gap column per child indicator + mean_child_gap.
    """
    bm = benchmarks or BENCHMARKS
    child_gap_cols = [f"{ind}_gap" for ind in CHILD_INDICATORS if f"{ind}_gap" in gaps_df.columns]
    meta_cols = [c for c in ["CountryName", "DHS_CountryCode", "SurveyYear"] if c in gaps_df.columns]

    summary = gaps_df[meta_cols + child_gap_cols].copy()
    summary["mean_child_gap"] = summary[child_gap_cols].mean(axis=1, skipna=True)
    return summary.sort_values("mean_child_gap", ascending=False).reset_index(drop=True)


def generate_gap_heatmap_data(
    gaps_df: pd.DataFrame,
    indicators: Optional[List[str]] = None,
    benchmarks: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Returns a tidy matrix suitable for Seaborn / Plotly heatmap rendering.

    Rows = countries, Columns = selected indicators (gap scores 0–1).
    """
    bm = benchmarks or BENCHMARKS
    selected = indicators or list(bm.keys())
    gap_cols = [f"{ind}_gap" for ind in selected if f"{ind}_gap" in gaps_df.columns]

    country_col = "CountryName" if "CountryName" in gaps_df.columns else gaps_df.columns[0]
    heatmap = gaps_df[[country_col] + gap_cols].copy()
    heatmap = heatmap.set_index(country_col)
    # Rename columns to human-readable labels
    rename_map = {
        f"{ind}_gap": bm[ind]["category"] + " – " + ind.replace("_", " ").title()
        for ind in selected
        if f"{ind}_gap" in gaps_df.columns
    }
    heatmap = heatmap.rename(columns=rename_map)
    return heatmap


def get_top_gap_indicators(
    gaps_df: pd.DataFrame,
    n: int = 5,
    benchmarks: Optional[Dict[str, Dict]] = None,
) -> pd.Series:
    """
    Returns the n indicators with the highest mean gap across all regions.
    """
    bm = benchmarks or BENCHMARKS
    gap_cols = [f"{ind}_gap" for ind in bm if f"{ind}_gap" in gaps_df.columns]
    mean_gaps = gaps_df[gap_cols].mean(skipna=True).sort_values(ascending=False)
    return mean_gaps.head(n)


def get_benchmark_table(benchmarks: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
    """
    Returns a human-readable table of all benchmarks with metadata.
    """
    bm = benchmarks or BENCHMARKS
    rows = []
    for indicator, spec in bm.items():
        rows.append({
            "Indicator": indicator.replace("_", " ").title(),
            "Short Name": indicator,
            "Reference Value": spec["value"],
            "Unit": spec["unit"],
            "Direction": "Higher is better" if spec["higher_is_better"] else "Lower is better",
            "Category": spec.get("category", ""),
            "Domain": spec.get("domain", ""),
            "Source": spec["source"],
            "Year": spec["year"],
            "Explanation": spec["explanation"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset

    df, _ = get_latest_cross_sectional_dataset(min_year=2000)
    print(f"Loaded {len(df)} countries.")

    gaps = calculate_region_gaps(df)
    print(f"\nGap DataFrame shape: {gaps.shape}")
    print(gaps[["CountryName", "anc_4plus_coverage_gap", "under_five_mortality_rate_gap"]].head(10))

    maternal_summary = calculate_maternal_gap_summary(gaps)
    print(f"\nTop 5 maternal gap countries:")
    print(maternal_summary[["CountryName", "mean_maternal_gap"]].head(5))

    child_summary = calculate_child_gap_summary(gaps)
    print(f"\nTop 5 child gap countries:")
    print(child_summary[["CountryName", "mean_child_gap"]].head(5))

    top_gaps = get_top_gap_indicators(gaps)
    print(f"\nTop 5 indicators by mean gap:")
    print(top_gaps)

    bm_table = get_benchmark_table()
    print(f"\nBenchmark table ({len(bm_table)} entries):")
    print(bm_table[["Indicator", "Reference Value", "Unit", "Direction"]].to_string(index=False))
