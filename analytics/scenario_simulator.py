"""
Scenario Simulator Engine.

Allows analysts to model hypothetical improvements in selected indicators
and observe the downstream effect on gap indices and priority rankings.

DISCLAIMER: All simulated values are HYPOTHETICAL PROJECTIONS for analytical
exploration only. They do not represent forecasts, policy commitments, or
clinical outcomes. The original dataset is never modified.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from analytics.gap_detection import BENCHMARKS, calculate_region_gaps
from analytics.vulnerability import build_vulnerability_dataframe
from analytics.prioritisation import build_priority_report

# Indicators available for simulation (higher-is-better coverage indicators)
SIMULATABLE_INDICATORS = {
    "anc_4plus_coverage": {"label": "ANC 4+ Visits Coverage (%)", "min": 0, "max": 100, "step": 1},
    "anc_skilled_provider": {"label": "ANC from Skilled Provider (%)", "min": 0, "max": 100, "step": 1},
    "skilled_birth_attendance": {"label": "Skilled Birth Attendance (%)", "min": 0, "max": 100, "step": 1},
    "institutional_delivery": {"label": "Institutional Delivery (%)", "min": 0, "max": 100, "step": 1},
    "modern_contraceptive_prevalence": {"label": "Modern Contraceptive Prevalence (%)", "min": 0, "max": 100, "step": 1},
    "dpt3_vaccination_coverage": {"label": "DPT3 Vaccination Coverage (%)", "min": 0, "max": 100, "step": 1},
    "measles_vaccination_coverage": {"label": "Measles Vaccination Coverage (%)", "min": 0, "max": 100, "step": 1},
    "basic_vaccination_coverage": {"label": "Full Basic Immunisation (%)", "min": 0, "max": 100, "step": 1},
    "exclusive_breastfeeding_rate": {"label": "Exclusive Breastfeeding Rate (%)", "min": 0, "max": 100, "step": 1},
    "improved_drinking_water_access": {"label": "Improved Water Access (%)", "min": 0, "max": 100, "step": 1},
    "improved_sanitation_access": {"label": "Improved Sanitation Access (%)", "min": 0, "max": 100, "step": 1},
}


def apply_scenario(
    df: pd.DataFrame,
    country: str,
    adjustments: Dict[str, float],
) -> pd.DataFrame:
    """
    Returns a modified copy of `df` with hypothetical indicator values applied
    to the specified country. Original data is never mutated.

    Args:
        df:          Imputed wide-format DataFrame.
        country:     CountryName to modify.
        adjustments: {indicator_col: new_hypothetical_value}

    Returns:
        New DataFrame with hypothetical values applied. A column
        ``_simulated`` is set to True for the modified row.
    """
    sim_df = df.copy()
    if "_simulated" not in sim_df.columns:
        sim_df["_simulated"] = False

    mask = sim_df["CountryName"] == country
    if not mask.any():
        raise ValueError(f"Country '{country}' not found in DataFrame.")

    for col, val in adjustments.items():
        if col in sim_df.columns:
            # Cap at indicator's physical maximum (100 for percentages)
            bm = BENCHMARKS.get(col, {})
            physical_max = 100.0 if "%" in bm.get("unit", "%") else val
            sim_df.loc[mask, col] = min(float(val), physical_max)
        else:
            raise ValueError(f"Indicator column '{col}' not in dataset.")

    sim_df.loc[mask, "_simulated"] = True
    return sim_df


def run_scenario_comparison(
    df: pd.DataFrame,
    country: str,
    adjustments: Dict[str, float],
    alpha: float = 0.5,
) -> Dict:
    """
    Runs baseline vs. hypothetical scenario and returns a structured comparison.

    Returns dict with keys:
      baseline_gaps, scenario_gaps,
      baseline_vuln, scenario_vuln,
      baseline_priority, scenario_priority,
      delta_maternal, delta_child, delta_overall,
      delta_rank, country, adjustments
    """
    # Baseline
    baseline_gaps = calculate_region_gaps(df)
    baseline_vuln = build_vulnerability_dataframe(baseline_gaps, alpha=alpha)
    baseline_prio = build_priority_report(baseline_vuln)

    # Hypothetical scenario
    sim_df = apply_scenario(df, country, adjustments)
    scenario_gaps = calculate_region_gaps(sim_df)
    scenario_vuln = build_vulnerability_dataframe(scenario_gaps, alpha=alpha)
    scenario_prio = build_priority_report(scenario_vuln)

    # Extract deltas for the target country
    def _get_row(df_in, col):
        row = df_in[df_in["CountryName"] == country]
        return float(row[col].iloc[0]) if len(row) and col in row.columns else float("nan")

    b_mat = _get_row(baseline_vuln, "maternal_gap_index")
    s_mat = _get_row(scenario_vuln, "maternal_gap_index")
    b_chi = _get_row(baseline_vuln, "child_gap_index")
    s_chi = _get_row(scenario_vuln, "child_gap_index")
    b_ov  = _get_row(baseline_vuln, "overall_gap_index")
    s_ov  = _get_row(scenario_vuln, "overall_gap_index")
    b_rk  = _get_row(baseline_prio, "priority_rank")
    s_rk  = _get_row(scenario_prio, "priority_rank")

    # Indicator-level before/after for the country
    bm_ind = {col: _get_row(df, col) for col in adjustments}
    sc_ind = {col: _get_row(sim_df, col) for col in adjustments}

    return {
        "country": country,
        "adjustments": adjustments,
        "baseline_vuln": baseline_vuln,
        "scenario_vuln": scenario_vuln,
        "baseline_prio": baseline_prio,
        "scenario_prio": scenario_prio,
        "baseline_gaps": baseline_gaps,
        "scenario_gaps": scenario_gaps,
        "baseline_maternal": b_mat,
        "scenario_maternal": s_mat,
        "baseline_child": b_chi,
        "scenario_child": s_chi,
        "baseline_overall": b_ov,
        "scenario_overall": s_ov,
        "baseline_rank": int(b_rk) if not np.isnan(b_rk) else None,
        "scenario_rank": int(s_rk) if not np.isnan(s_rk) else None,
        "delta_maternal": s_mat - b_mat,
        "delta_child": s_chi - b_chi,
        "delta_overall": s_ov - b_ov,
        "delta_rank": (int(s_rk) - int(b_rk)) if (not np.isnan(b_rk) and not np.isnan(s_rk)) else None,
        "indicator_baseline": bm_ind,
        "indicator_scenario": sc_ind,
    }


def build_before_after_df(result: Dict) -> pd.DataFrame:
    """Returns a tidy before/after DataFrame for the adjusted indicators."""
    rows = []
    for col in result["adjustments"]:
        label = SIMULATABLE_INDICATORS.get(col, {}).get("label", col)
        rows.append({
            "Indicator": label,
            "Baseline": result["indicator_baseline"].get(col, float("nan")),
            "Hypothetical": result["indicator_scenario"].get(col, float("nan")),
            "Change": result["indicator_scenario"].get(col, 0) - result["indicator_baseline"].get(col, 0),
        })
    return pd.DataFrame(rows)
