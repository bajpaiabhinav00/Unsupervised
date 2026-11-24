"""
Geographical Visualisation Module.

Produces Plotly choropleth maps and regional overlays for the Maternal & Child
Health Gap Intelligence dashboard.

Uses natural-earth country boundaries via geopandas / plotly built-in ISO codes.
Falls back gracefully to tabular display when geographical dependencies are absent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# DHS country codes → ISO 3166-1 alpha-3 mapping (most common DHS countries)
DHS_TO_ISO3: Dict[str, str] = {
    "AF": "AFG", "AL": "ALB", "AO": "AGO", "AM": "ARM", "AZ": "AZE",
    "BD": "BGD", "BJ": "BEN", "BF": "BFA", "BI": "BDI", "CM": "CMR",
    "CV": "CPV", "CF": "CAF", "TD": "TCD", "CO": "COL", "KM": "COM",
    "CG": "COG", "CD": "COD", "CI": "CIV", "DR": "DOM", "EG": "EGY",
    "ET": "ETH", "GA": "GAB", "GH": "GHA", "GN": "GIN", "GY": "GUY",
    "HT": "HTI", "HN": "HND", "IA": "IND", "ID": "IDN", "JO": "JOR",
    "KE": "KEN", "KH": "KHM", "KY": "KGZ", "LA": "LAO", "LB": "LBR",
    "LS": "LSO", "MD": "MDA", "MG": "MDG", "MW": "MWI", "ML": "MLI",
    "MR": "MRT", "MA": "MAR", "MZ": "MOZ", "MM": "MMR", "NM": "NAM",
    "NP": "NPL", "NI": "NIC", "NE": "NER", "NG": "NGA", "PK": "PAK",
    "PH": "PHL", "PE": "PER", "RW": "RWA", "SN": "SEN", "SL": "SLE",
    "ZA": "ZAF", "SS": "SSD", "SD": "SDN", "TZ": "TZA", "TG": "TGO",
    "TL": "TLS", "TJ": "TJK", "TH": "THA", "TN": "TUN", "TR": "TUR",
    "TM": "TKM", "UG": "UGA", "UA": "UKR", "UZ": "UZB", "VN": "VNM",
    "YE": "YEM", "ZM": "ZMB", "ZW": "ZWE", "EK": "SWZ", "GU": "GTM",
    "MN": "MNG", "SZ": "SWZ",
}

# Colour scales (Plotly named scales)
COLOUR_SCALE_HIGH_BAD = "Reds"       # higher value = worse (mortality, stunting)
COLOUR_SCALE_HIGH_GOOD = "Blues_r"   # higher value = better (coverage)
COLOUR_SCALE_GAP = "YlOrRd"          # gap score (0 = good, 1 = bad)
COLOUR_SCALE_CLUSTER = "Set1"


def _add_iso3(df: pd.DataFrame) -> pd.DataFrame:
    """Add an iso_alpha3 column using DHS_CountryCode → ISO3 mapping."""
    df = df.copy()
    if "DHS_CountryCode" in df.columns:
        df["iso_alpha3"] = df["DHS_CountryCode"].map(DHS_TO_ISO3)
    elif "CountryName" in df.columns:
        # Attempt a name-based fallback using pycountry if installed
        try:
            import pycountry
            def name_to_iso3(name: str) -> Optional[str]:
                try:
                    return pycountry.countries.lookup(name).alpha_3
                except Exception:
                    return None
            df["iso_alpha3"] = df["CountryName"].map(name_to_iso3)
        except ImportError:
            df["iso_alpha3"] = None
    else:
        df["iso_alpha3"] = None
    return df


def create_choropleth_map(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    colour_scale: str = COLOUR_SCALE_GAP,
    hover_cols: Optional[list] = None,
    country_col: str = "CountryName",
    range_color: Optional[tuple] = None,
):
    """
    Create a Plotly choropleth map coloured by `value_col`.

    Args:
        df:           DataFrame with CountryName / DHS_CountryCode and the value column.
        value_col:    Column to use for colour encoding.
        title:        Map title.
        colour_scale: Plotly named colour scale.
        hover_cols:   Extra columns shown on hover.
        country_col:  Column used as country label.
        range_color:  (min, max) colour range.  Auto if None.

    Returns:
        plotly.graph_objects.Figure or None on failure.
    """
    try:
        import plotly.express as px
    except ImportError:
        logger.error("Plotly is not installed — cannot create choropleth map.")
        return None

    if value_col not in df.columns:
        logger.error(f"Column '{value_col}' not found in DataFrame.")
        return None

    map_df = _add_iso3(df)
    valid = map_df["iso_alpha3"].notna()
    if valid.sum() < 5:
        logger.warning(
            "Fewer than 5 countries could be matched to ISO-3 codes. "
            "The map may be largely empty."
        )

    hover_data = {col: True for col in (hover_cols or [])}
    hover_data["iso_alpha3"] = False  # hide the code from tooltip

    fig = px.choropleth(
        map_df[valid],
        locations="iso_alpha3",
        color=value_col,
        hover_name=country_col,
        hover_data=hover_data,
        color_continuous_scale=colour_scale,
        range_color=range_color or (map_df[value_col].min(), map_df[value_col].max()),
        title=title,
        projection="natural earth",
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=value_col.replace("_", " ").title()),
        height=500,
    )
    return fig


def create_cluster_map(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    country_col: str = "CountryName",
    title: str = "Cluster Membership by Country",
):
    """
    Create a choropleth showing cluster assignments as discrete colour categories.
    """
    try:
        import plotly.express as px
    except ImportError:
        return None

    map_df = _add_iso3(df).copy()
    map_df[cluster_col] = map_df[cluster_col].astype(str)
    valid = map_df["iso_alpha3"].notna()

    fig = px.choropleth(
        map_df[valid],
        locations="iso_alpha3",
        color=cluster_col,
        hover_name=country_col,
        title=title,
        projection="natural earth",
        category_orders={cluster_col: sorted(map_df[cluster_col].unique())},
    )
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=500,
    )
    return fig


def create_gap_layer_map(
    gaps_df: pd.DataFrame,
    layer: str = "maternal",
    title: Optional[str] = None,
):
    """
    Convenience wrapper to create a gap-score choropleth for 'maternal' or 'child' layer.

    Args:
        gaps_df: Output of ``calculate_region_gaps()`` merged with vulnerability indices.
        layer:   'maternal' or 'child'.
        title:   Override title.
    """
    col_map = {
        "maternal": "maternal_gap_index",
        "child": "child_gap_index",
        "overall": "overall_gap_index",
    }
    col = col_map.get(layer, "overall_gap_index")
    _title = title or f"{layer.title()} Healthcare Gap Index by Country"
    hover = ["SurveyYear"] if "SurveyYear" in gaps_df.columns else []
    return create_choropleth_map(
        gaps_df,
        value_col=col,
        title=_title,
        colour_scale=COLOUR_SCALE_GAP,
        hover_cols=hover,
        range_color=(0.0, 1.0),
    )


def fallback_table(df: pd.DataFrame, value_col: str, n: int = 20) -> pd.DataFrame:
    """
    Returns the top n rows sorted by value_col for tabular display when
    geographical visualisation is unavailable.
    """
    cols = [c for c in ["CountryName", "SurveyYear", value_col] if c in df.columns]
    return df[cols].sort_values(value_col, ascending=False).head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import get_latest_cross_sectional_dataset
    from analytics.gap_detection import calculate_region_gaps
    from analytics.vulnerability import build_vulnerability_dataframe

    df, _ = get_latest_cross_sectional_dataset(min_year=2000)
    gaps = calculate_region_gaps(df)
    vuln = build_vulnerability_dataframe(gaps)
    merged = df[["CountryName", "DHS_CountryCode", "SurveyYear"]].merge(vuln, on="CountryName", how="left")

    fig = create_gap_layer_map(merged, layer="overall")
    if fig:
        print("Choropleth created successfully.")
        fig.show()
    else:
        print("Fallback — no Plotly or insufficient ISO matches.")
        print(fallback_table(merged, "overall_gap_index"))
