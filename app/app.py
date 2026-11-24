"""
Maternal & Child Health Gap Intelligence and Resource Prioritisation System.
Production-grade Streamlit application — Clinical Intelligence theme.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "app"))

import pandas as pd
import streamlit as st

import config

# ---------------------------------------------------------------------------
# Dual-import guard
# ---------------------------------------------------------------------------
try:
    from app.views import (
        view_executive_overview, view_dataset_explorer,
        view_gap_detection, view_clustering_lab, view_pca_studio,
        view_vulnerability_profiles, view_priority_planner,
        view_scenario_simulator, view_reports,
    )
except (ModuleNotFoundError, ImportError):
    from views import (
        view_executive_overview, view_dataset_explorer,
        view_gap_detection, view_clustering_lab, view_pca_studio,
        view_vulnerability_profiles, view_priority_planner,
        view_scenario_simulator, view_reports,
    )

# ---------------------------------------------------------------------------
# Page config — MUST be first st call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MCH Gap Intelligence | Clinical Analytics Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM — Clinical Intelligence Theme
# Colour palette:
#   Primary:    #0F2B5B  (deep navy)
#   Secondary:  #0E7490  (teal)
#   Accent:     #3B82F6  (blue)
#   Background: #F8FAFC  (soft grey-white)
#   Surface:    #FFFFFF
#   Text:       #1E293B  (dark slate)
#   Warning:    #D97706  (amber)
#   Critical:   #DC2626  (muted red)
#   Success:    #16A34A  (green)
#   Border:     #E2E8F0
# ---------------------------------------------------------------------------
_THEME_LIGHT = """
:root {
  --clr-primary:    #0F2B5B;
  --clr-secondary:  #0E7490;
  --clr-accent:     #3B82F6;
  --clr-bg:         #F8FAFC;
  --clr-surface:    #FFFFFF;
  --clr-text:       #1E293B;
  --clr-muted:      #64748B;
  --clr-border:     #E2E8F0;
  --clr-warn:       #D97706;
  --clr-critical:   #DC2626;
  --clr-success:    #16A34A;
  --clr-high-gap:   #DC2626;
  --clr-low-gap:    #16A34A;
  --shadow-sm:      0 1px 3px rgba(0,0,0,.08);
  --shadow-md:      0 4px 12px rgba(0,0,0,.10);
  --radius-sm:      6px;
  --radius-md:      10px;
  --radius-lg:      16px;
}
"""

_THEME_DARK = """
:root {
  --clr-primary:    #60A5FA;
  --clr-secondary:  #22D3EE;
  --clr-accent:     #818CF8;
  --clr-bg:         #0F172A;
  --clr-surface:    #1E293B;
  --clr-text:       #F1F5F9;
  --clr-muted:      #94A3B8;
  --clr-border:     #334155;
  --clr-warn:       #FCD34D;
  --clr-critical:   #F87171;
  --clr-success:    #4ADE80;
  --clr-high-gap:   #F87171;
  --clr-low-gap:    #4ADE80;
  --shadow-sm:      0 1px 3px rgba(0,0,0,.3);
  --shadow-md:      0 4px 12px rgba(0,0,0,.4);
  --radius-sm:      6px;
  --radius-md:      10px;
  --radius-lg:      16px;
}
"""

_CSS_SHARED = """
/* ── Global ──────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--clr-bg) !important;
    color: var(--clr-text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--clr-surface) !important;
    border-right: 1px solid var(--clr-border) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--clr-text); }

/* ── Top navigation bar brand ────────────────────────── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1rem 0 0.5rem 0;
    border-bottom: 2px solid var(--clr-secondary);
    margin-bottom: 1rem;
}
.brand-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, var(--clr-primary), var(--clr-secondary));
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
}
.brand-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--clr-primary);
    line-height: 1.2;
}
.brand-sub {
    font-size: 0.72rem;
    color: var(--clr-muted);
    font-weight: 400;
}

/* ── KPI Cards ───────────────────────────────────────── */
.kpi-card {
    background: var(--clr-surface);
    border: 1px solid var(--clr-border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow .2s;
}
.kpi-card:hover { box-shadow: var(--shadow-md); }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--clr-muted);
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--clr-primary);
    line-height: 1.1;
}
.kpi-value.critical { color: var(--clr-critical); }
.kpi-value.success  { color: var(--clr-success); }
.kpi-delta {
    font-size: 0.75rem;
    color: var(--clr-muted);
    margin-top: 2px;
}

/* ── Section headings ────────────────────────────────── */
.section-heading {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--clr-primary);
    border-left: 3px solid var(--clr-secondary);
    padding-left: 10px;
    margin: 1.4rem 0 0.8rem 0;
}

/* ── Info/warning/disclaimer banners ─────────────────── */
.banner {
    border-radius: var(--radius-sm);
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    margin-bottom: 1rem;
    border-left: 4px solid;
}
.banner-info     { background:#EFF6FF; border-color: var(--clr-accent); color:#1E40AF; }
.banner-warn     { background:#FFFBEB; border-color: var(--clr-warn);   color:#92400E; }
.banner-critical { background:#FEF2F2; border-color: var(--clr-critical); color:#991B1B; }
.banner-success  { background:#F0FDF4; border-color: var(--clr-success); color:#166534; }

/* ── Tier badges ─────────────────────────────────────── */
.tier-high { background:#FEE2E2; color:#991B1B; border-radius:4px; padding:2px 8px; font-weight:600; font-size:.8rem; }
.tier-mid  { background:#FEF3C7; color:#92400E; border-radius:4px; padding:2px 8px; font-weight:600; font-size:.8rem; }
.tier-low  { background:#D1FAE5; color:#065F46; border-radius:4px; padding:2px 8px; font-weight:600; font-size:.8rem; }

/* ── Page title ──────────────────────────────────────── */
.page-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--clr-primary);
    margin-bottom: 0.2rem;
}
.page-subtitle {
    font-size: 0.88rem;
    color: var(--clr-muted);
    margin-bottom: 1.2rem;
}

/* ── Disclaimer box ──────────────────────────────────── */
.disclaimer {
    background: var(--clr-surface);
    border: 1px solid var(--clr-border);
    border-radius: var(--radius-sm);
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: var(--clr-muted);
    margin-top: 1rem;
}

/* ── Hypothesis label ────────────────────────────────── */
.hypo-label {
    display: inline-block;
    background: #FEF3C7;
    color: #92400E;
    border-radius: 4px;
    padding: 1px 7px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .05em;
}

/* ── Streamlit overrides ─────────────────────────────── */
.stMetric { background: var(--clr-surface); border-radius: var(--radius-sm); padding:.6rem; }
div[data-testid="metric-container"] { background: var(--clr-surface); border-radius: var(--radius-sm); padding:.6rem !important; border: 1px solid var(--clr-border); }
.stButton > button {
    background: var(--clr-primary);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    font-weight: 600;
    padding: .5rem 1.4rem;
    transition: background .2s;
}
.stButton > button:hover { background: var(--clr-secondary); }
h1, h2, h3 { color: var(--clr-primary) !important; }
.stDataFrame { border: 1px solid var(--clr-border); border-radius: var(--radius-sm); }
"""

# ---------------------------------------------------------------------------
# Dark-mode toggle (session state)
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

theme_vars = _THEME_DARK if st.session_state.dark_mode else _THEME_LIGHT
st.markdown(f"<style>{theme_vars}{_CSS_SHARED}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar brand + dark-mode toggle
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
      <div class="brand-logo">🏥</div>
      <div>
        <div class="brand-name">MCH Gap Intelligence</div>
        <div class="brand-sub">Clinical Analytics Platform</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dark mode toggle — explicit key prevents duplicate-ID error on reruns
    dm = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="sidebar_dark_mode")
    if dm != st.session_state.dark_mode:
        st.session_state.dark_mode = dm
        st.rerun()

    st.markdown("---")

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset and clustering outputs…")
def load_app_data() -> dict:
    raw_df         = pd.read_csv(config.RAW_DATA_FILE)
    cleaned_df     = pd.read_csv(config.CLEANED_DATA_FILE)
    imputed_df     = pd.read_csv(config.IMPUTED_DATA_FILE)
    standardized_df= pd.read_csv(config.STANDARDIZED_DATA_FILE)

    kmeans_metrics    = pd.read_csv(config.TABLES_DIR / "kmeans_metrics_k2_to_10.csv")
    kmedoids_metrics  = pd.read_csv(config.TABLES_DIR / "kmedoids_manhattan_metrics_k2_to_10.csv")
    hier_metrics      = pd.read_csv(config.TABLES_DIR / "hierarchical_linkage_evaluation.csv")
    pca_var           = pd.read_csv(config.TABLES_DIR / "pca_variance_explained.csv")
    pca_loadings      = pd.read_csv(config.TABLES_DIR / "pca_feature_loadings.csv")
    country_assignments = pd.read_csv(config.TABLES_DIR / "country_cluster_assignments.csv")
    stability_df      = pd.read_csv(config.TABLES_DIR / "stability_validation_summary.csv")

    with open(config.TABLES_DIR / "cluster_narratives.json") as f:
        narratives = json.load(f)
    with open(config.TABLES_DIR / "kmeans_vs_kmedoids_head_to_head.json") as f:
        head_to_head = json.load(f)

    return dict(
        raw_df=raw_df, cleaned_df=cleaned_df, imputed_df=imputed_df,
        standardized_df=standardized_df, kmeans_metrics=kmeans_metrics,
        kmedoids_metrics=kmedoids_metrics, hier_metrics=hier_metrics,
        pca_var=pca_var, pca_loadings=pca_loadings,
        country_assignments=country_assignments, stability_df=stability_df,
        narratives=narratives, head_to_head=head_to_head,
    )


@st.cache_data(show_spinner="Computing gap & vulnerability analytics…")
def load_analytics_data(n_countries: int) -> dict:
    from analytics.gap_detection import calculate_region_gaps
    from analytics.vulnerability import build_vulnerability_dataframe
    from analytics.prioritisation import build_priority_report, run_scenario_analysis, PREDEFINED_SCENARIOS
    from analytics.cluster_intelligence import generate_cluster_intelligence_report

    imputed_df = pd.read_csv(config.IMPUTED_DATA_FILE)
    assignments = pd.read_csv(config.TABLES_DIR / "country_cluster_assignments.csv")

    gaps_df  = calculate_region_gaps(imputed_df)
    vuln_df  = build_vulnerability_dataframe(gaps_df, alpha=0.5)
    prio_df  = build_priority_report(vuln_df)
    scen_df  = run_scenario_analysis(vuln_df, PREDEFINED_SCENARIOS)

    cluster_col = next(
        (c for c in ["KMeans_K4","KMeans_Cluster","Cluster"] if c in assignments.columns), None
    )
    intel_report = None
    if cluster_col:
        merged = imputed_df.merge(
            assignments[["CountryName", cluster_col]], on="CountryName", how="left"
        )
        labels = merged[cluster_col].fillna(0).astype(int).values
        intel_report = generate_cluster_intelligence_report(merged, labels, gaps_df=gaps_df)

    return dict(
        gaps_df=gaps_df, vuln_df=vuln_df, priority_df=prio_df,
        scenario_df=scen_df, intel_report=intel_report,
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data     = load_app_data()
feat_cols= [c for c in config.FEATURE_COLUMNS if c in data["imputed_df"].columns]
analytics= load_analytics_data(len(data["imputed_df"]))

# ---------------------------------------------------------------------------
# Sidebar — downloads (run-counter makes keys unique every rerun)
# ---------------------------------------------------------------------------
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
st.session_state.run_count += 1
_rc = st.session_state.run_count

with st.sidebar:
    st.markdown("### 📥 Quick Exports")
    st.download_button("Cleaned Dataset",
        data["cleaned_df"].to_csv(index=False),
        "mch_cleaned.csv", "text/csv", key=f"sb1_{_rc}")
    st.download_button("Cluster Assignments",
        data["country_assignments"].to_csv(index=False),
        "cluster_assignments.csv", "text/csv", key=f"sb2_{_rc}")
    if analytics["priority_df"] is not None:
        st.download_button("Priority Rankings",
            analytics["priority_df"].to_csv(index=False),
            "priority_rankings.csv", "text/csv", key=f"sb3_{_rc}")
    if analytics["gaps_df"] is not None:
        st.download_button("Healthcare Gaps",
            analytics["gaps_df"].to_csv(index=False),
            "healthcare_gaps.csv", "text/csv", key=f"sb4_{_rc}")
    st.markdown("---")
    st.caption("🌍 DHS Program · Gates Foundation")
    st.caption("⚠️ Indices are analytical — not official classifications.")


# ---------------------------------------------------------------------------
# Page callbacks
# ---------------------------------------------------------------------------
def page_overview():           view_executive_overview(data, analytics, feat_cols)
def page_dataset():            view_dataset_explorer(data, feat_cols)
def page_gaps():               view_gap_detection(data, analytics, feat_cols)
def page_clustering():         view_clustering_lab(data, analytics, feat_cols)
def page_pca():                view_pca_studio(data, analytics, feat_cols)
def page_vulnerability():      view_vulnerability_profiles(data, analytics, feat_cols)
def page_priority():           view_priority_planner(data, analytics, feat_cols)
def page_scenario():           view_scenario_simulator(data, analytics, feat_cols)
def page_reports():            view_reports(data, analytics, feat_cols)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
pg = st.navigation({
    "Overview": [
        st.Page(page_overview, title="Executive Overview", icon="🏥",
                default=True, url_path="overview"),
    ],
    "Data Intelligence": [
        st.Page(page_dataset,  title="Dataset Explorer",   icon="🔍", url_path="dataset"),
        st.Page(page_gaps,     title="Gap Detection",       icon="📉", url_path="gaps"),
    ],
    "Machine Learning": [
        st.Page(page_clustering, title="Clustering Lab",   icon="🧪", url_path="clustering"),
        st.Page(page_pca,        title="PCA Studio",       icon="🔭", url_path="pca"),
    ],
    "Intelligence": [
        st.Page(page_vulnerability, title="Vulnerability Profiles", icon="📊", url_path="vulnerability"),
        st.Page(page_priority,      title="Priority Planner",       icon="🗺️", url_path="priority"),
        st.Page(page_scenario,      title="Scenario Simulator",     icon="🔬", url_path="scenario"),
    ],
    "Reports": [
        st.Page(page_reports, title="Reports & Exports", icon="📄", url_path="reports"),
    ],
})
pg.run()
