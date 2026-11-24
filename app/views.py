"""
View functions for the 8-page Maternal & Child Health Gap Intelligence dashboard.

Each view_* function is called by the page callback in app.py.
All displayed values are computed from actual data — no hardcoded statistics.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from analytics.gap_detection import (
    BENCHMARKS,
    MATERNAL_INDICATORS,
    CHILD_INDICATORS,
    calculate_region_gaps,
    calculate_maternal_gap_summary,
    calculate_child_gap_summary,
    get_top_gap_indicators,
    get_benchmark_table,
    generate_gap_heatmap_data,
)
from analytics.vulnerability import (
    build_vulnerability_dataframe,
    compute_maternal_gap_index,
    compute_child_gap_index,
    compute_overall_index,
    compute_sensitivity_analysis,
    _normalise_weights,
)
from analytics.prioritisation import (
    compute_priority_scores,
    categorise_priorities,
    run_scenario_analysis,
    build_priority_report,
    PREDEFINED_SCENARIOS,
)
from analytics.cluster_intelligence import (
    compute_cluster_profiles,
    generate_cluster_descriptions,
    compute_z_score_profiles,
    PROFILE_INDICATORS,
)
from analytics.geo_visualisation import (
    create_choropleth_map,
    create_cluster_map,
    create_gap_layer_map,
    fallback_table,
    COLOUR_SCALE_GAP,
)


# ---------------------------------------------------------------------------
# Shared helper utilities
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 1, suffix: str = "") -> str:
    """Format a numeric value safely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}{suffix}"


def _section(title: str, icon: str = "") -> None:
    label = f"{icon} {title}".strip() if icon else title
    st.markdown(f'<div class="section-heading">{label}</div>', unsafe_allow_html=True)


def _info_box(text: str) -> None:
    st.markdown(
        f'<div class="banner banner-info">ℹ️ {text}</div>',
        unsafe_allow_html=True,
    )


def _warn_box(text: str) -> None:
    st.markdown(
        f'<div class="banner banner-warn">⚠️ {text}</div>',
        unsafe_allow_html=True,
    )


def _tier_badge(tier: str) -> str:
    if "Higher" in tier:
        return f'<span class="tier-high">{tier}</span>'
    elif "Moderate" in tier:
        return f'<span class="tier-mid">{tier}</span>'
    else:
        return f'<span class="tier-low">{tier}</span>'


# ===========================================================================
# PAGE 1: Executive Overview
# ===========================================================================

def view_executive_overview(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    st.markdown(
        '<div class="main-title">🏥 Maternal & Child Health Gap Intelligence System</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Resource Prioritisation Using Unsupervised Machine Learning · DHS Program Data</div>',
        unsafe_allow_html=True,
    )

    imputed_df = data["imputed_df"]
    assignments = data["country_assignments"]
    gaps_df = analytics.get("gaps_df")
    vuln_df = analytics.get("vuln_df")
    priority_df = analytics.get("priority_df")

    n_countries = len(imputed_df)
    n_indicators = len(feat_cols)
    yr_col = "SurveyYear" if "SurveyYear" in imputed_df.columns else None
    yr_range = (
        f"{int(imputed_df[yr_col].min())}–{int(imputed_df[yr_col].max())}"
        if yr_col else "N/A"
    )
    cluster_col = next(
        (c for c in ["KMeans_K4", "KMeans_Cluster", "Cluster"] if c in assignments.columns),
        None,
    )
    n_clusters = int(assignments[cluster_col].nunique()) if cluster_col else "N/A"

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Countries Analysed", n_countries)
    c2.metric("Health Indicators", n_indicators)
    c3.metric("Survey Years", yr_range)
    c4.metric("Clusters Identified", n_clusters)
    if vuln_df is not None:
        highest_gap_country = vuln_df.iloc[0]["CountryName"]
        c5.metric("Highest Gap", highest_gap_country)

    st.divider()

    # --- Gap summary cards ---
    if gaps_df is not None and vuln_df is not None:
        col_l, col_r = st.columns(2)
        with col_l:
            _section("Top 5 Maternal Gap Countries", "🤱")
            mat_sum = calculate_maternal_gap_summary(gaps_df).head(5)
            for _, row in mat_sum.iterrows():
                mg = row["mean_maternal_gap"]
                colour = "gap-high" if mg > 0.4 else ("gap-low" if mg < 0.2 else "")
                st.markdown(
                    f'**{row["CountryName"]}** — mean gap: '
                    f'<span class="{colour}">{mg:.3f}</span>',
                    unsafe_allow_html=True,
                )

        with col_r:
            _section("Top 5 Child Gap Countries", "👶")
            child_sum = calculate_child_gap_summary(gaps_df).head(5)
            for _, row in child_sum.iterrows():
                cg = row["mean_child_gap"]
                colour = "gap-high" if cg > 0.5 else ("gap-low" if cg < 0.2 else "")
                st.markdown(
                    f'**{row["CountryName"]}** — mean gap: '
                    f'<span class="{colour}">{cg:.3f}</span>',
                    unsafe_allow_html=True,
                )

    st.divider()

    # --- Highest gap indicators ---
    if gaps_df is not None:
        _section("Highest Mean Indicator Gaps Across All Countries", "📉")
        top_gaps = get_top_gap_indicators(gaps_df, n=8)
        fig = px.bar(
            x=top_gaps.values,
            y=[col.replace("_gap", "").replace("_", " ").title() for col in top_gaps.index],
            orientation="h",
            labels={"x": "Mean Normalised Gap (0–1)", "y": "Indicator"},
            color=top_gaps.values,
            color_continuous_scale="Reds",
            title="Mean Gap Score by Indicator (all countries)",
        )
        fig.update_layout(showlegend=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Geographical overview ---
    if analytics.get("vuln_df") is not None and "DHS_CountryCode" in imputed_df.columns:
        _section("Overall Healthcare Gap Index — Geographical View", "🌍")
        merged = imputed_df[["CountryName", "DHS_CountryCode", "SurveyYear"]].merge(
            analytics["vuln_df"][["CountryName", "overall_gap_index"]],
            on="CountryName", how="left",
        )
        fig_map = create_gap_layer_map(merged, layer="overall")
        if fig_map:
            st.plotly_chart(fig_map, width="stretch")
        else:
            st.dataframe(fallback_table(merged, "overall_gap_index"))

    st.divider()

    # --- Methodology disclaimer ---
    with st.expander("📋 Methodology & Disclaimers", expanded=False):
        st.markdown("""
**Data Source**: DHS Program (dhsprogram.com) REST API — real indicator values for 70+ countries.

**Clustering**: K-Means (K=4, optimal by Davies-Bouldin), validated with Hierarchical and K-Medoids.

**Gap Detection**: Normalised deviation from WHO/UNICEF/SDG benchmarks (all documented in the Gap Detection page).

**Vulnerability Indices**: Composite weighted averages of normalised gap scores (Maternal, Child, Overall).

**Priority Tiers**: Relative analytical categories within this sample — not official government or WHO classifications.

**Limitations**:
- Cross-country survey comparability varies by indicator definition and survey vintage.
- Benchmarks reflect global targets; regional contexts may differ.
- Higher gap scores flag potential areas for further investigation, not confirmed adverse outcomes.
        """)


# ===========================================================================
# PAGE 2: Dataset Explorer
# ===========================================================================

def view_dataset_explorer(data: dict, feat_cols: List[str]) -> None:
    st.markdown('<div class="page-title">🔍 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Browse, filter and validate the DHS indicator dataset.</div>', unsafe_allow_html=True)

    imputed_df = data["imputed_df"]
    cleaned_df = data["cleaned_df"]

    # --- Upload custom dataset ---
    _section("Upload Your Own Dataset", "📤")
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file (must contain CountryName + indicator columns)",
        type=["csv", "xlsx"],
        help="The file must contain a 'CountryName' column and at least some DHS indicator columns.",
    )
    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                custom_df = pd.read_csv(uploaded)
            else:
                custom_df = pd.read_excel(uploaded)
            st.success(f"✅ Loaded '{uploaded.name}': {custom_df.shape[0]} rows × {custom_df.shape[1]} cols")
            active_df = custom_df
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            active_df = imputed_df
    else:
        active_df = imputed_df
        st.info("Using the pre-processed DHS dataset (70 countries). Upload a file above to use custom data.")

    st.divider()

    # --- Filters ---
    _section("Filters", "🎛️")
    col1, col2, col3 = st.columns(3)

    countries = sorted(active_df["CountryName"].dropna().unique().tolist()) if "CountryName" in active_df.columns else []
    selected_countries = col1.multiselect("Countries", countries, default=[])

    yr_col = "SurveyYear" if "SurveyYear" in active_df.columns else None
    if yr_col:
        yr_min, yr_max = int(active_df[yr_col].min()), int(active_df[yr_col].max())
        yr_range = col2.slider("Survey Year Range", yr_min, yr_max, (yr_min, yr_max))
    else:
        yr_range = None

    num_cols = [c for c in active_df.columns if c in feat_cols]
    selected_inds = col3.multiselect("Show Indicators", num_cols, default=num_cols[:6] if num_cols else [])

    # Apply filters
    display_df = active_df.copy()
    if selected_countries:
        display_df = display_df[display_df["CountryName"].isin(selected_countries)]
    if yr_col and yr_range:
        display_df = display_df[
            (display_df[yr_col] >= yr_range[0]) & (display_df[yr_col] <= yr_range[1])
        ]

    meta_cols = [c for c in ["CountryName", "DHS_CountryCode", "SurveyYear"] if c in display_df.columns]
    show_cols = meta_cols + (selected_inds if selected_inds else num_cols[:6])
    show_cols = [c for c in show_cols if c in display_df.columns]

    st.dataframe(display_df[show_cols], width="stretch", height=400)
    st.caption(f"Showing {len(display_df)} of {len(active_df)} countries.")

    st.divider()

    # --- Missing value analysis ---
    _section("Missing Value Analysis", "🕳️")
    missing = active_df[num_cols].isna().mean() * 100
    missing = missing.sort_values(ascending=False)
    missing_df = pd.DataFrame({
        "Indicator": missing.index,
        "Missing %": missing.values.round(1),
        "Countries with Data": (active_df[num_cols].notna().sum()).reindex(missing.index).values,
    })
    col_miss, col_bar = st.columns([1, 2])
    with col_miss:
        st.dataframe(missing_df, width="stretch")
    with col_bar:
        fig = px.bar(
            missing_df,
            x="Missing %",
            y="Indicator",
            orientation="h",
            color="Missing %",
            color_continuous_scale="Oranges",
            title="% Missing per Indicator",
        )
        fig.update_layout(height=400, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Data quality report ---
    _section("Data Quality Report", "✅")
    total_cells = active_df[num_cols].size
    missing_cells = active_df[num_cols].isna().sum().sum()
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Countries", len(active_df))
    q2.metric("Total Indicators", len(num_cols))
    q3.metric("Missing Cells", f"{missing_cells:,} ({100*missing_cells/total_cells:.1f}%)")
    q4.metric("Complete Rows", int((active_df[num_cols].notna().all(axis=1)).sum()))

    # Duplicate check
    dup_count = active_df.duplicated(subset=["CountryName"] if "CountryName" in active_df.columns else None).sum()
    if dup_count > 0:
        _warn_box(f"⚠️ {dup_count} duplicate country records detected.")
    else:
        st.success("✅ No duplicate country records found.")

    # Range checks
    out_of_range = []
    for col in num_cols:
        vals = active_df[col].dropna()
        if "mortality" in col.lower():
            if (vals < 0).any() or (vals > 500).any():
                out_of_range.append(col)
        else:
            if (vals < 0).any() or (vals > 100).any():
                out_of_range.append(col)
    if out_of_range:
        _warn_box(f"Indicators with potentially out-of-range values: {', '.join(out_of_range)}")
    else:
        st.success("✅ All indicator values are within plausible ranges.")


# ===========================================================================
# PAGE 3: Healthcare Gap Detection
# ===========================================================================

def view_gap_detection(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    st.markdown('<div class="page-title">📉 Healthcare Gap Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Normalised gaps vs. WHO/UNICEF/SDG benchmarks. '
        'A gap of 0 means the benchmark is met; larger values indicate larger shortfalls.</div>',
        unsafe_allow_html=True,
    )
    _warn_box(
        "These gap scores are analytical reference points, not clinical severity assessments. "
        "Deviation from a benchmark flags potential areas for further investigation."
    )

    imputed_df = data["imputed_df"]
    gaps_df = analytics.get("gaps_df")

    if gaps_df is None:
        st.error("Gap data not available. Re-run the pipeline.")
        return

    # --- Benchmark configuration panel ---
    with st.expander("⚙️ Benchmark Configuration & Sources", expanded=False):
        bm_table = get_benchmark_table()
        st.dataframe(
            bm_table[["Indicator", "Reference Value", "Unit", "Direction", "Source", "Year"]],
            width="stretch",
        )

    st.divider()

    # --- Domain selector ---
    domain = st.radio(
        "Select Analysis Domain",
        ["Maternal Health", "Child Health", "Combined"],
        horizontal=True,
    )

    if domain == "Maternal Health":
        indicator_set = MATERNAL_INDICATORS
        summary = calculate_maternal_gap_summary(gaps_df)
        gap_col = "mean_maternal_gap"
        title_sfx = "Maternal"
    elif domain == "Child Health":
        indicator_set = CHILD_INDICATORS
        summary = calculate_child_gap_summary(gaps_df)
        gap_col = "mean_child_gap"
        title_sfx = "Child"
    else:
        indicator_set = list(BENCHMARKS.keys())
        m_sum = calculate_maternal_gap_summary(gaps_df)
        c_sum = calculate_child_gap_summary(gaps_df)
        summary = m_sum.merge(
            c_sum[["CountryName", "mean_child_gap"]], on="CountryName", how="outer"
        )
        summary["mean_combined_gap"] = (
            summary["mean_maternal_gap"].fillna(0) + summary["mean_child_gap"].fillna(0)
        ) / 2
        summary = summary.sort_values("mean_combined_gap", ascending=False).reset_index(drop=True)
        gap_col = "mean_combined_gap"
        title_sfx = "Combined"

    # --- Top 20 bar chart ---
    top20 = summary.head(20)
    fig = px.bar(
        top20,
        x=gap_col,
        y="CountryName",
        orientation="h",
        color=gap_col,
        color_continuous_scale="YlOrRd",
        title=f"Top 20 Countries by Mean {title_sfx} Healthcare Gap",
        labels={gap_col: "Mean Gap Score", "CountryName": "Country"},
    )
    fig.update_layout(height=500, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    # --- Gap heatmap ---
    st.divider()
    _section(f"{title_sfx} Gap Heatmap (Normalised, 0 = Benchmark Met)", "🔥")

    # Limit to top 30 countries for readability
    top_countries = summary["CountryName"].head(30).tolist()
    hm_gaps = gaps_df[gaps_df["CountryName"].isin(top_countries)].copy()

    gap_cols_sel = [f"{i}_gap" for i in indicator_set if f"{i}_gap" in hm_gaps.columns]
    hm_data = hm_gaps[["CountryName"] + gap_cols_sel].set_index("CountryName")
    hm_data.columns = [c.replace("_gap", "").replace("_", " ").title() for c in hm_data.columns]

    if not hm_data.empty:
        fig_hm = px.imshow(
            hm_data.astype(float),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title=f"{title_sfx} Indicator Gaps — Top 30 Countries",
            labels=dict(color="Gap Score"),
            zmin=0,
        )
        fig_hm.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_hm, width="stretch")

    st.divider()

    # --- Detailed country drill-down ---
    _section("Country Drill-Down", "🔎")
    selected_country = st.selectbox(
        "Select Country",
        summary["CountryName"].tolist(),
    )
    country_gaps = gaps_df[gaps_df["CountryName"] == selected_country]
    if not country_gaps.empty:
        country_gap_cols = [c for c in gaps_df.columns if c.endswith("_gap")]
        vals = country_gaps[country_gap_cols].iloc[0]
        vals.index = [c.replace("_gap", "").replace("_", " ").title() for c in vals.index]
        vals = vals.dropna().sort_values(ascending=False)

        fig_drill = px.bar(
            x=vals.values,
            y=vals.index,
            orientation="h",
            color=vals.values,
            color_continuous_scale="YlOrRd",
            title=f"Gap Scores for {selected_country}",
            labels={"x": "Gap Score", "y": "Indicator"},
        )
        fig_drill.update_layout(showlegend=False, height=400, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_drill, width="stretch")

    # Download
    st.download_button(
        "📥 Download Gap Report (CSV)",
        data=gaps_df.to_csv(index=False),
        file_name=f"healthcare_gaps_{title_sfx.lower()}.csv",
        mime="text/csv",
        key=f"dl_gap_{domain}",
    )


# ===========================================================================
# PAGE 4: Clustering Laboratory
# ===========================================================================

def view_clustering_lab(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    st.markdown('<div class="page-title">🧪 Clustering Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Compare K-Means, K-Medoids, and Hierarchical clustering on healthcare indicators.</div>', unsafe_allow_html=True)

    kmeans_metrics = data["kmeans_metrics"]
    kmedoids_metrics = data["kmedoids_metrics"]
    hier_metrics = data["hier_metrics"]
    assignments = data["country_assignments"]
    stability_df = data["stability_df"]

    # --- Algorithm selector ---
    algo = st.radio("Algorithm", ["K-Means", "K-Medoids (Manhattan)", "Hierarchical"], horizontal=True)

    st.divider()

    if algo == "K-Means":
        _section("K-Means Evaluation Across K=2–10", "📊")
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.line(
                kmeans_metrics, x="K", y="Silhouette",
                title="Silhouette Score vs K",
                markers=True, color_discrete_sequence=["#2563EB"],
            )
            st.plotly_chart(fig, width="stretch")
        with col_b:
            fig2 = px.line(
                kmeans_metrics, x="K", y="DaviesBouldin",
                title="Davies-Bouldin Index vs K (lower=better)",
                markers=True, color_discrete_sequence=["#DC2626"],
            )
            st.plotly_chart(fig2, width="stretch")

        # Elbow
        fig_elbow = px.line(
            kmeans_metrics, x="K", y="WCSS",
            title="WCSS Elbow Curve",
            markers=True,
        )
        st.plotly_chart(fig_elbow, width="stretch")

        st.dataframe(kmeans_metrics, width="stretch")

    elif algo == "K-Medoids (Manhattan)":
        _section("K-Medoids Evaluation Across K=2–10", "📊")
        # K-Medoids uses 'Silhouette_Manhattan' column (Manhattan-distance silhouette)
        sil_col = "Silhouette_Manhattan" if "Silhouette_Manhattan" in kmedoids_metrics.columns else "Silhouette"
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.line(
                kmedoids_metrics, x="K", y=sil_col,
                title="Silhouette Score vs K (Manhattan)",
                markers=True, color_discrete_sequence=["#7C3AED"],
            )
            st.plotly_chart(fig, width="stretch")
        with col_b:
            fig2 = px.line(
                kmedoids_metrics, x="K", y="DaviesBouldin",
                title="Davies-Bouldin Index vs K (lower=better)",
                markers=True, color_discrete_sequence=["#D97706"],
            )
            st.plotly_chart(fig2, width="stretch")
        st.dataframe(kmedoids_metrics, width="stretch")

    else:
        _section("Hierarchical Clustering — Linkage Comparison", "🌳")
        st.dataframe(hier_metrics, width="stretch")
        # column is 'Cophenetic_Correlation' (with underscore)
        coph_col = "Cophenetic_Correlation" if "Cophenetic_Correlation" in hier_metrics.columns else "CopheneticCorrelation"
        fig = px.bar(
            hier_metrics, x="Linkage", y=coph_col,
            title="Cophenetic Correlation by Linkage Method (higher=better)",
            color=coph_col,
            color_continuous_scale="Greens",
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # --- Cluster assignments ---
    _section("Country Cluster Assignments (K=4)", "🗂️")
    cluster_col = next(
        (c for c in ["KMeans_K4", "KMeans_Cluster", "Cluster"] if c in assignments.columns),
        None,
    )
    if cluster_col:
        clusters = sorted(assignments[cluster_col].unique())
        selected_cluster = st.selectbox("Filter by Cluster", ["All"] + [str(c) for c in clusters])
        if selected_cluster == "All":
            show_assign = assignments
        else:
            show_assign = assignments[assignments[cluster_col] == int(selected_cluster)]
        st.dataframe(show_assign, width="stretch", height=400)
        st.caption(f"Showing {len(show_assign)} countries.")

    st.divider()

    # --- Stability ---
    _section("Clustering Stability Summary", "🔁")
    st.dataframe(stability_df, width="stretch")
    _info_box(
        "Stability was assessed using random seed perturbation (Adjusted Rand Index across seeds) "
        "and bootstrap resampling. Higher ARI = more stable cluster assignments."
    )


# ===========================================================================
# PAGE 5: PCA Studio
# ===========================================================================

def view_pca_studio(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    st.markdown('<div class="page-title">🔭 PCA Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Dimensionality reduction visualisation with cluster overlays and feature loadings.</div>', unsafe_allow_html=True)
    _info_box(
        "PCA loading magnitudes indicate how strongly each indicator contributes to a principal component. "
        "They do NOT imply causal relationships."
    )

    pca_var = data["pca_var"]
    pca_loadings = data["pca_loadings"]
    assignments = data["country_assignments"]
    imputed_df = data["imputed_df"]

    # Merge PCA coordinates if available from outputs
    pca_coords_path = config.TABLES_DIR / "pca_coordinates.csv"
    if pca_coords_path.exists():
        pca_coords = pd.read_csv(pca_coords_path)
    else:
        # Recompute on the fly
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA as _PCA
        X = imputed_df[feat_cols].values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        pca = _PCA(n_components=min(10, len(feat_cols)))
        Xp = pca.fit_transform(Xs)
        pca_coords = pd.DataFrame(
            Xp[:, :5],
            columns=[f"PC{i+1}" for i in range(min(5, Xp.shape[1]))],
        )
        pca_coords["CountryName"] = imputed_df["CountryName"].values

    cluster_col = next(
        (c for c in ["KMeans_K4", "KMeans_Cluster", "Cluster"] if c in assignments.columns),
        None,
    )
    if cluster_col:
        pca_coords = pca_coords.merge(
            assignments[["CountryName", cluster_col]], on="CountryName", how="left"
        )
        pca_coords[cluster_col] = pca_coords[cluster_col].fillna(-1).astype(int).astype(str)

    st.divider()
    _section("Explained Variance", "📐")
    fig_var = px.bar(
        pca_var,
        x="Component",
        y="Explained_Variance_Pct",
        title="Explained Variance (%) per Principal Component",
        color="Explained_Variance_Pct",
        color_continuous_scale="Blues",
        labels={"Explained_Variance_Pct": "% Variance", "Component": "PC"},
    )
    # Add cumulative line
    fig_var.add_scatter(
        x=pca_var["Component"], y=pca_var["Cumulative_Variance_Pct"],
        mode="lines+markers", name="Cumulative %",
        line=dict(color="#DC2626", width=2),
        yaxis="y",
    )
    fig_var.update_layout(showlegend=True, height=350, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_var, width="stretch")

    st.divider()

    # --- 2D scatter ---
    _section("2D PCA Scatter — Cluster Overlay", "⬡")
    if "PC1" in pca_coords.columns and "PC2" in pca_coords.columns:
        fig2d = px.scatter(
            pca_coords,
            x="PC1", y="PC2",
            color=cluster_col if cluster_col else None,
            hover_name="CountryName",
            title="PCA Biplot — PC1 vs PC2 (coloured by cluster)",
            height=500,
        )
        st.plotly_chart(fig2d, width="stretch")

    # --- 3D scatter ---
    if "PC3" in pca_coords.columns:
        with st.expander("🌐 3D PCA Scatter"):
            fig3d = px.scatter_3d(
                pca_coords,
                x="PC1", y="PC2", z="PC3",
                color=cluster_col if cluster_col else None,
                hover_name="CountryName",
                title="PCA 3D — PC1, PC2, PC3",
                height=600,
            )
            st.plotly_chart(fig3d, width="stretch")

    st.divider()

    # --- Feature loadings ---
    _section("Feature Loadings (PC1 & PC2)", "⚡")
    if not pca_loadings.empty:
        # Rename the index column for clarity
        load_df = pca_loadings.copy()
        feature_col = load_df.columns[0]  # 'Unnamed: 0' or similar
        load_df = load_df.rename(columns={feature_col: "Feature"})
        load_df["Feature"] = load_df["Feature"].str.replace("_", " ").str.title()

        col_l, col_r = st.columns(2)
        with col_l:
            fig_l1 = px.bar(
                load_df.sort_values("PC1", ascending=True),
                x="PC1", y="Feature",
                orientation="h",
                title="PC1 Loadings",
                color="PC1",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )
            fig_l1.update_layout(height=420, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_l1, width="stretch")
        with col_r:
            fig_l2 = px.bar(
                load_df.sort_values("PC2", ascending=True),
                x="PC2", y="Feature",
                orientation="h",
                title="PC2 Loadings",
                color="PC2",
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
            )
            fig_l2.update_layout(height=420, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_l2, width="stretch")

        st.dataframe(load_df[["Feature", "PC1", "PC2", "PC3", "PC4", "PC5"]], width="stretch")


# ===========================================================================
# PAGE 6: Vulnerability Profiles
# ===========================================================================

def view_vulnerability_profiles(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    st.markdown('<div class="page-title">📊 Vulnerability Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Composite Maternal, Child, and Overall Healthcare Gap Indices.</div>', unsafe_allow_html=True)
    _warn_box(
        "These indices are exploratory composites for analytical triage. "
        "They are NOT validated clinical vulnerability scores."
    )

    imputed_df = data["imputed_df"]
    assignments = data["country_assignments"]
    gaps_df = analytics.get("gaps_df")

    if gaps_df is None:
        st.error("Gap data unavailable.")
        return

    # --- Weight sliders ---
    _section("Customise Index Weights", "⚙️")
    col_a, col_b, col_c = st.columns(3)
    alpha = col_a.slider(
        "Maternal weight (α)",
        0.0, 1.0, 0.5, 0.05,
        help="α=0.5 means equal weight; α=1.0 means maternal index only.",
    )
    _info_box(f"Overall Index = {alpha:.2f} × Maternal Index + {1-alpha:.2f} × Child Index")

    # Recompute with custom alpha
    mhgi = compute_maternal_gap_index(gaps_df)
    chgi = compute_child_gap_index(gaps_df)
    overall = compute_overall_index(mhgi, chgi, alpha=alpha)

    vuln_custom = gaps_df[["CountryName"]].copy()
    vuln_custom["maternal_gap_index"] = mhgi.values
    vuln_custom["child_gap_index"] = chgi.values
    vuln_custom["overall_gap_index"] = overall.values
    vuln_custom = vuln_custom.sort_values("overall_gap_index", ascending=False).reset_index(drop=True)

    st.divider()

    # --- Index ranking chart ---
    _section("Overall Gap Index Ranking", "🏆")
    top_n = st.slider("Show top N countries", 10, len(vuln_custom), 20, 5)
    top_df = vuln_custom.head(top_n)

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=top_df["maternal_gap_index"], y=top_df["CountryName"],
        name="Maternal Gap", orientation="h",
        marker_color="#EF4444",
    ))
    fig_rank.add_trace(go.Bar(
        x=top_df["child_gap_index"], y=top_df["CountryName"],
        name="Child Gap", orientation="h",
        marker_color="#3B82F6",
    ))
    fig_rank.update_layout(
        barmode="group",
        title=f"Top {top_n} Countries — Maternal vs Child Gap Index",
        height=500,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig_rank, width="stretch")

    st.divider()

    # --- Cluster profiles ---
    cluster_col = next(
        (c for c in ["KMeans_K4", "KMeans_Cluster", "Cluster"] if c in assignments.columns),
        None,
    )
    if cluster_col and analytics.get("intel_report"):
        _section("Cluster Gap Profiles", "🔬")
        report = analytics["intel_report"]
        profiles = report["profiles"]

        # Z-score profiles — merge imputed_df with assignments to get aligned labels
        radar_inds = [i for i in PROFILE_INDICATORS if i in imputed_df.columns][:8]
        z_profiles = None
        if cluster_col and cluster_col in assignments.columns:
            try:
                _merged = imputed_df.merge(
                    assignments[["CountryName", cluster_col]],
                    on="CountryName", how="left",
                )
                _labels = _merged[cluster_col].fillna(0).astype(int).values
                if len(_labels) == len(_merged):
                    z_profiles = compute_z_score_profiles(_merged.drop(columns=[cluster_col]), _labels)
            except Exception:
                z_profiles = None

        # Cluster descriptions
        for cid in sorted(report["descriptions"].keys()):
            with st.expander(f"Cluster {cid} ({report['cluster_sizes'].get(cid, '?')} countries)", expanded=(cid == 0)):
                st.markdown(f'<div class="narrative">{report["descriptions"][cid]}</div>', unsafe_allow_html=True)

                if cid in profiles.index:
                    p = profiles.loc[cid]
                    p_inds = [i for i in PROFILE_INDICATORS if i in p.index]
                    if p_inds:
                        vals = p[p_inds].dropna()
                        global_means = imputed_df[[i for i in p_inds if i in imputed_df.columns]].mean()
                        compare = pd.DataFrame({
                            "Indicator": [i.replace("_", " ").title() for i in vals.index],
                            "Cluster Mean": vals.values.round(1),
                            "Global Mean": global_means.reindex(vals.index).values.round(1),
                        })
                        st.dataframe(compare, width="stretch")

        # Z-score heatmap across clusters
        if z_profiles is not None and not z_profiles.empty:
            _section("Cluster Z-Score Deviation Heatmap", "🌡️")
            z_display = z_profiles.copy()
            z_display.columns = [c.replace("_", " ").title() for c in z_display.columns]
            z_display.index = [f"Cluster {i}" for i in z_display.index]
            fig_z = px.imshow(
                z_display.astype(float),
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                aspect="auto",
                title="Deviation from Global Mean (Z-Score) per Cluster",
                labels=dict(color="Z-Score"),
            )
            fig_z.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(fig_z, width="stretch")
            st.markdown(
                '<div class="banner banner-info">Blue = below global mean · Red = above global mean. '
                'For lower-is-better indicators (e.g. mortality), red signals higher burden.</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Full table ---
    _section("Full Vulnerability Index Table", "📋")
    if "SurveyYear" in imputed_df.columns:
        vuln_custom = vuln_custom.merge(
            imputed_df[["CountryName", "SurveyYear"]], on="CountryName", how="left"
        )
    st.dataframe(vuln_custom.round(4), width="stretch", height=400)
    st.download_button(
        "📥 Download Vulnerability Indices (CSV)",
        data=vuln_custom.to_csv(index=False),
        file_name="vulnerability_indices.csv",
        mime="text/csv",
        key=f"dl_vuln_{int(alpha*100)}",
    )


# ===========================================================================
# PAGE 7: Resource Priority Planner
# ===========================================================================

def view_priority_planner(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    st.markdown('<div class="page-title">🗺️ Resource Priority Planner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Rank regions by configurable healthcare gap weights. '
        'Tier categories are relative analytical groupings — not official classifications.</div>',
        unsafe_allow_html=True,
    )

    gaps_df = analytics.get("gaps_df")
    if gaps_df is None:
        st.error("Gap data unavailable.")
        return

    # --- Weight configuration ---
    _section("Configure Priority Weights", "⚙️")
    col1, col2 = st.columns(2)
    w_mat = col1.slider("Maternal gap weight", 0.0, 1.0, 0.5, 0.05)
    w_chi = 1.0 - w_mat
    col2.metric("Child gap weight (auto)", f"{w_chi:.2f}")

    n_tiers = st.radio("Priority tiers", [2, 3, 4], index=1, horizontal=True)

    custom_weights = {"maternal_gap": w_mat, "child_gap": w_chi}

    # Recompute vulnerability with current alpha
    alpha = w_mat
    mhgi = compute_maternal_gap_index(gaps_df)
    chgi = compute_child_gap_index(gaps_df)
    overall = compute_overall_index(mhgi, chgi, alpha=alpha)
    vuln_live = gaps_df[["CountryName"]].copy()
    vuln_live["maternal_gap_index"] = mhgi.values
    vuln_live["child_gap_index"] = chgi.values
    vuln_live["overall_gap_index"] = overall.values

    scored = compute_priority_scores(vuln_live, weights=custom_weights)
    scored["priority_tier"] = categorise_priorities(scored["priority_score"], n_tiers=n_tiers)

    st.divider()

    # --- Priority map ---
    if "DHS_CountryCode" in data["imputed_df"].columns:
        _section("Priority Map", "🗺️")
        map_df = data["imputed_df"][["CountryName", "DHS_CountryCode", "SurveyYear"]].merge(
            scored[["CountryName", "priority_score"]], on="CountryName", how="left"
        )
        fig_map = create_choropleth_map(
            map_df, value_col="priority_score",
            title="Priority Score by Country",
            colour_scale="YlOrRd",
            hover_cols=["SurveyYear"],
            range_color=(0, scored["priority_score"].max()),
        )
        if fig_map:
            st.plotly_chart(fig_map, width="stretch")

    st.divider()

    # --- Ranked table ---
    _section("Priority Rankings", "📋")
    tier_filter = st.multiselect(
        "Filter by Tier",
        options=scored["priority_tier"].unique().tolist(),
        default=scored["priority_tier"].unique().tolist(),
    )
    display_scored = scored[scored["priority_tier"].isin(tier_filter)]

    # Format priority_tier with colour badges
    def _style_tier(val: str) -> str:
        if "Higher" in val:
            return "background-color: #FEE2E2; color: #991B1B"
        elif "Moderate" in val:
            return "background-color: #FEF3C7; color: #92400E"
        return "background-color: #D1FAE5; color: #065F46"

    display_cols = ["priority_rank", "CountryName", "maternal_gap_index", "child_gap_index",
                    "priority_score", "priority_tier"]
    display_cols = [c for c in display_cols if c in display_scored.columns]

    styled = display_scored[display_cols].style.map(
        _style_tier, subset=["priority_tier"]
    ).format({
        "maternal_gap_index": "{:.4f}",
        "child_gap_index": "{:.4f}",
        "priority_score": "{:.2f}",
    })
    st.dataframe(styled, width="stretch", height=450)

    st.divider()

    # --- Scenario comparison ---
    _section("Scenario Analysis", "🎭")
    _info_box("Compare how regional rankings shift under different weighting assumptions.")

    scenario_df = run_scenario_analysis(vuln_live, PREDEFINED_SCENARIOS)

    # Rank stability metric
    if "rank_stability" in scenario_df.columns:
        fig_stab = px.histogram(
            scenario_df, x="rank_stability",
            title="Rank Stability Distribution (max rank change across scenarios)",
            color_discrete_sequence=["#7C3AED"],
        )
        st.plotly_chart(fig_stab, width="stretch")

    with st.expander("Full Scenario Comparison Table"):
        st.dataframe(scenario_df.set_index("CountryName"), width="stretch")

    st.divider()

    # --- Sensitivity analysis ---
    _section("Weight Sensitivity Analysis (Monte Carlo)", "📈")
    with st.expander("Run sensitivity analysis (may take a few seconds)"):
        n_mc = st.slider("Monte Carlo samples", 50, 500, 100, 50)
        if st.button("Run Sensitivity Analysis"):
            with st.spinner("Sampling weight space…"):
                from analytics.prioritisation import compute_weight_sensitivity
                sens = compute_weight_sensitivity(vuln_live, n_samples=n_mc)
            st.dataframe(sens.head(20), width="stretch")
            fig_sens = px.scatter(
                sens, x="rank_mean", y="rank_std",
                hover_name="CountryName",
                title="Rank Mean vs Rank Std Dev — Stability Assessment",
                labels={"rank_mean": "Mean Priority Rank", "rank_std": "Rank Std Dev"},
                color="rank_std",
                color_continuous_scale="Reds",
            )
            st.plotly_chart(fig_sens, width="stretch")

    # --- Download ---
    st.download_button(
        "📥 Download Priority Rankings (CSV)",
        data=scored.to_csv(index=False),
        file_name="priority_rankings_custom.csv",
        mime="text/csv",
        key=f"dl_prio_w{int(w_mat*100)}",
    )


# ===========================================================================
# PAGE 8: Reports & Exports
# ===========================================================================

def view_reports(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    st.markdown('<div class="page-title">📄 Reports &amp; Exports</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Download all analysis outputs in machine-readable formats.</div>', unsafe_allow_html=True)

    _section("Dataset Artefacts", "🗄️")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "📥 Cleaned Dataset",
            data=data["cleaned_df"].to_csv(index=False),
            file_name="maternal_child_health_cleaned.csv",
            mime="text/csv", key="rep_dl_clean",
        )
        st.caption(f"{len(data['cleaned_df'])} countries, post-missingness filter")

    with col2:
        st.download_button(
            "📥 Imputed Dataset",
            data=data["imputed_df"].to_csv(index=False),
            file_name="maternal_child_health_imputed.csv",
            mime="text/csv", key="rep_dl_imputed",
        )
        st.caption("Median-imputed feature matrix")

    with col3:
        st.download_button(
            "📥 Standardised Dataset",
            data=data["standardized_df"].to_csv(index=False),
            file_name="maternal_child_health_standardized.csv",
            mime="text/csv", key="rep_dl_std",
        )
        st.caption("Z-scored feature matrix used for clustering")

    st.divider()
    _section("Clustering Results", "🔬")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.download_button(
            "📥 Cluster Assignments",
            data=data["country_assignments"].to_csv(index=False),
            file_name="country_cluster_assignments.csv",
            mime="text/csv", key="rep_dl_assign",
        )
    with col5:
        st.download_button(
            "📥 K-Means Metrics",
            data=data["kmeans_metrics"].to_csv(index=False),
            file_name="kmeans_evaluation.csv",
            mime="text/csv", key="rep_dl_km",
        )
    with col6:
        st.download_button(
            "📥 Hierarchical Metrics",
            data=data["hier_metrics"].to_csv(index=False),
            file_name="hierarchical_linkage_evaluation.csv",
            mime="text/csv", key="rep_dl_hier",
        )

    st.divider()
    _section("Analytics Outputs", "📊")
    col7, col8, col9 = st.columns(3)

    with col7:
        if analytics.get("gaps_df") is not None:
            st.download_button(
                "📥 Healthcare Gap Scores",
                data=analytics["gaps_df"].to_csv(index=False),
                file_name="healthcare_gaps.csv",
                mime="text/csv", key="rep_dl_gaps",
            )
            st.caption("Per-country per-indicator normalised gap scores")

    with col8:
        if analytics.get("vuln_df") is not None:
            st.download_button(
                "📥 Vulnerability Indices",
                data=analytics["vuln_df"].to_csv(index=False),
                file_name="vulnerability_indices.csv",
                mime="text/csv", key="rep_dl_vuln",
            )
            st.caption("Maternal, Child, Overall Gap Indices")

    with col9:
        if analytics.get("priority_df") is not None:
            st.download_button(
                "📥 Priority Rankings",
                data=analytics["priority_df"].to_csv(index=False),
                file_name="priority_rankings.csv",
                mime="text/csv", key="rep_dl_prio",
            )
            st.caption("Baseline equal-weight priority ranking")

    if analytics.get("scenario_df") is not None:
        st.download_button(
            "📥 Scenario Comparison",
            data=analytics["scenario_df"].to_csv(index=False),
            file_name="scenario_comparison.csv",
            mime="text/csv", key="rep_dl_scen",
        )

    st.divider()
    _section("Benchmark Reference Table", "📚")
    bm_df = get_benchmark_table()
    st.dataframe(bm_df, width="stretch")
    st.download_button(
        "📥 Download Benchmark Table",
        data=bm_df.to_csv(index=False),
        file_name="who_unicef_benchmarks.csv",
        mime="text/csv", key="rep_dl_bm",
    )

    st.divider()
    _section("Full Analysis Report (Markdown)", "📝")
    readme_path = config.BASE_DIR / "README.md"
    if readme_path.exists():
        readme_text = readme_path.read_text()
        with st.expander("Preview README.md"):
            st.markdown(readme_text[:5000] + "\n\n*(truncated — download for full report)*")
        st.download_button(
            "📥 Download Full README.md",
            data=readme_text,
            file_name="README.md",
            mime="text/markdown", key="rep_dl_readme",
        )

    st.divider()
    _section("PCA Outputs", "🔭")
    col10, col11 = st.columns(2)
    with col10:
        st.download_button(
            "📥 PCA Variance Explained",
            data=data["pca_var"].to_csv(index=False),
            file_name="pca_variance_explained.csv",
            mime="text/csv", key="rep_dl_pcavar",
        )
    with col11:
        st.download_button(
            "📥 PCA Feature Loadings",
            data=data["pca_loadings"].to_csv(index=False),
            file_name="pca_feature_loadings.csv",
            mime="text/csv", key="rep_dl_pcaload",
        )

    st.divider()
    with st.expander("ℹ️ Methodological Notes"):
        st.markdown("""
**Gap Normalisation**: For higher-is-better indicators: `gap = max(0, (benchmark - observed) / benchmark)`. For lower-is-better indicators: `gap = max(0, (observed - benchmark) / benchmark)`.

**Vulnerability Index**: Weighted average of normalised indicator gaps. Default α = 0.5 (equal maternal/child weight).

**Priority Score**: `100 × (w_mat × MHGI + w_child × CHGI)` with configurable weights.

**Priority Tiers**: Quantile-based thresholds within the analytical sample. Categories are relative (not absolute).

**Data Source**: DHS Program REST API (https://api.dhsprogram.com/), real indicator values, 70 countries retained after 35% missingness filter.

**Benchmarks**: WHO Global Health Observatory, UNICEF SOWC 2023, SDG targets 3.1, 3.2, 3.7, 3.8, 6.1, 6.2, IA2030.
        """)


# ===========================================================================
# PAGE 8: Scenario Simulator
# ===========================================================================

def view_scenario_simulator(data: dict, analytics: dict, feat_cols: List[str]) -> None:
    from analytics.scenario_simulator import (
        SIMULATABLE_INDICATORS, run_scenario_comparison, build_before_after_df
    )

    st.markdown('<div class="page-title">🔬 Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Model hypothetical improvements and observe their effect on gap indices and priority rankings.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="banner banner-warn">⚠️ <strong>Hypothetical Analysis Only.</strong> '
        'Simulated values are projections for analytical exploration. '
        'The original dataset is never modified. Results must not be presented as forecasts or policy commitments.</div>',
        unsafe_allow_html=True,
    )

    imputed_df = data["imputed_df"]
    countries = sorted(imputed_df["CountryName"].unique().tolist())

    # --- Country selector ---
    st.markdown('<div class="section-heading">🌍 Select Country to Simulate</div>', unsafe_allow_html=True)
    country = st.selectbox("Country", countries)

    # Show current indicator values for this country
    country_row = imputed_df[imputed_df["CountryName"] == country]
    if country_row.empty:
        st.error(f"No data for {country}.")
        return

    st.markdown('<div class="section-heading">⚙️ Adjust Hypothetical Indicator Values</div>', unsafe_allow_html=True)
    st.caption("Move sliders to simulate improvements. Only indicators with available data are shown.")

    adjustments: Dict[str, float] = {}
    available = {k: v for k, v in SIMULATABLE_INDICATORS.items() if k in imputed_df.columns}

    # Layout: 3 columns of sliders
    items = list(available.items())
    cols = st.columns(3)
    for i, (ind, meta) in enumerate(items):
        current_val = float(country_row[ind].iloc[0]) if not pd.isna(country_row[ind].iloc[0]) else 50.0
        new_val = cols[i % 3].slider(
            meta["label"],
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=round(current_val, 1),
            step=float(meta["step"]),
            help=f"Current value: {current_val:.1f}",
        )
        if abs(new_val - current_val) > 0.01:
            adjustments[ind] = new_val

    st.divider()

    if not adjustments:
        st.info("👆 Move at least one slider above the current value to run a simulation.")
        return

    if st.button("▶ Run Simulation", type="primary"):
        with st.spinner(f"Running scenario for {country}…"):
            try:
                result = run_scenario_comparison(imputed_df, country, adjustments)
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                return

        # --- Summary metrics ---
        st.markdown('<div class="section-heading">📊 Simulation Results</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="hypo-label">Hypothetical</span> results for <strong>{country}</strong>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        def _delta_color(d):
            return "gap-low" if d < 0 else ("gap-high" if d > 0 else "")

        c1.metric("Maternal Gap Index",
                  f"{result['scenario_maternal']:.4f}",
                  f"{result['delta_maternal']:+.4f}",
                  delta_color="inverse")
        c2.metric("Child Gap Index",
                  f"{result['scenario_child']:.4f}",
                  f"{result['delta_child']:+.4f}",
                  delta_color="inverse")
        c3.metric("Overall Gap Index",
                  f"{result['scenario_overall']:.4f}",
                  f"{result['delta_overall']:+.4f}",
                  delta_color="inverse")
        rank_delta = result.get("delta_rank")
        rank_txt = f"{rank_delta:+d}" if rank_delta is not None else "N/A"
        c4.metric("Priority Rank",
                  str(result["scenario_rank"]) if result["scenario_rank"] else "N/A",
                  rank_txt,
                  delta_color="inverse")

        st.divider()

        # --- Before / After indicator table ---
        st.markdown('<div class="section-heading">📋 Before vs. After — Adjusted Indicators</div>', unsafe_allow_html=True)
        ba_df = build_before_after_df(result)
        ba_df["Change"] = ba_df["Change"].map(lambda x: f"{x:+.1f}")
        ba_df["Baseline"] = ba_df["Baseline"].map(lambda x: f"{x:.1f}")
        ba_df["Hypothetical"] = ba_df["Hypothetical"].map(lambda x: f"{x:.1f}")
        st.dataframe(ba_df, width="stretch", hide_index=True)

        st.divider()

        # --- Gap index comparison chart ---
        st.markdown('<div class="section-heading">📉 Gap Index: Baseline vs. Hypothetical</div>', unsafe_allow_html=True)
        compare_data = pd.DataFrame({
            "Scenario": ["Baseline", "Hypothetical"],
            "Maternal Gap": [result["baseline_maternal"], result["scenario_maternal"]],
            "Child Gap":    [result["baseline_child"],    result["scenario_child"]],
            "Overall Gap":  [result["baseline_overall"],  result["scenario_overall"]],
        })
        fig_comp = px.bar(
            compare_data.melt(id_vars="Scenario", var_name="Index", value_name="Value"),
            x="Index", y="Value", color="Scenario", barmode="group",
            color_discrete_map={"Baseline": "#0E7490", "Hypothetical": "#F97316"},
            title=f"Gap Index Comparison — {country}",
            labels={"Value": "Gap Index", "Index": ""},
        )
        fig_comp.update_layout(height=380, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_comp, width="stretch")

        # --- Full ranking comparison (top 20) ---
        st.divider()
        st.markdown('<div class="section-heading">🏆 Priority Ranking: Baseline vs. Hypothetical (Top 20)</div>', unsafe_allow_html=True)
        b_top = result["baseline_prio"][["CountryName","priority_score","priority_rank"]].head(20)
        s_top = result["scenario_prio"][["CountryName","priority_score","priority_rank"]].head(20)
        merge = b_top.merge(s_top, on="CountryName", suffixes=("_base","_sim"))
        merge["Rank Change"] = merge["priority_rank_sim"] - merge["priority_rank_base"]
        merge = merge.rename(columns={
            "priority_rank_base":"Rank (Baseline)",
            "priority_rank_sim":"Rank (Hypothetical)",
            "priority_score_base":"Score (Baseline)",
            "priority_score_sim":"Score (Hypothetical)",
        })
        # Highlight the simulated country
        def _highlight(row):
            if row["CountryName"] == country:
                return ["background-color: #FEF9C3"] * len(row)
            return [""] * len(row)
        styled = merge[["CountryName","Rank (Baseline)","Rank (Hypothetical)","Rank Change","Score (Baseline)","Score (Hypothetical)"]].style.apply(_highlight, axis=1)
        st.dataframe(styled, width="stretch", hide_index=True)

        st.markdown('<div class="disclaimer">These results are hypothetical projections derived from adjusted input values. They are not predictions, clinical assessments, or policy recommendations.</div>', unsafe_allow_html=True)

        # Download
        sim_export = result["scenario_prio"].copy()
        sim_export["scenario_note"] = f"Hypothetical simulation for {country}"
        st.download_button(
            "📥 Download Scenario Results (CSV)",
            sim_export.to_csv(index=False),
            f"scenario_{country.replace(' ','_')}.csv",
            "text/csv",
        )
