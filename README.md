# 🏥 Maternal & Child Health Gap Intelligence and Resource Prioritisation System

> **Unsupervised Machine Learning for Global Health Analytics**  
> Identifying countries experiencing maternal and child healthcare gaps, discovering regions with similar challenges through clustering, and helping public-health analysts prioritise areas for investigation and resource planning.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Key Features](#3-key-features)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Dataset & Data Sources](#6-dataset--data-sources)
7. [ML Methodology](#7-ml-methodology)
8. [Evaluation Results](#8-evaluation-results)
9. [WHO/SDG Benchmarks](#9-whosdg-benchmarks)
10. [Installation](#10-installation)
11. [Running the Application](#11-running-the-application)
12. [Docker Deployment](#12-docker-deployment)
13. [Project Structure](#13-project-structure)
14. [Dashboard Pages](#14-dashboard-pages)
15. [Analytics Modules](#15-analytics-modules)
16. [Testing](#16-testing)
17. [Deployment Options](#17-deployment-options)
18. [Limitations & Responsible Use](#18-limitations--responsible-use)
19. [Future Enhancements](#19-future-enhancements)

---

## 1. Project Overview

This system applies unsupervised machine learning to Demographic and Health Surveys (DHS) Program data from **70 countries** to:

- **Detect** maternal and child healthcare gaps relative to WHO/UNICEF/SDG benchmarks
- **Cluster** countries by healthcare profile similarity using K-Means, K-Medoids, and Hierarchical methods
- **Rank** countries by configurable priority scores for resource planning
- **Simulate** hypothetical indicator improvements to estimate their impact on gap indices

The application is built as a production-grade Streamlit dashboard with a Clinical Intelligence design theme, interactive Plotly visualisations, and a fully tested analytics backend.

---

## 2. Problem Statement

Maternal and child health outcomes vary dramatically across low- and middle-income countries. Public-health analysts face a challenge: **which countries need attention first, and what do they have in common?**

Traditional country rankings use single indicators. This system uses **multi-indicator unsupervised clustering** to:

1. Identify countries with structurally similar healthcare challenges
2. Quantify gaps across 17 indicators vs. documented benchmarks
3. Provide configurable, evidence-based prioritisation
4. Enable scenario analysis to explore hypothetical interventions

---

## 3. Key Features

| Feature | Description |
|---------|-------------|
| 🌍 **70-Country Analysis** | Real DHS Program data across Africa, Asia, and Latin America |
| 📉 **Gap Detection** | 17 indicators vs. WHO/UNICEF/SDG benchmarks |
| 🧪 **Multi-Algorithm Clustering** | K-Means, K-Medoids (Manhattan), Hierarchical (Ward/Average/Complete) |
| 🔭 **PCA Studio** | Interactive biplot, scree plot, loading heatmap |
| 📊 **Vulnerability Profiles** | Composite gap indices (Maternal, Child, Overall) |
| 🗺️ **Priority Planner** | Configurable weights, tier categories, choropleth maps |
| 🎛️ **Scenario Simulator** | Hypothetical indicator adjustments and re-ranking |
| 📄 **Reports & Exports** | All outputs downloadable as CSV/JSON |
| 🌙 **Dark Mode** | Full light/dark theme toggle |
| 🐳 **Docker Ready** | Containerised deployment with health checks |

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard (app/)                     │
│  app.py (navigation, theme, data loading)                        │
│  views.py (9 page view functions)                                │
└───────────────────┬─────────────────────────────────────────────┘
                    │ cached @st.cache_data
┌───────────────────▼─────────────────────────────────────────────┐
│                   Analytics Layer (analytics/)                   │
│  gap_detection.py  │  vulnerability.py  │  cluster_intelligence  │
│  prioritisation.py │  geo_visualisation │  scenario_simulator    │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│                   ML Pipeline (src/)                             │
│  data_acquisition → preprocessing → clustering → PCA → eval     │
│  kmeans.py  │  hierarchical.py  │  kmedoids.py  │  pca.py        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│                   Data Layer                                     │
│  data/raw/      (DHS API responses)                              │
│  data/processed/ (cleaned, imputed, z-scored matrices)           │
│  outputs/tables/ (metrics CSVs, analytics CSVs)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Technology Stack

| Category | Library | Version |
|----------|---------|---------|
| **Language** | Python | 3.12 |
| **Dashboard** | Streamlit | 1.67 |
| **ML** | scikit-learn | ≥1.4 |
| **ML — K-Medoids** | scikit-learn-extra | ≥0.3 |
| **Numerical** | NumPy, SciPy | ≥1.26 / ≥1.13 |
| **Data** | Pandas | ≥2.1 |
| **Visualisation** | Plotly | ≥5.19 |
| **Statistics** | Statsmodels | ≥0.14 |
| **HTTP** | Requests | ≥2.31 |
| **Testing** | pytest | ≥8.0 |
| **Containers** | Docker + Compose | — |

---

## 6. Dataset & Data Sources

**Primary Source**: [DHS Program](https://dhsprogram.com/) — Demographic and Health Surveys, funded in part by the [Bill & Melinda Gates Foundation](https://www.gatesfoundation.org/).

Data is fetched via the [DHS API](https://api.dhsprogram.com/) (StatCompiler endpoint) and cached locally.

### Indicators Used (17 total)

**Maternal Health (9)**
| Code | Indicator |
|------|-----------|
| `anc_visits_pct` | Antenatal Care — ≥1 visit (%) |
| `anc4_visits_pct` | Antenatal Care — ≥4 visits (%) |
| `skilled_birth_attendance` | Skilled Birth Attendance (%) |
| `institutional_delivery` | Institutional Delivery (%) |
| `postnatal_care_mother` | Postnatal Care — Mother (%) |
| `family_planning_modern` | Modern Contraceptive Prevalence (%) |
| `unmet_need_family_planning` | Unmet Need for Family Planning (%) |
| `tetanus_protection` | Tetanus Protection (%) |
| `iron_supplementation` | Iron Supplementation (%) |

**Child Health (8)**
| Code | Indicator |
|------|-----------|
| `infant_mortality_rate` | Infant Mortality Rate (per 1,000 LB) |
| `under5_mortality_rate` | Under-5 Mortality Rate (per 1,000 LB) |
| `dpt_vaccination` | DTP3 Vaccination Coverage (%) |
| `measles_vaccination` | Measles Vaccination Coverage (%) |
| `child_stunting` | Stunting Prevalence (%) |
| `child_wasting` | Wasting Prevalence (%) |
| `oral_rehydration_therapy` | ORT for Diarrhoea (%) |
| `postnatal_care_newborn` | Postnatal Care — Newborn (%) |

### Dataset Dimensions
- **70 countries** (mostly Sub-Saharan Africa + South/Southeast Asia)
- **Survey years**: approximately 2010–2023
- **Missing data**: handled with median imputation per indicator before clustering

---

## 7. ML Methodology

### 7.1 Preprocessing Pipeline

```
Raw DHS API data
  → Cross-sectional extraction (most recent survey per country)
  → Schema validation & type coercion
  → Missingness filter (drop countries with >50% missing indicators)
  → Median imputation (per-indicator, using non-missing values)
  → Z-score standardisation (mean=0, std=1) for clustering
```

### 7.2 Clustering Algorithms

#### K-Means (Primary)
- **Distance**: Euclidean on z-scored features
- **K range**: 2–10 evaluated
- **Optimal K**: 4 (Davies-Bouldin minimisation + Silhouette)
- **Initialisation**: k-means++ (10 restarts)
- **Convergence**: up to 500 iterations

#### K-Medoids (Validation)
- **Distance**: Manhattan (L1), more robust to outliers
- **K**: 4 (matched to K-Means for head-to-head comparison)
- **Agreement**: ARI = 0.409 vs. K-Means (moderate agreement, expected due to distance metric differences)

#### Hierarchical Clustering
- **Linkages evaluated**: Ward, Average, Complete
- **Best cophenetic correlation**: Average linkage (r = 0.757)
- **Cut**: 4 clusters to match K-Means

### 7.3 Dimensionality Reduction

**PCA** applied to z-scored feature matrix:
- PC1 explains the largest variance (see dashboard for exact values)
- PC2 reveals a second major axis of variation
- 2D biplot used for cluster visualisation
- Loadings heatmap shows indicator contributions

### 7.4 Gap Detection

For each indicator *i* and country *c*:

```
gap(c, i) = |observed(c, i) − benchmark(i)| / benchmark(i)
           capped at 1.0 for lower-is-better indicators (e.g. mortality)
           = 0 if benchmark is met or exceeded
```

### 7.5 Vulnerability Indices

Composite indices computed as weighted averages of normalised gap scores:

- **Maternal Gap Index** = weighted mean of 9 maternal indicator gaps
- **Child Gap Index** = weighted mean of 8 child indicator gaps  
- **Overall Gap Index** = weighted combination (default 50/50)

Default weights are uniform (equal). Users can reconfigure via the Priority Planner page.

### 7.6 Priority Scoring

```
Priority Score = w_mat × Maternal_Gap_Index
              + w_child × Child_Gap_Index
              + w_pop × Population_Weight_Factor   (optional)
```

Countries categorised into tiers: **Higher**, **Moderate**, **Lower** (relative to the sample distribution — not absolute thresholds).

---

## 8. Evaluation Results

All metrics computed from actual model runs on the 70-country DHS dataset.

### K-Means (Optimal K=4)

| Metric | Value |
|--------|-------|
| Silhouette (Euclidean) | **0.171** |
| Davies-Bouldin | **1.631** |
| Calinski-Harabasz | **18.6** |
| WCSS | 644.4 |
| Cluster sizes | 20, 20, 5, 25 |

> A Silhouette score of 0.171 is typical for real-world multi-indicator health data with overlapping country profiles. It reflects genuine cluster structure even though boundaries are soft.

### K-Medoids (Manhattan, K=4)

| Metric | Value |
|--------|-------|
| Silhouette (Manhattan) | 0.106 |
| Davies-Bouldin | 2.191 |
| ARI vs. K-Means | 0.409 |

### Hierarchical Clustering

| Linkage | Cophenetic r | Silhouette | Davies-Bouldin |
|---------|-------------|------------|----------------|
| Ward | 0.572 | 0.145 | 1.809 |
| Average | **0.757** | 0.144 | 0.980 |
| Complete | 0.646 | 0.160 | 1.789 |

> Average linkage gives the best dendrogram fidelity (cophenetic r = 0.757). Ward linkage is most comparable to K-Means in cluster compactness.

---

## 9. WHO/SDG Benchmarks

| Indicator | Benchmark | Unit | Source | Year |
|-----------|-----------|------|--------|------|
| ANC ≥1 visit | 100.0 | % | WHO | 2016 |
| ANC ≥4 visits | 90.0 | % | WHO ANC Guidelines | 2016 |
| Skilled birth attendance | 90.0 | % | SDG 3.1.2 | 2015 |
| Institutional delivery | 90.0 | % | SDG 3.1 | 2015 |
| Postnatal care (mother) | 90.0 | % | WHO | 2013 |
| Modern contraception | 75.0 | % | FP2030 Target | 2020 |
| Unmet family planning need | 10.0 | % | SDG 3.7 | 2015 |
| Tetanus protection | 90.0 | % | WHO EPI | 2020 |
| Iron supplementation | 90.0 | % | WHO | 2016 |
| Infant mortality rate | 12.0 | per 1,000 LB | SDG 3.2.2 | 2015 |
| Under-5 mortality rate | 25.0 | per 1,000 LB | SDG 3.2.1 | 2015 |
| DTP3 vaccination | 90.0 | % | WHO EPI | 2020 |
| Measles vaccination | 95.0 | % | WHO EPI | 2020 |
| Stunting prevalence | 10.0 | % | WHA Global Target | 2025 |
| Wasting prevalence | 5.0 | % | WHA Global Target | 2025 |
| ORT for diarrhoea | 80.0 | % | WHO/UNICEF | 2000 |
| Postnatal care (newborn) | 90.0 | % | WHO | 2013 |

---

## 10. Installation

### Prerequisites

- Python 3.11+ (tested on 3.12)
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-org/mch-gap-intelligence.git
cd mch-gap-intelligence

# 2. Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env to set DHS_API_KEY if needed (optional for public endpoints)
```

### Environment Variables (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DHS_API_KEY` | *(empty)* | DHS API key (optional, public endpoints work without it) |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `STREAMLIT_PORT` | `8501` | Port for the Streamlit server |

---

## 11. Running the Application

### Option A — Run pipeline then dashboard (first run)

```bash
# Step 1: Run the full ML pipeline (fetches data, trains models, saves outputs)
MPLCONFIGDIR=/tmp/matplotlib python3 -m src.pipeline_runner

# Step 2: Launch the dashboard
MPLCONFIGDIR=/tmp/matplotlib streamlit run app/app.py
```

### Option B — Dashboard only (pre-computed outputs exist)

```bash
streamlit run app/app.py
```

The dashboard will be available at **http://localhost:8501**.

### Streamlit Config

A `.streamlit/config.toml` file is provided for production settings:

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#0F2B5B"
backgroundColor = "#F8FAFC"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#1E293B"
font = "sans serif"
```

---

## 12. Docker Deployment

```bash
# Build and start
docker-compose up --build

# Or build manually
docker build -t mch-gap-intelligence .
docker run -p 8501:8501 mch-gap-intelligence
```

The container exposes port 8501. A health check pings `/_stcore/health` every 30 seconds.

---

## 13. Project Structure

```
mch-gap-intelligence/
├── app/
│   ├── app.py                   # Main Streamlit entry point (navigation, CSS, data loading)
│   └── views.py                 # 9 page view functions
├── src/
│   ├── data_acquisition.py      # DHS API fetch + caching
│   ├── preprocessing.py         # Schema validation, imputation, z-scoring
│   ├── cross_sectional.py       # Latest-survey-per-country extraction
│   ├── clustering.py            # K-Means experiments (K=2–10)
│   ├── hierarchical.py          # Hierarchical clustering + evaluation
│   ├── kmedoids.py              # K-Medoids (Manhattan distance)
│   ├── dimensionality_reduction.py  # PCA + loading extraction
│   ├── interpretation.py        # Cluster narrative generation
│   └── pipeline_runner.py       # End-to-end pipeline orchestrator
├── analytics/
│   ├── gap_detection.py         # WHO/SDG benchmark gap calculation
│   ├── vulnerability.py         # Composite gap index computation
│   ├── cluster_intelligence.py  # Cluster profiling & z-score analysis
│   ├── prioritisation.py        # Priority scoring & tier categorisation
│   ├── geo_visualisation.py     # Choropleth & cluster maps (Plotly)
│   └── scenario_simulator.py    # Hypothetical indicator simulation
├── tests/
│   ├── test_analytics.py        # 62 unit tests for analytics modules
│   └── test_pipeline.py         # 8 integration tests for the ML pipeline
├── data/
│   ├── raw/                     # DHS API responses (cached JSON)
│   └── processed/               # Cleaned, imputed, z-scored CSVs
├── outputs/
│   ├── figures/                 # Saved matplotlib/plotly figures
│   ├── models/                  # Saved model objects (joblib)
│   └── tables/                  # All metrics and analytics CSVs
├── notebooks/
│   └── exploration.ipynb        # Exploratory data analysis
├── config.py                    # Centralised configuration constants
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose service definition
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore patterns
└── README.md                    # This document
```

---

## 14. Dashboard Pages

| # | Page | Description |
|---|------|-------------|
| 1 | **Executive Overview** | KPI summary, top gap countries, global choropleth, methodology |
| 2 | **Dataset Explorer** | Browse/filter DHS data, upload custom CSV/Excel, data quality report |
| 3 | **Healthcare Gap Detection** | Benchmark gaps by domain (Maternal/Child/Combined), heatmap, country drill-down |
| 4 | **Clustering Laboratory** | K-Means elbow/silhouette, K-Medoids comparison, Hierarchical dendrogram-style metrics |
| 5 | **PCA Studio** | Scree plot, 2D biplot with cluster colours, loading heatmap |
| 6 | **Vulnerability Profiles** | Composite gap indices, country rankings, sensitivity analysis |
| 7 | **Resource Priority Planner** | Configurable weight sliders, priority tier map, predefined scenarios |
| 8 | **Scenario Simulator** | Hypothetical indicator adjustments, before/after comparison, re-ranking |
| 9 | **Reports & Exports** | Download all datasets and analytics outputs as CSV/JSON |

---

## 15. Analytics Modules

### `analytics/gap_detection.py`
- 17 WHO/SDG benchmarks with direction (higher-is-better / lower-is-better)
- `calculate_indicator_gap()` — normalised gap for a single country/indicator
- `calculate_region_gaps()` — full 70×17 gap matrix
- `calculate_maternal_gap_summary()`, `calculate_child_gap_summary()` — domain summaries
- `get_top_gap_indicators()` — top-N indicators by mean gap
- `get_benchmark_table()` — benchmark reference table with sources

### `analytics/vulnerability.py`
- `compute_maternal_gap_index()`, `compute_child_gap_index()`, `compute_overall_index()`
- `build_vulnerability_dataframe()` — combined index DataFrame
- `compute_sensitivity_analysis()` — robustness of rankings under weight perturbation

### `analytics/cluster_intelligence.py`
- `compute_cluster_profiles()` — mean indicator values per cluster
- `generate_cluster_descriptions()` — narrative descriptions of cluster characteristics
- `compute_z_score_profiles()` — z-score heatmap data for cluster comparison

### `analytics/prioritisation.py`
- `compute_priority_scores()` — weighted priority score per country
- `categorise_priorities()` — tier assignment (Higher/Moderate/Lower)
- `run_scenario_analysis()` — apply predefined weighting scenarios
- `PREDEFINED_SCENARIOS` — 5 pre-built scenarios (Equal, Maternal-Focus, Child-Focus, Mortality-Focus, Immunisation-Focus)

### `analytics/geo_visualisation.py`
- `create_choropleth_map()`, `create_cluster_map()`, `create_gap_layer_map()`
- DHS country code → ISO3 mapping for Plotly geo charts
- `fallback_table()` — table fallback when country codes cannot be mapped

### `analytics/scenario_simulator.py`
- `SIMULATABLE_INDICATORS` — 11 percentage-based indicators (adjustable via sliders)
- `apply_scenario()` — modify a country's indicator values by percentage
- `run_scenario_comparison()` — recompute vulnerability + priority under hypothetical values
- `build_before_after_df()` — before/after indicator table for display

---

## 16. Testing

```bash
# Run the full test suite
MPLCONFIGDIR=/tmp/matplotlib python3 -m pytest tests/ -v

# Run with coverage
MPLCONFIGDIR=/tmp/matplotlib python3 -m pytest tests/ --cov=analytics --cov=src --cov-report=term-missing
```

### Test Results

```
70 passed, 1 warning in 5.05s
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_analytics.py` | 62 | gap_detection, vulnerability, cluster_intelligence, prioritisation |
| `test_pipeline.py` | 8 | data_acquisition, preprocessing, clustering, PCA, hierarchical, kmedoids |

### Test Scope

- Gap calculation correctness (benchmark met → gap = 0, worst case → gap = 1)
- Composite index computation with custom weights
- Priority score ordering consistency
- Cluster profiling shape and column correctness
- Pipeline integration: raw load → cross-sectional → imputation → clustering → PCA
- Edge cases: empty DataFrames, all-missing indicators, duplicate countries, negative mortality values, percentages above 100

---

## 17. Deployment Options

### Local (Development)
```bash
streamlit run app/app.py
```

### Docker (Production)
```bash
docker-compose up --build -d
```

### Streamlit Community Cloud
1. Push to a public GitHub repository
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file path: `app/app.py`
4. Add environment secrets if needed

### Heroku / Railway / Render
Add a `Procfile`:
```
web: streamlit run app/app.py --server.port=$PORT --server.headless=true
```

---

## 18. Limitations & Responsible Use

> **Important**: This system is an analytical research tool, not a clinical decision-support system.

| Limitation | Detail |
|------------|--------|
| **Survey comparability** | DHS indicator definitions and survey methodologies vary by country and survey year |
| **Cross-sectional design** | Most recent survey per country; temporal trends are not modelled |
| **Benchmark applicability** | Global WHO/SDG targets may not reflect regional or national policy priorities |
| **Priority tiers** | Relative analytical categories within this 70-country sample — not official government or WHO classifications |
| **Scenario projections** | Hypothetical values only; not forecasts, clinical assessments, or policy recommendations |
| **Cluster interpretation** | Cluster profiles describe statistical similarity of indicator values — causal mechanisms are not implied |
| **Missing indicators** | Countries with high missingness are excluded; results may not fully represent all low-income settings |

---

## 19. Future Enhancements

- [ ] **Temporal analysis**: Track indicator trends across multiple survey years per country
- [ ] **Sub-national data**: Integrate DHS regional (admin-1) data for within-country analysis
- [ ] **Additional clustering**: DBSCAN for noise detection, GMM for soft cluster membership
- [ ] **Automated narrative generation**: LLM-generated cluster descriptions and policy briefs
- [ ] **Multi-language support**: Dashboard in French, Portuguese, Spanish (key DHS languages)
- [ ] **Real-time DHS API refresh**: Scheduled pipeline runs to incorporate new surveys automatically
- [ ] **User authentication**: Role-based access for public-health organisations
- [ ] **Report PDF export**: Automated PDF generation of full country reports

---

## Acknowledgements

- **Data**: DHS Program (ICF International), funded by USAID and the Bill & Melinda Gates Foundation
- **Benchmarks**: World Health Organization (WHO), UNICEF, Sustainable Development Goals framework
- **ML Libraries**: scikit-learn, scikit-learn-extra, SciPy, Pandas, NumPy
- **Visualisation**: Plotly, Streamlit

---

*Last updated: November 2026 | Python 3.12 | Streamlit 1.67 | 70 countries | 17 indicators*
