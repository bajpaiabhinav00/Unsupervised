"""
Central configuration module for Maternal-Child Health Clustering project.
Defines random seeds, directory paths, indicator definitions, and model parameters.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"

# Specific File Paths
RAW_DATA_FILE = RAW_DATA_DIR / "dhs_maternal_child_indicators_raw.csv"
SURVEY_META_FILE = RAW_DATA_DIR / "dhs_surveys_metadata.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "maternal_child_health_cleaned.csv"
IMPUTED_DATA_FILE = PROCESSED_DATA_DIR / "maternal_child_health_imputed.csv"
STANDARDIZED_DATA_FILE = PROCESSED_DATA_DIR / "maternal_child_health_standardized.csv"
DATA_DICTIONARY_FILE = DATA_DIR / "data_dictionary.json"

# Reproducibility
RANDOM_SEED = 42

# Clustering Parameters
K_RANGE = list(range(2, 11))
OPTIMAL_K_DEFAULT = 4
N_INIT = 20
MAX_ITER = 300

# DHS API Configuration
DHS_API_BASE = "https://api.dhsprogram.com/rest/dhs/data"
DHS_INDICATORS_URL = "https://api.dhsprogram.com/rest/dhs/indicators"

# Selected DHS Indicators Specification
INDICATORS_CONFIG = {
    # Maternal Health
    "RH_ANCN_W_N4P": {
        "short_name": "anc_4plus_coverage",
        "label": "Antenatal Care 4+ Visits",
        "category": "Maternal Health",
        "unit": "% of women (15-49)",
        "definition": "Percentage of women age 15-49 with a live birth in the past 5 years who received 4 or more antenatal care visits.",
        "higher_is_better": True,
    },
    "RH_ANCP_W_SKP": {
        "short_name": "anc_skilled_provider",
        "label": "Antenatal Care from Skilled Provider",
        "category": "Maternal Health",
        "unit": "% of women (15-49)",
        "definition": "Percentage of women age 15-49 who attended antenatal care provided by skilled healthcare personnel (doctor, nurse, midwife).",
        "higher_is_better": True,
    },
    "RH_DELA_C_SKP": {
        "short_name": "skilled_birth_attendance",
        "label": "Skilled Birth Attendance",
        "category": "Maternal Health",
        "unit": "% of live births",
        "definition": "Percentage of live births attended by skilled health personnel (doctor, nurse, midwife, auxiliary midwife).",
        "higher_is_better": True,
    },
    "RH_DELP_C_DHF": {
        "short_name": "institutional_delivery",
        "label": "Institutional Delivery",
        "category": "Maternal Health",
        "unit": "% of live births",
        "definition": "Percentage of live births delivered in a health facility (public or private).",
        "higher_is_better": True,
    },
    "FP_CUSA_W_MOD": {
        "short_name": "modern_contraceptive_prevalence",
        "label": "Modern Contraceptive Prevalence Rate",
        "category": "Maternal Health",
        "unit": "% married women (15-49)",
        "definition": "Percentage of currently married women age 15-49 currently using any modern method of contraception.",
        "higher_is_better": True,
    },
    # Child Mortality
    "CM_ECMT_C_IMR": {
        "short_name": "infant_mortality_rate",
        "label": "Infant Mortality Rate",
        "category": "Child Mortality",
        "unit": "Deaths per 1,000 live births",
        "definition": "Probability of dying before the first birthday per 1,000 live births for the 5-year period preceding the survey.",
        "higher_is_better": False,
    },
    "CM_ECMT_C_U5M": {
        "short_name": "under_five_mortality_rate",
        "label": "Under-Five Mortality Rate",
        "category": "Child Mortality",
        "unit": "Deaths per 1,000 live births",
        "definition": "Probability of dying before the fifth birthday per 1,000 live births for the 5-year period preceding the survey.",
        "higher_is_better": False,
    },
    # Child Immunization
    "CH_VACS_C_DP3": {
        "short_name": "dpt3_vaccination_coverage",
        "label": "DPT 3 / Pentavalent 3 Vaccination",
        "category": "Child Immunization",
        "unit": "% children (12-23 mos)",
        "definition": "Percentage of children age 12-23 months who received three doses of DPT / Pentavalent vaccine.",
        "higher_is_better": True,
    },
    "CH_VACS_C_MSL": {
        "short_name": "measles_vaccination_coverage",
        "label": "Measles Vaccination Coverage",
        "category": "Child Immunization",
        "unit": "% children (12-23 mos)",
        "definition": "Percentage of children age 12-23 months who received the first dose of measles vaccine.",
        "higher_is_better": True,
    },
    "CH_VACS_C_BAS": {
        "short_name": "basic_vaccination_coverage",
        "label": "Full Basic Immunization Coverage",
        "category": "Child Immunization",
        "unit": "% children (12-23 mos)",
        "definition": "Percentage of children age 12-23 months who received all basic vaccinations (BCG, measles, and 3 doses each of DPT and polio).",
        "higher_is_better": True,
    },
    # Child Nutrition
    "CN_NUTS_C_HA2": {
        "short_name": "child_stunting_prevalence",
        "label": "Childhood Stunting (HAZ < -2 SD)",
        "category": "Child Nutrition",
        "unit": "% children under 5",
        "definition": "Percentage of children under age 5 whose height-for-age is below minus two standard deviations from the WHO Child Growth Standards median.",
        "higher_is_better": False,
    },
    "CN_NUTS_C_WH2": {
        "short_name": "child_wasting_prevalence",
        "label": "Childhood Wasting (WHZ < -2 SD)",
        "category": "Child Nutrition",
        "unit": "% children under 5",
        "definition": "Percentage of children under age 5 whose weight-for-height is below minus two standard deviations from the WHO Child Growth Standards median.",
        "higher_is_better": False,
    },
    "CN_NUTS_C_WA2": {
        "short_name": "child_underweight_prevalence",
        "label": "Childhood Underweight (WAZ < -2 SD)",
        "category": "Child Nutrition",
        "unit": "% children under 5",
        "definition": "Percentage of children under age 5 whose weight-for-age is below minus two standard deviations from the WHO Child Growth Standards median.",
        "higher_is_better": False,
    },
    "CN_IYCB_C_EXB": {
        "short_name": "exclusive_breastfeeding_rate",
        "label": "Exclusive Breastfeeding (< 6 mos)",
        "category": "Child Nutrition",
        "unit": "% infants (0-5 mos)",
        "definition": "Percentage of infants under age 6 months who are exclusively breastfed.",
        "higher_is_better": True,
    },
    # Socioeconomic & WASH
    "WS_SRCE_H_IMP": {
        "short_name": "improved_drinking_water_access",
        "label": "Access to Improved Water",
        "category": "Socioeconomic & WASH",
        "unit": "% households",
        "definition": "Percentage of households with access to an improved source of drinking water.",
        "higher_is_better": True,
    },
    "WS_TLET_H_IMP": {
        "short_name": "improved_sanitation_access",
        "label": "Access to Improved Sanitation",
        "category": "Socioeconomic & WASH",
        "unit": "% households",
        "definition": "Percentage of households with an improved, non-shared toilet or sanitation facility.",
        "higher_is_better": True,
    },
    "ED_EDAT_W_SEC": {
        "short_name": "female_secondary_education",
        "label": "Female Secondary Education",
        "category": "Socioeconomic & WASH",
        "unit": "% women (15-49)",
        "definition": "Percentage of women age 15-49 who have attained secondary education or higher.",
        "higher_is_better": True,
    },
}

# Mapping dicts
ID_TO_SHORT = {k: v["short_name"] for k, v in INDICATORS_CONFIG.items()}
SHORT_TO_LABEL = {v["short_name"]: v["label"] for k, v in INDICATORS_CONFIG.items()}
SHORT_TO_CATEGORY = {v["short_name"]: v["category"] for k, v in INDICATORS_CONFIG.items()}
FEATURE_COLUMNS = [v["short_name"] for v in INDICATORS_CONFIG.values()]
