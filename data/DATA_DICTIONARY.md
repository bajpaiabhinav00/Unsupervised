# DHS Program Maternal-Child Health Data Dictionary

This data dictionary outlines the 17 national-level indicators acquired from the **Demographic and Health Surveys (DHS) Program** via the official DHS REST API. These indicators align with priorities funded by global public health initiatives, including the Bill & Melinda Gates Foundation.

## Indicator Specifications

| Indicator ID | Feature Identifier (Code) | Category | Indicator Definition | Measurement Unit | Directionality |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RH_ANCN_W_N4P** | `anc_4plus_coverage` | Maternal Health | Percentage of women age 15–49 with a live birth in the 5 years preceding the survey who attended four or more antenatal care visits. | % of women (15–49) | Higher is better |
| **RH_ANCP_W_SKP** | `anc_skilled_provider` | Maternal Health | Percentage of women age 15–49 who received antenatal care from skilled healthcare personnel (doctor, nurse, midwife). | % of women (15–49) | Higher is better |
| **RH_DELA_C_SKP** | `skilled_birth_attendance` | Maternal Health | Percentage of live births delivered with assistance from a skilled provider (doctor, nurse, midwife, auxiliary midwife). | % of live births | Higher is better |
| **RH_DELP_C_DHF** | `institutional_delivery` | Maternal Health | Percentage of live births delivered in a health facility (public or private medical institution). | % of live births | Higher is better |
| **FP_CUSA_W_MOD** | `modern_contraceptive_prevalence` | Maternal Health | Modern Contraceptive Prevalence Rate (mCPR): Percentage of currently married women age 15–49 using modern family planning methods. | % married women (15–49) | Higher is better |
| **CM_ECMT_C_IMR** | `infant_mortality_rate` | Child Mortality | Infant Mortality Rate: Probability of dying between birth and exact age 1, expressed per 1,000 live births (5-year reference window). | Deaths per 1,000 live births | Lower is better |
| **CM_ECMT_C_U5M** | `under_five_mortality_rate` | Child Mortality | Under-Five Mortality Rate: Probability of dying between birth and exact age 5, expressed per 1,000 live births (5-year reference window). | Deaths per 1,000 live births | Lower is better |
| **CH_VACS_C_DP3** | `dpt3_vaccination_coverage` | Child Immunization | Percentage of surviving children age 12–23 months who received three doses of DPT / Pentavalent vaccine. | % children (12–23 mos) | Higher is better |
| **CH_VACS_C_MSL** | `measles_vaccination_coverage` | Child Immunization | Percentage of children age 12–23 months who received at least one dose of measles-containing vaccine (MCV1). | % children (12–23 mos) | Higher is better |
| **CH_VACS_C_BAS** | `basic_vaccination_coverage` | Child Immunization | Full basic vaccination coverage: Percentage of children 12–23 months receiving BCG, measles, and 3 doses of DPT & polio. | % children (12–23 mos) | Higher is better |
| **CN_NUTS_C_HA2** | `child_stunting_prevalence` | Child Nutrition | Childhood Stunting: Percentage of children under age 5 whose height-for-age is < -2 standard deviations from WHO Child Growth Standards. | % children under 5 | Lower is better |
| **CN_NUTS_C_WH2** | `child_wasting_prevalence` | Child Nutrition | Childhood Wasting (acute malnutrition): Percentage of children under 5 whose weight-for-height is < -2 standard deviations from WHO standards. | % children under 5 | Lower is better |
| **CN_NUTS_C_WA2** | `child_underweight_prevalence` | Child Nutrition | Childhood Underweight: Percentage of children under age 5 whose weight-for-age is < -2 standard deviations from WHO Child Growth Standards. | % children under 5 | Lower is better |
| **CN_IYCB_C_EXB** | `exclusive_breastfeeding_rate` | Child Nutrition | Exclusive Breastfeeding: Percentage of infants under 6 months of age who are fed exclusively with breast milk. | % infants (0–5 mos) | Higher is better |
| **WS_SRCE_H_IMP** | `improved_drinking_water_access` | Socioeconomic & WASH | Percentage of households with access to an improved drinking water source (piped, borehole, protected well/spring, rainwater). | % households | Higher is better |
| **WS_TLET_H_IMP** | `improved_sanitation_access` | Socioeconomic & WASH | Percentage of households using an improved, non-shared sanitation facility (flush/pour-flush, ventilated pit latrine, composting toilet). | % households | Higher is better |
| **ED_EDAT_W_SEC** | `female_secondary_education` | Socioeconomic & WASH | Percentage of women age 15–49 who have completed secondary education or attained higher educational qualifications. | % women (15–49) | Higher is better |

## Dataset Provenance & Methodological Notes
- **Source**: Demographic and Health Surveys (DHS) Program ([dhsprogram.com](https://dhsprogram.com/)).
- **Survey Design**: Nationally representative, stratified, two-stage probability cluster sampling.
- **Aggregation Level**: National level indicators aggregated from standardized survey microdata utilizing DHS standard sampling weights.
- **Reference Window**: The analysis selects the most recent standard DHS survey for each country to capture contemporary health system profiles while preserving survey cross-comparability.
