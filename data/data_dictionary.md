# Sharing dataframe dictionary

`data/analysis_bins.csv` is the combined empirical dataset and the only empirical CSV input required for fitting. It contains all included sessions and both bottles; the CSV files in `results/` are saved fitted parameters or derived outputs.

The CSV is the single interval-level analysis dataframe. One row represents one mouse in one source session during one nominal 3-hour interval. The two bottles are columns, so summing total_g does not double-count bottles. MouseID is a session identifier; inferred_animal_id links likely biological animals and may span sessions/conditions. Neither denotes independently verified colony-wide identity in every record.

## Core fields

| Fields | Meaning / units |
|---|---|
| MouseID, Mouse, File | Unique session key, local cage label, original workbook name. Never merge across sessions using Mouse alone. |
| inferred_animal_id, identity_basis | Cross-session animal grouping and the evidence used. Genotype conflicts remain separate. |
| experiment_date, inferred_birth_date, recorded_birth_date | Filename date, date inferred from effective analysis age, and parsed raw registry DOB. Disagreements are retained. |
| Sex, Age | Effective analysis sex and age (days), taken from the raw animal register where available at the author's instruction. Notebook fallback is explicitly labelled. |
| Sex_notebook, Age_notebook, sex_source, age_source, demographic_resolution | Preserved original notebook values and the source/resolution applied to each effective field. |
| Genotype, Condition, Condition_real | Preserved notebook/combined metadata. Condition_real groups control/PA/RF conditions. |
| age_group | Young ≤180 d, early middle 181–300 d, middle 301–450 d, old >450 d; <60 d is Juvenile. |
| Time_start, Time_end | Nominal clock hours, unwrapped beyond 24 in chronological session order. |
| ZT | Interval-center hour relative to recorded reference zero, wrapped to [0,24). DD uses prior reference time, not measured free-running circadian phase. |
| bin_index, elapsed_h, duration_h | Zero-based chronological interval, midpoint hours since session start, and nominal interval duration. |
| total_g, reward_g, water_g | Total bottle loss, sweet-fluid loss and complementary water loss in grams. For water/water controls, reward_g means Bottle A loss. |
| preference | reward_g / total_g; blank when total_g is zero. No missing measurement is converted to zero. |
| total_g_per_h, phase, fluid | Nominal hourly rate, Day/Night midpoint reference half, and fluid identity. |
| bottle_A_Consumption, bottle_B_Consumption | Per-bottle clipped mass difference in grams, exactly reconstructing total_g. |
| bottle_A_Type, bottle_B_Type | Fluid in each bottle for that interval, including swaps. |
| bottle_*_raw_weight_start_g / raw_weight_end_g / raw_weight_difference_g | Raw masses and signed consecutive difference, before historical negative-loss clipping. |
| bottle_*_negative_difference_clipped | Whether a negative raw difference was clipped to zero to reproduce preprocessing. |
| bottle_*_source_weight_start_cell / source_weight_end_cell | Original spreadsheet cell references for the two mass measurements. |
| source_workbook, source_notebook, source_sheet, metadata_cell | Reproducible provenance within the project. source_sheet identifies the OOXML worksheet part (e.g. sheet1 means xl/worksheets/sheet1.xml), not necessarily the visible tab name. Original files are unchanged. |
| source_animal_tag, source_birth_date, source_age_days, source_sex | Raw registry fields or explicitly derived registry sex from tag. The raw sex/age are the documented preferred analysis source. |
| source_identity_link_verified / source_identity_link_basis | Whether linking the raw registry to the measured cage was corroborated, and how. Generic “Male/Female” entries are not unique animal tags. |
| source_metadata_conflict_fields / reason | Explicit field-level disagreements. |
| analysis_demographics_verified / conflicted | Historical corroboration versus disagreement with notebook metadata. Conflicted remains true for traceability after the raw-source decision; it is not a current exclusion flag. |
| recorded_reference_zero_h, authoritative_reference_zero_h, schedule_decision | Recorded timing choice and audit decision; “authoritative” denotes the retained analysis source, not a newly verified light-event log. |
| measurement_header_max_error_h, filename_condition_conflict | Maximum raw-header deviation from nominal time grid and disagreement between filename and coded condition. |
| source_status, original_MouseID | Retained versus recovered source status and original session key. |
| modeling_cohort | One earliest eligible sucrose/saccharin session per inferred animal; no AD/RF. |
| entrainment_cohort | Earliest eligible session per animal WITHIN Control/PA/RF-morning/RF-afternoon. Shared animals require paired/cluster-aware inference. |
| legacy_age_cohort | Current control/age figure cohort, retaining eligible sessions with effective raw-register demographics. The name preserves code compatibility, not old counts. |
| palatable_cohort, source_palatable | Sucrose/saccharin session flags. |
| excluded_AD, excluded_RF | Animal-level condition flags. AD rows have already been removed from this sharing file, so excluded_AD is false throughout; RF remains available with modelling flag false. |
| model_exclusion_reason, earliest_eligible_session, demographic_exclusion_reason | Transparent reasons/selected counterpart for downstream filtering. |

## Correct use

Filter modeling_cohort for the primary model inputs. Use entrainment_cohort for condition-specific plots and retain inferred_animal_id for repeated-condition tests. Age figures use legacy_age_cohort. Select one fluid/metric as appropriate; water controls define preference by bottle A, not by sucrose. Do not count interval rows or repeated sessions as independent animals.

Missing phases/ratios stay missing. Keep nominal time and source-header deviations distinguishable. Do not shift a session merely because it drinks more during the recorded day. The audit ledger lists excluded/incomplete source candidates separately; it is not a second competing analysis dataframe.
