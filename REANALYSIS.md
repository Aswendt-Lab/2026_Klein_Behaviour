# Longitudinal Reanalysis

This branch replaces the proportional-recovery analyses with models of raw repeated outcomes. It does not regress change on baseline, divide baseline severity into k-means groups, or compare Euclidean distances to PRR lines.

## Run

```powershell
python code/longitudinal_reanalysis.py
```

Results are written to `output/reanalysis_reviewer1/`. To select another location:

```powershell
python code/longitudinal_reanalysis.py --output-dir path/to/output
```

Run preparation tests with:

```powershell
pytest tests/test_longitudinal_reanalysis.py
```

## Inputs

The pipeline reads:

- `input/FM_JK_V1.xlsx`
- `input/Assessments_JK_V1.2.xlsx`
- `input/Stroke_types.csv`
- `input/Stroke_types.xlsx` as a consistency check
- `input/Mice_data.csv`

Human observations are retained whenever the outcome is available. The pipeline does not require complete data for all visits or assessments. Mouse rows marked as included and not marked for exclusion are retained, and repeated animal/day measurements are averaged because no trial-level identifier is available.

## Human Models

FM-UE, FM-LE, BI, and NIHSS are analyzed with random-intercept linear mixed models:

```text
raw score ~ categorical visit + (1 | participant)
```

If the estimated random-intercept model is singular or does not converge, the pipeline uses population-averaged Gaussian GEE with exchangeable within-participant correlation and records the fallback in the coefficient output.

The baseline-severity analysis uses only follow-up outcomes:

```text
raw T1/T2 score ~ categorical visit * continuous T0 score
                 + (1 | participant)
```

This prevents T0 from appearing simultaneously as both an outcome and a baseline covariate. The pipeline first attempts ordinal GEE for mRS because `statsmodels` does not provide ordinal mixed-effects models. If that model does not converge, it attempts a prespecified favorable-outcome split of mRS `0-2` versus `3-6` with logistic GEE. If sparse categories or complete separation prevent both approaches, it uses Gaussian GEE on raw mRS and labels that approximation explicitly.

For visualization, the pipeline predicts T1 and T2 scores at the observed 25th and 75th percentiles of continuous baseline deficit. These are reference profiles, not patient groups or clusters; all interaction tests use the continuous predictor.

Stroke type is reported descriptively but is not included in the primary models because the hemorrhage subgroup is too small for stable adjustment or interaction estimates.

Visit is categorical because only nominal T0, T1, and T2 labels are supplied. Exact days after stroke are not available in the human input files.

## Mouse Models

Each raw behavioral outcome is analyzed separately:

```text
raw outcome ~ categorical day * sham/stroke group + (1 | animal)
```

Among stroke animals, acute severity is retained continuously:

```text
raw day >= 7 outcome ~ categorical day * continuous day-3 outcome
                       + stroke type + (1 | animal)
```

No mouse clustering or proportional-recovery variables are used.

## Scale Boundaries

The analysis reports exact floor and ceiling counts at each visit and performs two diagnostic sensitivity analyses:

- Excluding participants at a boundary at T0
- Excluding boundary observations from the repeated-outcome model

Two FM-LE values (`95` and `96`) exceed the documented maximum of `86`. They are retained in the exported long data, reported in `human_range_violations.csv`, and excluded from models pending source verification.

The standard mRS range of `0-6` is used. The previous analysis used a maximum of `5`, although the input contains one score of `6`.

## Interpretation

Inference remains on raw outcomes. Standardized responsiveness and nominal three-visit slopes are descriptive cross-scale summaries; nominal slopes represent equally spaced visit indices because elapsed human assessment days are unavailable.

The cross-species display uses three conceptual landmarks: human T0/T1/T2 and mouse days 3/14/56. Recovery is oriented consistently and scaled to acute-stage variability within each outcome. Human and mouse subjects are unrelated, the observations are not matched pairs, and neither the selected stages nor the instruments should be interpreted as biologically or metrically equivalent.

The generated `analysis_summary.md` records model results, failures, and limitations. Statistical and clinical review is still required before replacing manuscript results.

## Figures

The pipeline exports manuscript-style SVG and PNG versions of:

- Proposed Figure 2, `human_raw_score_trajectories`: available subject trajectories, visit distributions, and mean 95% confidence intervals
- Proposed Figure 3, `human_floor_ceiling`: exact floor and ceiling percentages by assessment and visit
- Proposed Figure 4, `human_baseline_severity_trajectories`: model predictions at the 25th and 75th percentiles of continuous baseline deficit
- Proposed Figure 5, `cross_species_standardized_trajectories`: conceptual human-mouse recovery comparisons at three standardized stages
- Companion mouse figure, `mouse_raw_score_trajectories`: sham and stroke trajectories at days 0, 3, 7, 14, 21, 28, 42, and 56

Figures use the established manuscript settings: Calibri, 12 pt labels, 8 pt ticks, 300 dpi, 18 cm width, frameless legends, hidden top/right spines, and the existing Okabe-Ito-derived palette.

Create the captioned Word document with native SVG figures using:

```powershell
python code/create_reanalysis_figures_docx.py
```

The document is written to `output/reanalysis_reviewer1/longitudinal_reanalysis_figures_supervisor_revision.docx`. It includes the supervisor comments in red, analysis responses, captions, mean/SD tables, and primary model-test tables.

Complete tabular output is also consolidated in `output/reanalysis_reviewer1/reanalysis_statistical_tables.xlsx`. Its sheets include human and mouse descriptives, model coefficients and tests, boundary sensitivities, standardized responsiveness, nominal slopes and correlations, continuous baseline predictions, and cross-species stage summaries.
