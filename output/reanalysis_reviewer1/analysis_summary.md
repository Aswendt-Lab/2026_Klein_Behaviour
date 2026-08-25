# Longitudinal Reanalysis Summary

This analysis models raw repeated outcomes and does not use proportional-recovery regressions, change scores as outcomes, k-means severity groups, or Euclidean PRR distances.

## Data Use

- Human: 121 records with 1296 valid score observations.
- Mouse: 141 eligible animals with 2103 aggregated valid outcome observations.
- Human out-of-range observations: 2.
- Mouse out-of-range observations: 0.

## Models

- FM-UE, FM-LE, BI, and NIHSS use random-intercept linear mixed models on raw scores, with Gaussian GEE fallback when the random-effects fit is singular.
- mRS attempts ordinal GEE, then prespecified mRS 0-2 versus 3-6 logistic GEE, and finally Gaussian GEE on raw mRS if sparse categories prevent those models.
- Human visit is categorical; no equal spacing between T0, T1, and T2 is assumed.
- Baseline moderation models use only T1/T2 outcomes and continuous T0 severity, avoiding use of T0 as both outcome and covariate.
- Mouse models use all available days and continuous raw outcomes after averaging replicate animal/day observations.
- Benjamini-Hochberg correction is applied separately to prespecified human and mouse test families.

## Important Limitations

- Two FM-LE source values exceed the documented maximum of 86 and are excluded from modeling pending source verification.
- Nominal human visits are used because exact days after stroke are not present in the supplied workbooks.
- Gaussian mixed models may not fully represent bounded or zero-heavy outcomes; diagnostics and alternative outcome-specific models remain necessary before publication.
- The GEE analyses for mRS are population-averaged rather than subject-specific; the binary fallback loses ordinal information.
- Mouse replicate rows are averaged because trial-level identifiers are not supplied.
- Stroke type is summarized descriptively but is not included in primary models because the hemorrhage subgroup is small.

## Multiplicity-Adjusted Tests

| dataset   | assessment       | analysis                      | contrast                              |   n_subjects |   p_value |   p_value_fdr_bh |
|:----------|:-----------------|:------------------------------|:--------------------------------------|-------------:|----------:|-----------------:|
| human     | FM-UE            | raw_trajectory                | overall visit effect                  |          121 | 9.495e-21 |        1.187e-20 |
| human     | FM-UE            | baseline_moderation           | visit by continuous baseline severity |           94 | 0.5741    |        0.5741    |
| human     | FM-UE            | exclude_baseline_boundaries   | overall visit effect                  |          121 | 9.495e-21 |        9.495e-21 |
| human     | FM-UE            | exclude_boundary_observations | overall visit effect                  |          121 | 3.37e-16  |        4.212e-16 |
| human     | FM-LE            | raw_trajectory                | overall visit effect                  |          120 | 4.13e-20  |        4.13e-20  |
| human     | FM-LE            | baseline_moderation           | visit by continuous baseline severity |           93 | 0.04892   |        0.1223    |
| human     | FM-LE            | exclude_baseline_boundaries   | overall visit effect                  |          118 | 2.037e-21 |        2.547e-21 |
| human     | FM-LE            | exclude_boundary_observations | overall visit effect                  |          120 | 1.183e-15 |        1.183e-15 |
| human     | BI               | raw_trajectory                | overall visit effect                  |          120 | 2.33e-50  |        1.165e-49 |
| human     | BI               | baseline_moderation           | visit by continuous baseline severity |           95 | 0.08999   |        0.15      |
| human     | BI               | exclude_baseline_boundaries   | overall visit effect                  |          112 | 1.017e-49 |        5.086e-49 |
| human     | BI               | exclude_boundary_observations | overall visit effect                  |          117 | 4.59e-41  |        2.295e-40 |
| human     | NIHSS            | raw_trajectory                | overall visit effect                  |          121 | 6.563e-30 |        1.094e-29 |
| human     | NIHSS            | baseline_moderation           | visit by continuous baseline severity |           94 | 0.0008296 |        0.004148  |
| human     | NIHSS            | exclude_baseline_boundaries   | overall visit effect                  |          120 | 1.983e-30 |        3.305e-30 |
| human     | NIHSS            | exclude_boundary_observations | overall visit effect                  |          120 | 5.196e-23 |        8.659e-23 |
| human     | MRS              | raw_trajectory                | overall visit effect                  |          121 | 2.649e-39 |        6.622e-39 |
| human     | MRS              | baseline_moderation           | visit by continuous baseline severity |           98 | 0.4322    |        0.5402    |
| human     | MRS              | exclude_baseline_boundaries   | overall visit effect                  |          120 | 4.209e-39 |        1.052e-38 |
| human     | MRS              | exclude_boundary_observations | overall visit effect                  |          120 | 2.317e-39 |        5.794e-39 |
| mouse     | C_PawDragPercent | group_trajectory              | day by sham/stroke group              |          138 | 5.273e-23 |        7.909e-23 |
| mouse     | C_PawDragPercent | acute_severity_moderation     | day by continuous day-3 severity      |           87 | 0.09169   |        0.09169   |
| mouse     | GW_FootFault     | group_trajectory              | day by sham/stroke group              |          141 | 2.974e-37 |        8.921e-37 |
| mouse     | GW_FootFault     | acute_severity_moderation     | day by continuous day-3 severity      |          103 | 2.504e-11 |        7.513e-11 |
| mouse     | RB_HindlimbDrop  | group_trajectory              | day by sham/stroke group              |          139 | 1.743e-15 |        1.743e-15 |
| mouse     | RB_HindlimbDrop  | acute_severity_moderation     | day by continuous day-3 severity      |           92 | 2.108e-09 |        3.162e-09 |

## Model Failures

No model failures were recorded.

These results are exploratory and require statistical review, residual diagnostics, verification of scale coding, and reconciliation with the manuscript before inferential claims are made.