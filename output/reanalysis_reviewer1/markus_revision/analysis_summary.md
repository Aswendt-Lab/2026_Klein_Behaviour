# Longitudinal Reanalysis Summary

This analysis models raw repeated outcomes and does not use proportional-recovery regressions, change scores as outcomes, k-means severity groups, or Euclidean PRR distances.

## Data Use

- Human: provisional cohort of 120 records with 1292 valid score observations.
- Mouse: 141 eligible animals with 2103 aggregated valid outcome observations.
- Human out-of-range observations: 1.
- Mouse out-of-range observations: 0.

## Models

- FM-UE, FM-LE, BI, and NIHSS use random-intercept linear mixed models on raw scores, with Gaussian GEE fallback when the random-effects fit is singular.
- mRS attempts ordinal GEE, then prespecified mRS 0-2 versus 3-6 logistic GEE, and finally Gaussian GEE on raw mRS if sparse categories prevent those models.
- Human visit is categorical; no equal spacing between T0, T1, and T2 is assumed.
- Baseline moderation models use only T1/T2 outcomes and continuous T0 severity, avoiding use of T0 as both outcome and covariate.
- Mouse models use all available days and continuous raw outcomes after averaging replicate animal/day observations.
- Benjamini-Hochberg correction is applied separately to prespecified human and mouse test families.

## Important Limitations

- Two FM-LE source values exceed the documented maximum of 86: one is in the provisional cohort and one belongs to the excluded T1-only record. Neither is modeled pending source verification.
- Nominal human visits are used because exact days after stroke are not present in the supplied workbooks.
- Gaussian mixed models may not fully represent bounded or zero-heavy outcomes; diagnostics and alternative outcome-specific models remain necessary before publication.
- The GEE analyses for mRS are population-averaged rather than subject-specific; the binary fallback loses ordinal information.
- Mouse replicate rows are averaged because trial-level identifiers are not supplied.
- Stroke type is summarized descriptively but is not included in primary models because the hemorrhage subgroup is small.

## Multiplicity-Adjusted Tests

| dataset   | assessment       | analysis                                          | contrast                              |   n_subjects |   p_value |   p_value_fdr_bh |
|:----------|:-----------------|:--------------------------------------------------|:--------------------------------------|-------------:|----------:|-----------------:|
| human     | FM-UE            | raw_trajectory                                    | overall visit effect                  |          120 | 1.115e-20 |        1.394e-20 |
| human     | FM-UE            | baseline_moderation                               | visit by continuous baseline severity |           94 | 0.5741    |        0.5741    |
| human     | FM-UE            | baseline_moderation_exclude_baseline_boundaries   | visit by continuous baseline severity |           94 | 0.5741    |        0.5741    |
| human     | FM-UE            | baseline_moderation_exclude_boundary_observations | visit by continuous baseline severity |           92 | 0.3848    |        0.4249    |
| human     | FM-UE            | exclude_baseline_boundaries                       | overall visit effect                  |          120 | 1.115e-20 |        1.115e-20 |
| human     | FM-UE            | exclude_boundary_observations                     | overall visit effect                  |          120 | 3.99e-16  |        4.987e-16 |
| human     | FM-LE            | raw_trajectory                                    | overall visit effect                  |          120 | 4.13e-20  |        4.13e-20  |
| human     | FM-LE            | baseline_moderation                               | visit by continuous baseline severity |           93 | 0.04892   |        0.1223    |
| human     | FM-LE            | baseline_moderation_exclude_baseline_boundaries   | visit by continuous baseline severity |           91 | 0.06346   |        0.1058    |
| human     | FM-LE            | baseline_moderation_exclude_boundary_observations | visit by continuous baseline severity |           86 | 0.2255    |        0.3758    |
| human     | FM-LE            | exclude_baseline_boundaries                       | overall visit effect                  |          118 | 2.037e-21 |        2.547e-21 |
| human     | FM-LE            | exclude_boundary_observations                     | overall visit effect                  |          120 | 1.183e-15 |        1.183e-15 |
| human     | BI               | raw_trajectory                                    | overall visit effect                  |          119 | 5.121e-50 |        2.561e-49 |
| human     | BI               | baseline_moderation                               | visit by continuous baseline severity |           95 | 0.08999   |        0.15      |
| human     | BI               | baseline_moderation_exclude_baseline_boundaries   | visit by continuous baseline severity |           88 | 0.0554    |        0.1058    |
| human     | BI               | baseline_moderation_exclude_boundary_observations | visit by continuous baseline severity |           82 | 0.003231  |        0.008078  |
| human     | BI               | exclude_baseline_boundaries                       | overall visit effect                  |          111 | 2.317e-49 |        1.159e-48 |
| human     | BI               | exclude_boundary_observations                     | overall visit effect                  |          116 | 1.259e-40 |        6.293e-40 |
| human     | NIHSS            | raw_trajectory                                    | overall visit effect                  |          120 | 8.699e-30 |        1.45e-29  |
| human     | NIHSS            | baseline_moderation                               | visit by continuous baseline severity |           94 | 0.0008296 |        0.004148  |
| human     | NIHSS            | baseline_moderation_exclude_baseline_boundaries   | visit by continuous baseline severity |           93 | 0.001142  |        0.005711  |
| human     | NIHSS            | baseline_moderation_exclude_boundary_observations | visit by continuous baseline severity |           93 | 0.003187  |        0.008078  |
| human     | NIHSS            | exclude_baseline_boundaries                       | overall visit effect                  |          119 | 2.659e-30 |        4.432e-30 |
| human     | NIHSS            | exclude_boundary_observations                     | overall visit effect                  |          119 | 7.043e-23 |        1.174e-22 |
| human     | MRS              | raw_trajectory                                    | overall visit effect                  |          120 | 4.701e-39 |        1.175e-38 |
| human     | MRS              | baseline_moderation                               | visit by continuous baseline severity |           98 | 0.4313    |        0.5391    |
| human     | MRS              | baseline_moderation_exclude_baseline_boundaries   | visit by continuous baseline severity |           98 | 0.4313    |        0.5391    |
| human     | MRS              | baseline_moderation_exclude_boundary_observations | visit by continuous baseline severity |           98 | 0.4249    |        0.4249    |
| human     | MRS              | exclude_baseline_boundaries                       | overall visit effect                  |          119 | 7.442e-39 |        1.86e-38  |
| human     | MRS              | exclude_boundary_observations                     | overall visit effect                  |          119 | 4.172e-39 |        1.043e-38 |
| mouse     | C_PawDragPercent | group_trajectory                                  | day by sham/stroke group              |          138 | 5.273e-23 |        7.909e-23 |
| mouse     | C_PawDragPercent | acute_severity_moderation                         | day by continuous day-3 severity      |           87 | 0.09169   |        0.09169   |
| mouse     | GW_FootFault     | group_trajectory                                  | day by sham/stroke group              |          141 | 2.974e-37 |        8.921e-37 |
| mouse     | GW_FootFault     | acute_severity_moderation                         | day by continuous day-3 severity      |          103 | 2.504e-11 |        7.513e-11 |
| mouse     | RB_HindlimbDrop  | group_trajectory                                  | day by sham/stroke group              |          139 | 1.743e-15 |        1.743e-15 |
| mouse     | RB_HindlimbDrop  | acute_severity_moderation                         | day by continuous day-3 severity      |           92 | 2.108e-09 |        3.162e-09 |

## Model Failures

No model failures were recorded.

These results are exploratory and require statistical review, residual diagnostics, verification of scale coding, and reconciliation with the manuscript before inferential claims are made.