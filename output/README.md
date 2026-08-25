# NNR-26-0437 Reviewer Comments

## Manuscript

**Title:** Recovery trajectories after stroke depend on the functional assessment: a clinical cohort study with experimental stroke comparison

**Journal:** Neurorehabilitation & Neural Repair  
**Manuscript ID:** NNR-26-0437  
**Decision date:** 22 August 2026  
**Decision:** Rejected; a fully reworked study may be submitted as a new manuscript and would be evaluated de novo.

## Editor Summary

The clinical and experimental datasets address an important question, but the reviewers identified a fundamental problem with the manuscript's central analytical framework. Regressing change scores on baseline impairment introduces mathematical coupling. This affects the proportional recovery analyses, subgroup stratification, model comparisons, and cross-species conclusions.

Addressing these concerns requires substantial reanalysis and reframing. Additional concerns include small subgroup sizes, scale-boundary effects, and inconsistent participant numbers. The requested changes exceeded the scope of a major revision.

## Revision Checklist

### Central analysis

- [ ] Remove analyses that regress change scores (`Y - X`) on baseline scores (`X`).
- [ ] Reframe the manuscript away from the Proportional Recovery Rule (PRR).
- [ ] Model raw longitudinal outcome scores across T0, T1, and T2.
- [ ] Use linear mixed-effects models (LMMs) or suitable generalized mixed-effects models (GLMMs).
- [ ] Treat baseline severity as a continuous predictor rather than splitting participants into two groups.
- [ ] Test whether baseline severity modifies trajectories using a `Time x Baseline` interaction.
- [ ] Consider nonlinear time effects, polynomial terms, piecewise splines, or natural cubic splines.
- [ ] Use actual days after stroke if these data are available.

### Scale properties

- [ ] Quantify floor and ceiling effects for every clinical scale at T0, T1, and T2.
- [ ] Perform sensitivity analyses excluding participants at scale floors or ceilings.
- [ ] Account for differences in what each scale measures, including impairment, activity, disability, and neurological deficit.
- [ ] Compare scales using standardized raw outcomes and longitudinal responsiveness rather than normalized distances in change-score space.

### Clustering and subgroups

- [ ] Remove or justify the use of k-means clustering with `k = 2`.
- [ ] Do not describe baseline-severity groups as good or poor recovery groups.
- [ ] If clustering is retained, report validation metrics such as silhouette scores or the gap statistic.
- [ ] Avoid dichotomizing continuous baseline severity because of information and power loss.
- [ ] Remove unsupported subgroup conclusions based on very small participant counts.
- [ ] Report counts alongside percentages and avoid excessive decimal precision.

### Model comparison

- [ ] Remove Euclidean-distance comparisons between PRR and best-fit models.
- [ ] Remove comparisons performed in the mathematically coupled `(X, Y - X)` coordinate space.
- [ ] Do not constrain the best-fit model using an intercept optimized for the PRR model.
- [ ] Remove ad hoc normalization of Euclidean distances by each scale's maximum score.

### Human cohort reporting

- [ ] Reconcile the reported sample sizes of 37, 36, 31, and all analysis-specific subsets.
- [ ] Explain every exclusion and reduced sample size at the relevant analysis.
- [ ] Clarify the exclusion criterion concerning other diseases that determine prognosis.
- [ ] Reconsider analyses and discussion of hemorrhagic stroke because only five participants had bleeding.
- [ ] Clarify what is meant by damage beyond one vascular territory and diffuse or non-unilateral injury patterns.
- [ ] Review whether the inclusion criteria adequately address multiple strokes or complex lesion patterns.

### Mouse analysis

- [ ] Reframe the mouse section around uncoupled longitudinal trajectories rather than PRR.
- [ ] Provide sufficient methodological detail about mouse motor-performance measurements.
- [ ] Explain and strengthen the conceptual link between the mouse and human analyses.
- [ ] Apply longitudinal mixed-effects models to continuous raw mouse outcomes.
- [ ] Limit cross-species conclusions to comparisons supported by compatible analyses and measurements.

### Manuscript framing

- [ ] Focus the central question on how scale selection, scale boundaries, and baseline severity influence detectable recovery heterogeneity.
- [ ] Distinguish clearly between scales measuring different constructs and domains.
- [ ] Temper conclusions where observed differences are small or statistically nonsignificant.
- [ ] Reassess qualitative interpretations of PRR non-fitters and other very small groups.

## Reviewer 1

### Overall Assessment

The manuscript addresses an important topic in neurorehabilitation by examining motor recovery across multiple clinical assessment scales (FM-UE, FM-LE, BI, NIHSS, and mRS) and translating these findings to rodent behavioral paradigms. The dataset has high potential value because it includes three longitudinal time points in humans and parallel rodent cohorts.

However, framing the core findings around the PRR introduces severe statistical pitfalls, including mathematical coupling, scale-ceiling artifacts, arbitrary stratification, and invalid model comparisons in a non-orthogonal coordinate space.

The reviewer strongly encourages reframing the manuscript away from PRR. Instead, the study should evaluate how scale selection, scale boundaries, and baseline severity influence detectable heterogeneity in recovery trajectories across human and rodent models.

### Major Comment 1: Mathematical Coupling in Change-Score Models

The primary analyses regress change scores (`Y - X`, where `X` is baseline and `Y` is the post-acute outcome) on baseline severity (`X`). Regressing a change score on one of its own components creates inherent mathematical coupling. This artifact inflates correlation coefficients and can force regression slopes toward approximately 0.7, even in simulated data consisting entirely of uniform random noise.

The reviewer recommends abandoning change scores as dependent variables. Raw post-stroke outcome scores should instead be modeled directly, with baseline score included as an independent covariate, for example:

```text
Y = beta_1 X + beta_0
```

A longitudinal mixed-effects model would make better use of the available data.

### Major Comment 2: Use Longitudinal Models

The study collected data at three distinct time points:

| Time point | Description |
| --- | --- |
| T0 | Acute baseline within 72 hours |
| T1 | End of early rehabilitation at 14-21 days |
| T2 | Six-month follow-up |

Reducing these measurements to one static change score (`T2 - T0`) discards temporal resolution, masks nonlinear recovery dynamics such as rapid acute recovery or a late plateau, and reduces statistical power by excluding subjects with a missing visit.

The reviewer recommends LMMs or GLMMs using raw scores across all three time points. One proposed model is:

```text
Score_ij = beta_0
         + beta_1 Time_ij
         + beta_2 Baseline_i
         + beta_3 (Time_ij x Baseline_i)
         + u_0i
         + u_1i Time_ij
         + epsilon_ij
```

The `Time x Baseline` interaction directly tests whether baseline severity alters recovery trajectories without relying on mathematically coupled change scores.

Nonlinear trajectories can be modeled with quadratic terms or piecewise linear splines. If exact days after stroke are available, time can be modeled continuously rather than categorically.

### Major Comment 3: Ceiling Effects and Scale Compression

Clinical outcomes such as FM, BI, NIHSS, and mRS have structural floors and ceilings. When follow-up scores cluster near a scale boundary, variance collapses. This compression can strengthen mathematical coupling and artificially inflate the variance attributed to baseline severity.

The reviewer notes that controlling for ceiling artifacts can reduce the apparent variance explained by PRR from 80-90% to 20-30%.

Moving away from change-score models will help, but raw-score models can still be affected when outcomes plateau at a hard scale boundary. The manuscript should report the exact proportion of participants at the floor and ceiling of every clinical scale at T0, T1, and T2. Sensitivity analyses should exclude those participants.

### Major Comment 4: K-Means and Dichotomous Stratification

The manuscript applies k-means clustering with `k = 2` to baseline scores and labels participants as good or poor recovery groups. The reviewer identifies three problems.

**Conceptual misnomer:** T0 measures acute deficit severity, not recovery. At T0, these groups can only be described as high or low severity, not good or poor recovery.

**Unvalidated assumption:** Selecting `k = 2` resembles the PRR fitter/non-fitter dichotomy, but no validity metric demonstrates that the baseline distribution is bimodal rather than continuous. Metrics such as silhouette scores or the gap statistic would be needed to support clustering.

**Loss of power:** Converting a continuous variable into two groups discards variance, reduces statistical power, and can hide nonlinear relationships.

The reviewer recommends preserving baseline severity as a continuous predictor. GLMs or LMMs can test nonlinear effects using polynomial terms or natural cubic splines and their interactions with time.

### Major Comment 5: PRR and Best-Fit Distance Comparisons

The Euclidean-distance comparison between PRR and best-fit models in `(X, Y - X)` space should be removed.

**Constrained reference model:** The best-fit model uses the intercept optimized during PRR fitting and varies only the slope. This is a circular comparison rather than an independent benchmark.

**Coupled coordinate space:** Euclidean distance in `(X, Y - X)` space treats the axes as orthogonal even though `X` appears in both axes. Standard parametric tests such as t-tests are therefore not valid for these distance metrics.

**Ad hoc normalization:** Dividing two-dimensional Euclidean distances by a scale's maximum score is a nonstandard metric that distorts error variance across scales.

The reviewer recommends standardizing outcome measures, such as with z-scores, and using an LMM to compare responsiveness, recovery rates, individual slopes, and cross-scale correlations over time. Similar uncoupled longitudinal models should be used for continuous raw rodent outcomes. The translational analysis should compare cross-species trajectories rather than PRR fits.

### Suggested Background Reading

1. Hope TM, Friston K, Price CJ, Leff AP, Rotshtein P, Bowman H. Recovery after stroke: not so proportional after all? *Brain*. https://doi.org/10.1093/brain/awy302
2. Bonkhoff AK, Hope T, Bzdok D, Guggisberg AG, Hawe RL, Dukelow SP, Rehme AK, Fink GR, Grefkes C, Bowman H. Bringing proportional recovery into proportion: Bayesian modelling of post-stroke motor impairment. *Brain*. 2020;143(7):2189-2206.
3. Lohse KR, Hawe RL, Dukelow SP, Scott SH. Statistical considerations for drawing conclusions about recovery. *Neurorehabilitation and Neural Repair*. 2021;35(1):10-22.

## Reviewer 2

### General Assessment

The study examines several clinical scores used to quantify impairment and disability and to track recovery over six months. It also includes recovery analyses in mice after stroke. The FM showed the widest range of recovery patterns and identified slightly more individuals without continuous improvement. However, differences between scales were generally small and statistically nonsignificant.

The reviewer emphasizes that the scales do not measure the same constructs. Some assess impairment, others assess disability, and the scales cover different functions and domains. Several analyses and interpretations are therefore challenging.

### Major Comment 1: Baseline Versus Change Regression

The major problem is the regression of initial impairment against change in impairment in both human and mouse data. This approach generated the concept that most stroke participants show 70% proportional recovery because fitted slopes often approach 0.7.

Prior work has shown that regressions between baseline and change are mathematically problematic. Even random numbers can generate slopes near 0.7. These analyses are invalid and should be removed from the manuscript.

### Major Comment 2: Small Subgroups

The subgroup analyses contain too few individuals to support meaningful interpretation or conclusions. On page 14, percentages for the top and bottom halves of groups often differ by only one participant.

For example, 54% versus 45% among 11 non-steady cases represents six versus five participants. A similar issue applies to FM-UE, and the BI analysis begins with only four participants.

Percentages based on such small counts should not be reported to two decimal places. Small sample sizes also undermine the qualitative interpretation of PRR non-fitters on page 18.

### Major Comment 3: Mouse Analysis

The mouse analysis appears added on and does not fit clearly with the rest of the paper. The manuscript provides insufficient detail about how mouse motor performance was quantified.

The central statistical concern also applies to the mouse data: initial impairment should not be compared with change in impairment because the variables are mathematically coupled.

### Minor Comment 1: Exclusion Criteria

Page 6, line 58 refers to "other diseases determining prognosis." Clarify whether participants could have neurological impairments as long as those impairments were not expected to affect the clinical scores.

### Minor Comment 2: Participant Numbers

The participant count is difficult to follow. The flowchart reports 37 participants at T2, while the results and analyses sum to 36. The methods on page 11, line 15 refer to four out of 31 patients.

The number of participants used in every analysis should be clear. Every reduction in sample size should be explained at the relevant point.

### Minor Comment 3: Purpose of the Cluster Test

Page 8, line 46 describes dividing participants into two groups using clustering and then applying a t-test to determine whether regression errors differ between groups. The purpose and interpretation of this test are unclear.

### Minor Comment 4: Hemorrhagic Stroke Subgroup

Figure 1B includes only five participants with stroke caused by bleeding. The reviewer recommends reducing the discussion of this subgroup and questions whether the diagram should be included given the very limited sample.

### Minor Comment 5: Lesion Description and Inclusion Criteria

Page 18, line 20 states that a common feature among outliers was brain damage extending beyond a single vascular territory, including diffuse or non-unilateral injury patterns.

The reviewer asks what this means and whether these participants had two separate strokes. The inclusion criteria may need to be clarified or strengthened.

### References Cited by Reviewer 2

1. Bonkhoff AK, Hope T, Bzdok D, Guggisberg AG, Hawe RL, Dukelow SP. Bringing proportional recovery into proportion: Bayesian modelling of post-stroke motor impairment. *Brain*. 2020;143:2189-2206.
2. Hawe RL, Scott SH, Dukelow SP. Response to letter to the Editor: Taking proportional out of stroke recovery. *Stroke*. 2019;50(5):e126.
3. Lohse K, Hawe R, Dukelow SP, Scott SH. Statistical limitations on drawing inferences about proportional recovery. *Neurorehabilitation and Neural Repair*. 2021;35:10-22.

## Proposed Reframing

The reviewers converge on a viable new direction for the study:

> How do assessment-scale selection, scale boundaries, and baseline severity influence the observed heterogeneity and timing of recovery trajectories after stroke in humans and experimental models?

This reframing would use raw longitudinal outcomes, retain baseline severity as a continuous variable, explicitly account for scale boundaries, and compare human and mouse trajectories without relying on proportional-recovery regressions.
