# Markus ToDo Status

Source documents: `output/reanalysis_reviewer1/markus/ToDos.docx` and `report_updated_trajectory_framework.docx`.

## Implemented in the first revision pass

1. Cohort audit and flow tables: implemented from the supplied files. The auditable provisional human cohort is 120 T0 records with stroke metadata; record 121 is a T1-only orphan retained for audit but excluded from analyses. The historical N=91 remains unresolved because no eligibility ledger is supplied. Mouse flow is 162 source animals, 141 eligible animals, 111 stroke, and 30 sham.
2. Final uncoupled models: implemented. Raw-score models and continuous Time x Baseline models are retained.
3. Figure 4/Table 3 consistency: implemented. Figure 4 predictions now come from the exact fitted model used for each outcome's test.
4. Boundary moderation sensitivity: implemented for exclusion of baseline-boundary participants and exclusion of individual boundary observations, with separate FDR families.
5. Scale audit: extended FM-UE 0-126 and FM-LE 0-86 definitions are documented from the manuscript methods. FM-LE values 95 and 96 remain unresolved source queries and are not recoded.
6. mRS sensitivity: corrected to attempt a no-intercept ordinal GEE and reject numerically unstable fits. The current baseline-moderation fallback is binary favorable mRS 0-2 versus 3-6; the raw trajectory requires the explicitly labeled Gaussian GEE fallback.
7. Multiple testing: FDR remains separated by scientific test family. Family and model labels are now exported and displayed in the report.

## Next analysis block

1. Human trajectory phenotypes require a clinically reviewed threshold registry. The repository contains only an uncited six-point FM statement and no defensible scale-specific MCID/MDC values for the extended FM composites, BI, NIHSS, or mRS. Direction-only classifications can be generated as an exploratory sensitivity but should not be presented as final phenotypes.
2. Cross-scale temporal discrimination, non-steady proportions, agreement/kappa, heatmap, and continuous-baseline logistic models depend on the phenotype definitions above.
3. Clinical characterization is blocked beyond stroke type. No participant-linked age, sex, treatment, vascular-risk, etiology, hemorrhagic-transformation, or lesion-complexity variables are supplied.
4. Hemorrhage can currently be shown descriptively by trajectory after phenotypes are finalized; formal subgroup inference should not be used.
5. Mouse D3-D14 phenotype thresholds can be estimated from paired sham change. D14-D56 sham data are insufficient: there are no complete late sham pairs for paw drag and only one each for foot faults and hindlimb drops. A pooled sham residual threshold would be exploratory, not a test-retest MDC.
6. The three-feature human-mouse comparison will combine overall recovery, non-steady phenotype frequency, and continuous acute-severity moderation after threshold decisions are finalized.
7. Removal of legacy PRR text from the manuscript remains pending; the new analysis outputs themselves do not use PRR, k-means severity groups, or baseline-versus-change inference.

## Required external inputs

1. Authoritative 91-participant eligibility/flow ledger keyed by `record_id`.
2. Source verification for FM-LE values 95 and 96 and confirmation that exported FM totals consistently include H/I/J domains.
3. Clinically approved MCID/MDC thresholds and citations for each exact assessment form.
4. Participant-linked clinical metadata for atypical-case characterization.
5. Confirmation whether an exploratory pooled-sham late mouse threshold is acceptable despite sparse D56 sham observations.
