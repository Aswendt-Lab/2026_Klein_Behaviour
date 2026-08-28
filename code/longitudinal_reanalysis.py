"""Reviewer-driven longitudinal reanalysis without PRR or change-score models."""

from __future__ import annotations

import argparse
import unicodedata
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from patsy import build_design_matrices
from scipy.special import expit
from scipy.stats import chi2
from scipy.stats import spearmanr
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE, OrdinalGEE
from statsmodels.stats.multitest import multipletests


HUMAN_SCALES = {
    "FM-UE": {"floor": 0.0, "ceiling": 126.0, "higher_better": True, "definition": "Extended FM-UE total including H/I/J; manuscript methods"},
    "FM-LE": {"floor": 0.0, "ceiling": 86.0, "higher_better": True, "definition": "Extended FM-LE total including H/I/J; manuscript methods"},
    "BI": {"floor": 0.0, "ceiling": 100.0, "higher_better": True, "definition": "Barthel Index standard range"},
    "NIHSS": {"floor": 0.0, "ceiling": 42.0, "higher_better": False, "definition": "NIHSS standard range"},
    "MRS": {"floor": 0.0, "ceiling": 6.0, "higher_better": False, "definition": "Modified Rankin Scale standard range"},
}

MOUSE_OUTCOMES = {
    "C_PawDragPercent": {"label": "Paw drag (%)", "floor": 0.0, "ceiling": 100.0},
    "GW_FootFault": {"label": "Grid-walk foot faults", "floor": 0.0, "ceiling": 100.0},
    "RB_HindlimbDrop": {"label": "Hindlimb drops", "floor": 0.0, "ceiling": None},
}


def ascii_text(value: object) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower().strip()


def classify_stroke(value: object) -> str:
    text = ascii_text(value)
    if not text:
        return "Unknown"
    if any(marker in text for marker in ("blutung", "icb", "sab")):
        return "Hemorrhage"
    if any(marker in text for marker in ("infarkt", "infrakt", "infakr", "infarct", "ischaem", "ischam")):
        return "Ischemic"
    return "Other/unknown"


def _melt_human_workbook(
    frame: pd.DataFrame,
    column_map: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    long = frame.melt(
        id_vars=["record_id"],
        value_vars=list(column_map),
        var_name="source_column",
        value_name="score",
    ).dropna(subset=["score"])
    mapped = long["source_column"].map(column_map)
    long["assessment"] = mapped.str[0]
    long["visit"] = mapped.str[1]
    return long[["record_id", "assessment", "visit", "score", "source_column"]]


def load_human_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = root / "input"
    fm_source = pd.read_excel(input_dir / "FM_JK_V1.xlsx")
    clinical_source = pd.read_excel(input_dir / "Assessments_JK_V1.2.xlsx")
    stroke_csv = pd.read_csv(input_dir / "Stroke_types.csv")
    stroke_xlsx = pd.read_excel(input_dir / "Stroke_types.xlsx")
    workbook_ids = set(fm_source["record_id"].dropna()) | set(clinical_source["record_id"].dropna())

    fm_map = {
        "T0_FM_UEx": ("FM-UE", "T0"),
        "T0_FM_LEx": ("FM-LE", "T0"),
        "T1_FM_Uex": ("FM-UE", "T1"),
        "T1_FM_Lex": ("FM-LE", "T1"),
        "T2_FM_UEx": ("FM-UE", "T2"),
        "T2_FM_LEx": ("FM-LE", "T2"),
    }
    clinical_map = {
        f"{assessment}_{visit}": (assessment, visit)
        for assessment in ("BI", "MRS", "NIHSS")
        for visit in ("T0", "T1", "T2")
    }

    human = pd.concat(
        [
            _melt_human_workbook(fm_source, fm_map),
            _melt_human_workbook(clinical_source, clinical_map),
        ],
        ignore_index=True,
    )
    duplicate_keys = human.duplicated(["record_id", "assessment", "visit"], keep=False)
    if duplicate_keys.any():
        duplicates = human.loc[duplicate_keys, ["record_id", "assessment", "visit"]]
        raise ValueError(f"Duplicate human subject/assessment/visit rows found:\n{duplicates}")

    stroke_csv = stroke_csv.rename(columns={"stroke_type": "stroke_type_raw"})
    human = human.merge(stroke_csv[["record_id", "stroke_type_raw"]], on="record_id", how="left")
    t0_ids = set(human.loc[human["visit"].eq("T0"), "record_id"])
    metadata_ids = set(stroke_csv["record_id"])
    analysis_ids = t0_ids & metadata_ids
    human["analysis_eligible"] = human["record_id"].isin(analysis_ids)
    human["cohort_note"] = np.where(
        human["analysis_eligible"],
        "Provisional T0 cohort with stroke metadata",
        "Outside provisional cohort; retained for audit",
    )
    human["stroke_category"] = human["stroke_type_raw"].map(classify_stroke)
    human["visit"] = pd.Categorical(human["visit"], categories=["T0", "T1", "T2"], ordered=True)

    human["floor"] = human["assessment"].map(lambda value: HUMAN_SCALES[value]["floor"])
    human["ceiling"] = human["assessment"].map(lambda value: HUMAN_SCALES[value]["ceiling"])
    human["valid_score"] = human["score"].between(human["floor"], human["ceiling"], inclusive="both")
    human["at_floor"] = human["valid_score"] & human["score"].eq(human["floor"])
    human["at_ceiling"] = human["valid_score"] & human["score"].eq(human["ceiling"])

    score_range = human["ceiling"] - human["floor"]
    higher_better = human["assessment"].map(lambda value: HUMAN_SCALES[value]["higher_better"])
    human["oriented_fraction"] = np.where(
        higher_better,
        (human["score"] - human["floor"]) / score_range,
        (human["ceiling"] - human["score"]) / score_range,
    )
    human.loc[~human["valid_score"], "oriented_fraction"] = np.nan
    human["deficit_fraction"] = 1 - human["oriented_fraction"]
    human["oriented_score"] = np.where(higher_better, human["score"], human["ceiling"] - human["score"])
    human["recovery_z"] = np.nan
    for assessment, group in human.loc[human["valid_score"] & human["analysis_eligible"]].groupby("assessment"):
        baseline = group.loc[group["visit"] == "T0", "oriented_score"]
        baseline_mean = baseline.mean()
        baseline_sd = baseline.std()
        mask = human["assessment"].eq(assessment) & human["valid_score"] & human["analysis_eligible"]
        human.loc[mask, "recovery_z"] = (human.loc[mask, "oriented_score"] - baseline_mean) / baseline_sd

    stroke_compare = stroke_csv.merge(
        stroke_xlsx.rename(columns={"stroke_type": "stroke_type_xlsx"}),
        on="record_id",
        how="outer",
    )
    same_stroke_text = (
        stroke_compare["stroke_type_raw"].map(ascii_text)
        == stroke_compare["stroke_type_xlsx"].map(ascii_text)
    )
    quality = pd.DataFrame(
        [
            {"check": "FM workbook rows", "value": len(fm_source), "detail": "source rows"},
            {"check": "Assessment workbook rows", "value": len(clinical_source), "detail": "source rows"},
            {"check": "Human records in either workbook", "value": len(workbook_ids), "detail": "all record IDs, including records without selected scores"},
            {"check": "Human records with any score", "value": human["record_id"].nunique(), "detail": "records with at least one selected score"},
            {"check": "Provisional human analysis cohort", "value": len(analysis_ids), "detail": "records with T0 score and stroke metadata; protocol eligibility ledger unavailable"},
            {"check": "Scored records outside provisional cohort", "value": human.loc[~human["analysis_eligible"], "record_id"].nunique(), "detail": "retained for audit but excluded from analyses"},
            {"check": "Human score observations", "value": len(human), "detail": "non-missing raw scores"},
            {"check": "Duplicate human keys", "value": int(duplicate_keys.sum()), "detail": "record/assessment/visit"},
            {"check": "Out-of-range human scores", "value": int((~human["valid_score"]).sum()), "detail": "preserved and excluded from models"},
            {"check": "Stroke CSV rows", "value": len(stroke_csv), "detail": "canonical stroke metadata"},
            {"check": "Stroke CSV/XLSX normalized text mismatches", "value": int((~same_stroke_text).sum()), "detail": "encoding-insensitive comparison"},
        ]
    )
    return human, quality


def load_mouse_input(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(root / "input" / "Mice_data.csv")
    animal_col = "StudyID_old"
    source["animal_id"] = source[animal_col].fillna(source["StudyID"])
    source["eligible"] = source["IncludedInStudy"].eq("Ja") & ~source["Exclude"].eq("Exclude")
    study_eligible = source.loc[source["eligible"]].copy()
    eligible = study_eligible.loc[study_eligible["TimePointMerged"].notna()].copy()

    metadata_columns = ["Group", "StrokeType", "Group name", "Strain"]
    metadata_conflicts = 0
    metadata_rows = []
    for animal_id, group in eligible.groupby("animal_id"):
        row = {"animal_id": animal_id}
        for column in metadata_columns:
            values = group[column].dropna().unique()
            if len(values) > 1:
                metadata_conflicts += 1
            row[column] = values[0] if len(values) else np.nan
        metadata_rows.append(row)
    metadata = pd.DataFrame(metadata_rows)

    duplicate_rows = eligible.duplicated(["animal_id", "TimePointMerged"], keep=False)
    aggregated = (
        eligible.groupby(["animal_id", "TimePointMerged"], as_index=False)[list(MOUSE_OUTCOMES)]
        .mean()
        .merge(metadata, on="animal_id", how="left")
        .rename(columns={"TimePointMerged": "day", "Group name": "study_group"})
    )
    aggregated["day"] = aggregated["day"].astype(int)
    aggregated["group"] = aggregated["Group"].fillna("Unknown")
    aggregated["stroke_type"] = aggregated["StrokeType"].fillna("Unknown")

    mouse_long = aggregated.melt(
        id_vars=["animal_id", "day", "group", "stroke_type", "study_group", "Strain"],
        value_vars=list(MOUSE_OUTCOMES),
        var_name="outcome",
        value_name="value",
    ).dropna(subset=["value"])
    mouse_long["floor"] = mouse_long["outcome"].map(lambda value: MOUSE_OUTCOMES[value]["floor"])
    mouse_long["ceiling"] = mouse_long["outcome"].map(lambda value: MOUSE_OUTCOMES[value]["ceiling"])
    has_ceiling = mouse_long["ceiling"].notna()
    mouse_long["valid_value"] = mouse_long["value"].ge(mouse_long["floor"]) & (
        ~has_ceiling | mouse_long["value"].le(mouse_long["ceiling"])
    )
    mouse_long["at_floor"] = mouse_long["valid_value"] & mouse_long["value"].eq(mouse_long["floor"])
    mouse_long["at_ceiling"] = has_ceiling & mouse_long["valid_value"] & mouse_long["value"].eq(mouse_long["ceiling"])

    quality = pd.DataFrame(
        [
            {"check": "Mouse CSV rows", "value": len(source), "detail": "source rows"},
            {"check": "Mouse CSV animals", "value": source["animal_id"].nunique(), "detail": "unique StudyID_old with StudyID fallback"},
            {"check": "Study-eligible mouse rows", "value": len(study_eligible), "detail": "IncludedInStudy=Ja and not Exclude"},
            {"check": "Eligible mouse rows with day", "value": len(eligible), "detail": "modeled source rows"},
            {"check": "Eligible mouse rows missing day", "value": int(study_eligible["TimePointMerged"].isna().sum()), "detail": "not modeled"},
            {"check": "Excluded/non-study mouse rows", "value": int((~source["eligible"]).sum()), "detail": "not modeled"},
            {"check": "Eligible mouse animals", "value": eligible["animal_id"].nunique(), "detail": "unique StudyID_old"},
            {"check": "Replicate mouse rows", "value": int(duplicate_rows.sum()), "detail": "rows in duplicated animal/day groups; averaged"},
            {"check": "Aggregated animal/day rows", "value": len(aggregated), "detail": "one row per animal/day"},
            {"check": "Mouse metadata conflicts", "value": metadata_conflicts, "detail": "animal/metadata fields with multiple values"},
            {"check": "Out-of-range mouse values", "value": int((~mouse_long["valid_value"]).sum()), "detail": "excluded from models"},
        ]
    )
    return mouse_long, quality


def availability_table(
    data: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    id_col: str,
) -> pd.DataFrame:
    return (
        data.groupby([outcome_col, time_col], observed=True)
        .agg(n_observations=(id_col, "size"), n_subjects=(id_col, "nunique"))
        .reset_index()
    )


def human_subject_flow(human: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    provisional = human.loc[human["analysis_eligible"]].copy()
    cohort_n = provisional["record_id"].nunique()
    stage_rows = [
        {"stage": "Records with at least one supplied score", "n_subjects": human["record_id"].nunique(), "status": "Observed in supplied score columns"},
        {"stage": "Provisional T0 cohort with stroke metadata", "n_subjects": cohort_n, "status": "Used for current analyses pending protocol ledger"},
        {"stage": "Scored records outside provisional cohort", "n_subjects": human.loc[~human["analysis_eligible"], "record_id"].nunique(), "status": "Retained in audit export; excluded from analyses"},
        {"stage": "Protocol-eligible cohort", "n_subjects": np.nan, "status": "Not recoverable from supplied files; authoritative 91-ID ledger required"},
    ]

    availability_rows = []
    for (assessment, visit), group in provisional.groupby(["assessment", "visit"], observed=True):
        available_ids = group["record_id"].nunique()
        invalid_ids = group.loc[~group["valid_score"], "record_id"].nunique()
        analyzed_ids = group.loc[group["valid_score"], "record_id"].nunique()
        availability_rows.append(
            {
                "assessment": assessment,
                "visit": visit,
                "provisional_cohort_n": cohort_n,
                "score_available_n": available_ids,
                "score_missing_n": cohort_n - available_ids,
                "invalid_score_n": invalid_ids,
                "analyzed_n": analyzed_ids,
            }
        )
    return pd.DataFrame(stage_rows), pd.DataFrame(availability_rows)


def mouse_subject_flow(mouse: pd.DataFrame, quality: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_animals = int(quality.loc[quality["check"] == "Mouse CSV animals", "value"].item())
    eligible_animals = mouse["animal_id"].nunique()
    group_counts = mouse.groupby("group")["animal_id"].nunique()
    stage_rows = [
        {"stage": "Animals in source CSV", "n_animals": source_animals, "rule": "Unique StudyID_old with StudyID fallback"},
        {"stage": "Eligible animals", "n_animals": eligible_animals, "rule": "IncludedInStudy=Ja and row not marked Exclude"},
        {"stage": "Eligible stroke animals", "n_animals": int(group_counts.get("Stroke", 0)), "rule": "Eligible animals with Group=Stroke"},
        {"stage": "Eligible sham animals", "n_animals": int(group_counts.get("Sham", 0)), "rule": "Eligible animals with Group=Sham"},
    ]

    availability_rows = []
    group_denominators = mouse.groupby("group")["animal_id"].nunique()
    for (outcome, group_name, day), group in mouse.groupby(["outcome", "group", "day"]):
        available = group["animal_id"].nunique()
        valid = group.loc[group["valid_value"], "animal_id"].nunique()
        denominator = int(group_denominators[group_name])
        availability_rows.append(
            {
                "outcome": outcome,
                "group": group_name,
                "day": day,
                "eligible_group_n": denominator,
                "outcome_available_n": available,
                "outcome_missing_n": denominator - available,
                "invalid_value_n": available - valid,
                "analyzed_n": valid,
            }
        )
    return pd.DataFrame(stage_rows), pd.DataFrame(availability_rows)


def boundary_table(
    data: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    value_col: str,
    valid_col: str,
) -> pd.DataFrame:
    rows = []
    for keys, group in data.groupby([outcome_col, time_col], observed=True):
        outcome, time = keys
        valid = group.loc[group[valid_col]].copy()
        floor = valid["floor"].iloc[0] if len(valid) else group["floor"].iloc[0]
        ceiling = valid["ceiling"].iloc[0] if len(valid) else group["ceiling"].iloc[0]
        width = ceiling - floor if pd.notna(ceiling) else np.nan
        near_floor = valid[value_col].le(floor + 0.1 * width) if pd.notna(width) else pd.Series(False, index=valid.index)
        near_ceiling = valid[value_col].ge(ceiling - 0.1 * width) if pd.notna(width) else pd.Series(False, index=valid.index)
        n = len(valid)
        at_floor = int(valid["at_floor"].sum())
        at_ceiling = int(valid["at_ceiling"].sum())
        rows.append(
            {
                outcome_col: outcome,
                time_col: time,
                "n": n,
                "n_invalid": int((~group[valid_col]).sum()),
                "floor": floor,
                "ceiling": ceiling,
                "n_at_floor": at_floor,
                "pct_at_floor": 100 * at_floor / n if n else np.nan,
                "n_at_ceiling": at_ceiling,
                "pct_at_ceiling": 100 * at_ceiling / n if n else np.nan,
                "n_near_floor_10pct_range": int(near_floor.sum()),
                "n_near_ceiling_10pct_range": int(near_ceiling.sum()),
                "median": valid[value_col].median(),
                "q1": valid[value_col].quantile(0.25),
                "q3": valid[value_col].quantile(0.75),
                "minimum": valid[value_col].min(),
                "maximum": valid[value_col].max(),
                "n_distinct": valid[value_col].nunique(),
            }
        )
    return pd.DataFrame(rows)


def descriptive_table(
    data: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    id_col: str,
) -> pd.DataFrame:
    return (
        data.groupby(group_cols, observed=True)[value_col]
        .agg(n_observations="size", mean="mean", sd="std", median="median", q1=lambda values: values.quantile(0.25), q3=lambda values: values.quantile(0.75), minimum="min", maximum="max")
        .join(data.groupby(group_cols, observed=True)[id_col].nunique().rename("n_subjects"))
        .reset_index()
    )


def human_responsiveness_tables(human: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = human.loc[human["valid_score"]].copy()
    standardized = descriptive_table(valid, ["assessment", "visit"], "recovery_z", "record_id")
    standardized = standardized.rename(columns={"mean": "standardized_mean", "sd": "standardized_sd"})

    paired_rows = []
    for assessment, group in valid.groupby("assessment"):
        baseline = group.loc[group["visit"] == "T0", ["record_id", "oriented_score"]].rename(columns={"oriented_score": "baseline_oriented"})
        for visit in ("T1", "T2"):
            paired = group.loc[group["visit"] == visit, ["record_id", "oriented_score"]].merge(baseline, on="record_id", how="inner")
            change = paired["oriented_score"] - paired["baseline_oriented"]
            paired_rows.append(
                {
                    "assessment": assessment,
                    "visit": visit,
                    "n_pairs": len(change),
                    "mean_oriented_change": change.mean(),
                    "sd_oriented_change": change.std(),
                    "standardized_response_mean": change.mean() / change.std() if change.std() > 0 else np.nan,
                }
            )
    return standardized, pd.DataFrame(paired_rows)


def human_nominal_slopes_and_correlations(human: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = human.loc[human["valid_score"]].copy()
    visit_number = {"T0": 0.0, "T1": 1.0, "T2": 2.0}
    valid["visit_number"] = valid["visit"].astype(str).map(visit_number)
    rows = []
    for (record_id, assessment), group in valid.groupby(["record_id", "assessment"]):
        group = group.dropna(subset=["recovery_z", "visit_number"]).sort_values("visit_number")
        if group["visit_number"].nunique() != 3:
            continue
        slope, intercept = np.polyfit(group["visit_number"], group["recovery_z"], 1)
        rows.append({"record_id": record_id, "assessment": assessment, "n_visits": 3, "nominal_visit_slope": slope, "intercept": intercept})
    slopes = pd.DataFrame(rows)
    wide = slopes.pivot(index="record_id", columns="assessment", values="nominal_visit_slope")
    correlations = []
    rng = np.random.default_rng(42)
    columns = list(HUMAN_SCALES)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            paired = wide[[left, right]].dropna()
            if len(paired) < 4:
                continue
            rho, p_value = spearmanr(paired[left], paired[right])
            bootstrap = []
            for _ in range(1000):
                sample = paired.iloc[rng.integers(0, len(paired), len(paired))]
                estimate = spearmanr(sample[left], sample[right]).statistic
                if np.isfinite(estimate):
                    bootstrap.append(estimate)
            correlations.append(
                {
                    "assessment_1": left,
                    "assessment_2": right,
                    "n_pairs": len(paired),
                    "spearman_rho": rho,
                    "ci_95_low": np.quantile(bootstrap, 0.025),
                    "ci_95_high": np.quantile(bootstrap, 0.975),
                    "p_value": p_value,
                }
            )
    correlation_table = pd.DataFrame(correlations)
    if not correlation_table.empty:
        correlation_table["p_value_fdr_bh"] = multipletests(correlation_table["p_value"], method="fdr_bh")[1]
    return slopes, correlation_table


def fit_mixed_model(data: pd.DataFrame, formula: str, group_col: str):
    model = smf.mixedlm(formula, data=data, groups=data[group_col], re_formula="1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(reml=False, method=["lbfgs", "powell"], maxiter=2000, disp=False)
    warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
    if not result.converged:
        raise RuntimeError("Mixed model did not converge")
    if not np.isfinite(result.fe_params).all() or not np.isfinite(result.bse_fe).all():
        raise RuntimeError("Mixed model returned non-finite fixed-effect estimates")
    if float(result.cov_re.iloc[0, 0]) <= 1e-8:
        raise RuntimeError("Mixed model random-intercept variance is on the boundary")
    if "singular" in warning_text.lower() or "not positive definite" in warning_text.lower():
        raise RuntimeError(f"Mixed model covariance is unreliable: {warning_text}")
    return result, warning_text


def fit_gaussian_gee(data: pd.DataFrame, formula: str, group_col: str):
    model = GEE.from_formula(
        formula,
        groups=group_col,
        data=data,
        family=Gaussian(),
        cov_struct=Exchangeable(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(maxiter=200)
        standard_errors = result.bse
    warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
    if not np.isfinite(result.params).all() or not np.isfinite(standard_errors).all():
        raise RuntimeError("Gaussian GEE returned non-finite estimates")
    return result, warning_text


def fit_continuous_repeated(data: pd.DataFrame, formula: str, group_col: str):
    try:
        result, warning_text = fit_mixed_model(data, formula, group_col)
        return result, "random_intercept_LMM", warning_text
    except Exception as mixed_error:
        result, warning_text = fit_gaussian_gee(data, formula, group_col)
        fallback = f"LMM fallback: {mixed_error!r}"
        warning_text = f"{fallback} | {warning_text}" if warning_text else fallback
        return result, "Gaussian_GEE_fallback", warning_text


def mixed_coefficients(result, metadata: dict[str, object], warning_text: str) -> list[dict[str, object]]:
    confidence = result.conf_int()
    rows = []
    for term in result.fe_params.index:
        rows.append(
            {
                **metadata,
                "term": term,
                "estimate": result.fe_params[term],
                "std_error": result.bse_fe[term],
                "ci_95_low": confidence.loc[term, 0],
                "ci_95_high": confidence.loc[term, 1],
                "p_value": result.pvalues[term],
                "n_observations": result.nobs,
                "n_subjects": len(result.model.group_labels),
                "random_intercept_variance": float(result.cov_re.iloc[0, 0]),
                "converged": bool(result.converged),
                "warnings": warning_text,
            }
        )
    return rows


def fit_ordinal_gee(data: pd.DataFrame, formula: str, group_col: str):
    model = OrdinalGEE.from_formula(
        formula,
        groups=group_col,
        data=data,
        cov_struct=Exchangeable(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(maxiter=200)
        standard_errors = result.bse
    warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
    if "iteration limit" in warning_text.lower():
        raise RuntimeError("Ordinal GEE did not converge")
    if not np.isfinite(result.params).all() or not np.isfinite(standard_errors).all():
        raise RuntimeError("Ordinal GEE returned non-finite estimates")
    if "Intercept" in result.params.index:
        raise RuntimeError("Ordinal GEE must not include an intercept")
    if result.params.abs().max() > 100 or standard_errors.abs().max() > 100:
        raise RuntimeError("Ordinal GEE returned implausibly large estimates or standard errors")
    covariance = result.cov_params().to_numpy(dtype=float)
    if not np.isfinite(covariance).all() or np.linalg.cond(covariance) > 1e12:
        raise RuntimeError("Ordinal GEE covariance is numerically unstable")
    return result, warning_text


def fit_mrs_repeated(data: pd.DataFrame, rhs: str, group_col: str):
    fallback_reasons = []
    try:
        result, warning_text = fit_ordinal_gee(data, f"score ~ 0 + {rhs}", group_col)
        return result, "ordinal_GEE", warning_text
    except Exception as ordinal_error:
        fallback_reasons.append(f"Ordinal GEE fallback: {ordinal_error!r}")
        binary = data.copy()
        binary["favorable_mrs"] = binary["score"].le(2).astype(int)
        classes_per_visit = binary.groupby("visit", observed=True)["favorable_mrs"].nunique()
        try:
            if (classes_per_visit < 2).any():
                raise RuntimeError("At least one visit has complete binary-outcome separation")
            model = GEE.from_formula(
                f"favorable_mrs ~ {rhs}",
                groups=group_col,
                data=binary,
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.fit(maxiter=200)
                standard_errors = result.bse
            warning_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
            if not np.isfinite(result.params).all() or not np.isfinite(standard_errors).all():
                raise RuntimeError("Binary mRS GEE returned non-finite estimates")
            if result.params.abs().max() > 100:
                raise RuntimeError("Binary mRS GEE returned implausibly large estimates")
            warning_parts = fallback_reasons + ([warning_text] if warning_text else [])
            return result, "binary_logistic_GEE_mRS_0_2", " | ".join(warning_parts)
        except Exception as binary_error:
            fallback_reasons.append(f"Binary GEE fallback: {binary_error!r}")
            result, warning_text = fit_gaussian_gee(data, f"score ~ {rhs}", group_col)
            warning_parts = fallback_reasons + ([warning_text] if warning_text else [])
            return result, "Gaussian_GEE_raw_mRS_fallback", " | ".join(warning_parts)


def gee_coefficients(result, metadata: dict[str, object], warning_text: str) -> list[dict[str, object]]:
    confidence = result.conf_int()
    rows = []
    for term in result.params.index:
        rows.append(
            {
                **metadata,
                "term": term,
                "estimate": result.params[term],
                "std_error": result.bse[term],
                "ci_95_low": confidence.loc[term, 0],
                "ci_95_high": confidence.loc[term, 1],
                "p_value": result.pvalues[term],
                "n_observations": result.nobs,
                "n_subjects": result.model.num_group,
                "random_intercept_variance": np.nan,
                "converged": bool(getattr(result, "converged", True)),
                "warnings": warning_text,
            }
        )
    return rows


def parameter_wald_test(result, term_tokens: tuple[str, ...]) -> tuple[float, int, float]:
    terms = list(result.params.index)
    selected = [index for index, term in enumerate(terms) if all(token in term for token in term_tokens)]
    if not selected:
        raise ValueError(f"No model terms matched {term_tokens}")
    selected_terms = [terms[index] for index in selected]
    estimates = result.params.loc[selected_terms].to_numpy(dtype=float)
    covariance = result.cov_params().loc[selected_terms, selected_terms].to_numpy(dtype=float)
    statistic = float(estimates.T @ np.linalg.pinv(covariance) @ estimates)
    degrees = len(selected_terms)
    return statistic, degrees, float(chi2.sf(statistic, degrees))


def apply_fdr(tests: pd.DataFrame) -> pd.DataFrame:
    tests = tests.copy()
    tests["p_value_fdr_bh"] = np.nan
    if tests.empty:
        return tests
    for _, indices in tests.groupby("family").groups.items():
        valid = tests.loc[indices, "p_value"].notna()
        valid_indices = tests.loc[indices].index[valid]
        if len(valid_indices):
            tests.loc[valid_indices, "p_value_fdr_bh"] = multipletests(
                tests.loc[valid_indices, "p_value"], method="fdr_bh"
            )[1]
    return tests


def prepare_baseline_moderation_data(assessment_data: pd.DataFrame, variant: str) -> pd.DataFrame:
    data = assessment_data.copy()
    if variant == "exclude_baseline_boundaries":
        boundary_ids = data.loc[
            data["visit"].eq("T0") & (data["at_floor"] | data["at_ceiling"]), "record_id"
        ]
        data = data.loc[~data["record_id"].isin(boundary_ids)].copy()
    elif variant == "exclude_boundary_observations":
        data = data.loc[~(data["at_floor"] | data["at_ceiling"])].copy()
    elif variant != "primary":
        raise ValueError(f"Unknown baseline-moderation variant: {variant}")

    baseline = data.loc[data["visit"] == "T0", ["record_id", "score", "deficit_fraction"]].rename(
        columns={"score": "baseline_score", "deficit_fraction": "baseline_deficit"}
    )
    followup = data.loc[data["visit"].isin(["T1", "T2"])].merge(baseline, on="record_id", how="inner")
    followup["visit"] = followup["visit"].cat.remove_unused_categories()
    subject_baselines = followup[["record_id", "baseline_deficit"]].drop_duplicates()
    baseline_mean = subject_baselines["baseline_deficit"].mean()
    baseline_sd = subject_baselines["baseline_deficit"].std()
    if not len(followup) or not np.isfinite(baseline_sd) or baseline_sd <= 0:
        return pd.DataFrame()
    followup["baseline_deficit_z"] = (followup["baseline_deficit"] - baseline_mean) / baseline_sd
    followup.attrs["baseline_mean"] = baseline_mean
    followup.attrs["baseline_sd"] = baseline_sd
    return followup


def fit_human_baseline_model(followup: pd.DataFrame, assessment: str):
    if assessment == "MRS":
        return fit_mrs_repeated(followup, "C(visit) * baseline_deficit_z", "record_id")
    return fit_continuous_repeated(followup, "score ~ C(visit) * baseline_deficit_z", "record_id")


def result_coefficient_rows(result, metadata: dict[str, object], model_name: str, warning_text: str) -> list[dict[str, object]]:
    if model_name == "random_intercept_LMM":
        return mixed_coefficients(result, metadata, warning_text)
    return gee_coefficients(result, metadata, warning_text)


def baseline_model_predictions(result, model_name: str, assessment: str, followup: pd.DataFrame) -> list[dict[str, object]]:
    if model_name == "ordinal_GEE":
        raise RuntimeError("Ordinal GEE category-probability plotting is not implemented")
    subject_baselines = followup[["record_id", "baseline_deficit"]].drop_duplicates()
    baseline_mean = followup.attrs["baseline_mean"]
    baseline_sd = followup.attrs["baseline_sd"]
    deficit_grid = np.linspace(subject_baselines["baseline_deficit"].min(), subject_baselines["baseline_deficit"].max(), 150)
    parameters = result.fe_params if model_name == "random_intercept_LMM" else result.params
    covariance = result.cov_params().loc[parameters.index, parameters.index].to_numpy(dtype=float)
    rows = []
    for visit in ("T1", "T2"):
        prediction_data = pd.DataFrame(
            {"visit": visit, "baseline_deficit": deficit_grid, "baseline_deficit_z": (deficit_grid - baseline_mean) / baseline_sd}
        )
        prediction_data["visit"] = pd.Categorical(prediction_data["visit"], categories=followup["visit"].cat.categories, ordered=True)
        design = np.asarray(build_design_matrices([result.model.data.design_info], prediction_data)[0])
        linear_prediction = design @ parameters.to_numpy(dtype=float)
        linear_se = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))
        if model_name == "binary_logistic_GEE_mRS_0_2":
            estimates = expit(linear_prediction)
            lower = expit(linear_prediction - 1.96 * linear_se)
            upper = expit(linear_prediction + 1.96 * linear_se)
            estimand = "Probability of favorable mRS (0-2)"
        else:
            estimates = linear_prediction
            lower = linear_prediction - 1.96 * linear_se
            upper = linear_prediction + 1.96 * linear_se
            estimand = "Raw follow-up score"
        for deficit, estimate, ci_low, ci_high in zip(deficit_grid, estimates, lower, upper):
            rows.append(
                {"assessment": assessment, "visit": visit, "baseline_deficit": deficit, "predicted_outcome": estimate, "ci_95_low": ci_low, "ci_95_high": ci_high, "estimand": estimand, "model": model_name, "n_observations": len(followup), "n_subjects": followup["record_id"].nunique()}
            )
    return rows


def run_human_models(human: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients: list[dict[str, object]] = []
    tests: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []

    for assessment in HUMAN_SCALES:
        assessment_data = human.loc[(human["assessment"] == assessment) & human["valid_score"]].copy()
        assessment_data["visit"] = assessment_data["visit"].cat.remove_unused_categories()
        try:
            if assessment == "MRS":
                result, model_name, warning_text = fit_mrs_repeated(assessment_data, "C(visit)", "record_id")
            else:
                result, model_name, warning_text = fit_continuous_repeated(assessment_data, "score ~ C(visit)", "record_id")
            metadata = {"assessment": assessment, "analysis": "raw_trajectory", "model": model_name}
            coefficients.extend(result_coefficient_rows(result, metadata, model_name, warning_text))
            statistic, degrees, p_value = parameter_wald_test(result, ("C(visit)",))
            tests.append({"assessment": assessment, "analysis": "raw_trajectory", "model": model_name, "family": "human_visit_effect", "contrast": "overall visit effect", "statistic": statistic, "df": degrees, "p_value": p_value, "n_observations": len(assessment_data), "n_subjects": assessment_data["record_id"].nunique()})
        except Exception as error:
            failures.append({"dataset": "human", "outcome": assessment, "analysis": "raw_trajectory", "error": repr(error)})

        for variant in ("primary", "exclude_baseline_boundaries", "exclude_boundary_observations"):
            followup = prepare_baseline_moderation_data(assessment_data, variant)
            if followup.empty or followup["visit"].nunique() < 2:
                continue
            analysis_name = "baseline_moderation" if variant == "primary" else f"baseline_moderation_{variant}"
            family = "human_visit_by_baseline" if variant == "primary" else f"human_visit_by_baseline_{variant}"
            try:
                result, model_name, warning_text = fit_human_baseline_model(followup, assessment)
                metadata = {"assessment": assessment, "analysis": analysis_name, "model": model_name}
                coefficients.extend(result_coefficient_rows(result, metadata, model_name, warning_text))
                statistic, degrees, p_value = parameter_wald_test(result, ("C(visit)", "baseline_deficit_z"))
                tests.append({"assessment": assessment, "analysis": analysis_name, "model": model_name, "family": family, "contrast": "visit by continuous baseline severity", "statistic": statistic, "df": degrees, "p_value": p_value, "n_observations": len(followup), "n_subjects": followup["record_id"].nunique()})
                if variant == "primary":
                    predictions.extend(baseline_model_predictions(result, model_name, assessment, followup))
            except Exception as error:
                failures.append({"dataset": "human", "outcome": assessment, "analysis": analysis_name, "error": repr(error)})

        for variant in ("exclude_baseline_boundaries", "exclude_boundary_observations"):
            if variant == "exclude_baseline_boundaries":
                boundary_ids = assessment_data.loc[(assessment_data["visit"] == "T0") & (assessment_data["at_floor"] | assessment_data["at_ceiling"]), "record_id"]
                sensitivity = assessment_data.loc[~assessment_data["record_id"].isin(boundary_ids)].copy()
            else:
                sensitivity = assessment_data.loc[~(assessment_data["at_floor"] | assessment_data["at_ceiling"])].copy()
            sensitivity["visit"] = sensitivity["visit"].cat.remove_unused_categories()
            if sensitivity["visit"].nunique() < 2:
                continue
            try:
                if assessment == "MRS":
                    result, model_name, warning_text = fit_mrs_repeated(sensitivity, "C(visit)", "record_id")
                else:
                    result, model_name, warning_text = fit_continuous_repeated(sensitivity, "score ~ C(visit)", "record_id")
                metadata = {"assessment": assessment, "analysis": variant, "model": model_name}
                coefficients.extend(result_coefficient_rows(result, metadata, model_name, warning_text))
                statistic, degrees, p_value = parameter_wald_test(result, ("C(visit)",))
                tests.append({"assessment": assessment, "analysis": variant, "model": model_name, "family": f"human_{variant}", "contrast": "overall visit effect", "statistic": statistic, "df": degrees, "p_value": p_value, "n_observations": len(sensitivity), "n_subjects": sensitivity["record_id"].nunique()})
            except Exception as error:
                failures.append({"dataset": "human", "outcome": assessment, "analysis": variant, "error": repr(error)})

    failure_columns = ["dataset", "outcome", "analysis", "error"]
    return pd.DataFrame(coefficients), apply_fdr(pd.DataFrame(tests)), pd.DataFrame(failures, columns=failure_columns), pd.DataFrame(predictions)


def run_mouse_models(mouse: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coefficients: list[dict[str, object]] = []
    tests: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for outcome in MOUSE_OUTCOMES:
        outcome_data = mouse.loc[(mouse["outcome"] == outcome) & mouse["valid_value"]].copy()
        day_order = sorted(outcome_data["day"].unique())
        outcome_data["day_factor"] = pd.Categorical(outcome_data["day"], categories=day_order, ordered=True)
        outcome_data["group"] = pd.Categorical(outcome_data["group"], categories=["Sham", "Stroke"])
        outcome_data = outcome_data.dropna(subset=["group"])
        try:
            result, model_name, warning_text = fit_continuous_repeated(outcome_data, "value ~ C(day_factor) * C(group)", "animal_id")
            metadata = {"assessment": outcome, "analysis": "group_trajectory", "model": model_name}
            if model_name == "random_intercept_LMM":
                coefficients.extend(mixed_coefficients(result, metadata, warning_text))
            else:
                coefficients.extend(gee_coefficients(result, metadata, warning_text))
            statistic, degrees, p_value = parameter_wald_test(result, ("C(day_factor)", "C(group)"))
            tests.append({"assessment": outcome, "analysis": "group_trajectory", "family": "mouse_day_by_group", "contrast": "day by sham/stroke group", "statistic": statistic, "df": degrees, "p_value": p_value, "n_observations": len(outcome_data), "n_subjects": outcome_data["animal_id"].nunique()})
        except Exception as error:
            failures.append({"dataset": "mouse", "outcome": outcome, "analysis": "group_trajectory", "error": repr(error)})

        stroke = outcome_data.loc[outcome_data["group"] == "Stroke"].copy()
        acute = stroke.loc[stroke["day"] == 3, ["animal_id", "value"]].rename(columns={"value": "acute_day3"})
        followup = stroke.loc[stroke["day"] >= 7].merge(acute, on="animal_id", how="inner")
        if len(followup):
            followup["day_factor"] = followup["day_factor"].cat.remove_unused_categories()
            acute_values = followup[["animal_id", "acute_day3"]].drop_duplicates()["acute_day3"]
            acute_mean = acute_values.mean()
            acute_sd = acute_values.std()
            if pd.notna(acute_sd) and acute_sd > 0:
                followup["acute_day3_z"] = (followup["acute_day3"] - acute_mean) / acute_sd
                try:
                    result, model_name, warning_text = fit_continuous_repeated(
                        followup,
                        "value ~ C(day_factor) * acute_day3_z + C(stroke_type)",
                        "animal_id",
                    )
                    metadata = {"assessment": outcome, "analysis": "acute_severity_moderation", "model": model_name}
                    if model_name == "random_intercept_LMM":
                        coefficients.extend(mixed_coefficients(result, metadata, warning_text))
                    else:
                        coefficients.extend(gee_coefficients(result, metadata, warning_text))
                    statistic, degrees, p_value = parameter_wald_test(result, ("C(day_factor)", "acute_day3_z"))
                    tests.append({"assessment": outcome, "analysis": "acute_severity_moderation", "family": "mouse_day_by_acute_severity", "contrast": "day by continuous day-3 severity", "statistic": statistic, "df": degrees, "p_value": p_value, "n_observations": len(followup), "n_subjects": followup["animal_id"].nunique()})
                except Exception as error:
                    failures.append({"dataset": "mouse", "outcome": outcome, "analysis": "acute_severity_moderation", "error": repr(error)})

    failure_columns = ["dataset", "outcome", "analysis", "error"]
    return pd.DataFrame(coefficients), apply_fdr(pd.DataFrame(tests)), pd.DataFrame(failures, columns=failure_columns)


def human_baseline_severity_predictions(human: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for assessment in HUMAN_SCALES:
        subset = human.loc[(human["assessment"] == assessment) & human["valid_score"]].copy()
        baseline = subset.loc[subset["visit"] == "T0", ["record_id", "score", "deficit_fraction"]].rename(
            columns={"score": "baseline_score", "deficit_fraction": "baseline_deficit"}
        )
        followup = subset.loc[subset["visit"].isin(["T1", "T2"])].merge(baseline, on="record_id", how="inner")
        followup["visit"] = followup["visit"].cat.remove_unused_categories()
        subject_baselines = followup[["record_id", "baseline_deficit"]].drop_duplicates()
        baseline_mean = subject_baselines["baseline_deficit"].mean()
        baseline_sd = subject_baselines["baseline_deficit"].std()
        if not len(followup) or not np.isfinite(baseline_sd) or baseline_sd <= 0:
            continue
        followup["baseline_deficit_z"] = (followup["baseline_deficit"] - baseline_mean) / baseline_sd
        result, _ = fit_gaussian_gee(followup, "score ~ C(visit) * baseline_deficit_z", "record_id")
        quantiles = subject_baselines["baseline_deficit"].quantile([0.25, 0.75])
        prediction_data = pd.DataFrame(
            [
                {"visit": visit, "baseline_deficit": deficit, "baseline_deficit_z": (deficit - baseline_mean) / baseline_sd, "severity_reference": label}
                for label, deficit in (("Q25 lower initial deficit", quantiles.loc[0.25]), ("Q75 higher initial deficit", quantiles.loc[0.75]))
                for visit in ("T1", "T2")
            ]
        )
        prediction_data["visit"] = pd.Categorical(prediction_data["visit"], categories=followup["visit"].cat.categories, ordered=True)
        design = np.asarray(build_design_matrices([result.model.data.design_info], prediction_data)[0])
        estimates = design @ result.params.to_numpy()
        covariance = result.cov_params().to_numpy()
        standard_errors = np.sqrt(np.einsum("ij,jk,ik->i", design, covariance, design))
        for prediction, estimate, standard_error in zip(prediction_data.to_dict("records"), estimates, standard_errors):
            rows.append(
                {
                    "assessment": assessment,
                    **prediction,
                    "predicted_score": estimate,
                    "ci_95_low": estimate - 1.96 * standard_error,
                    "ci_95_high": estimate + 1.96 * standard_error,
                    "n_subjects": followup["record_id"].nunique(),
                    "model": "Gaussian GEE visualization; continuous deficit used in model",
                }
            )
        score_range = HUMAN_SCALES[assessment]["ceiling"] - HUMAN_SCALES[assessment]["floor"]
        for label, deficit in (("Q25 lower initial deficit", quantiles.loc[0.25]), ("Q75 higher initial deficit", quantiles.loc[0.75])):
            if HUMAN_SCALES[assessment]["higher_better"]:
                baseline_score = HUMAN_SCALES[assessment]["ceiling"] - deficit * score_range
            else:
                baseline_score = HUMAN_SCALES[assessment]["floor"] + deficit * score_range
            rows.append(
                {
                    "assessment": assessment,
                    "visit": "T0",
                    "baseline_deficit": deficit,
                    "baseline_deficit_z": (deficit - baseline_mean) / baseline_sd,
                    "severity_reference": label,
                    "predicted_score": baseline_score,
                    "ci_95_low": np.nan,
                    "ci_95_high": np.nan,
                    "n_subjects": followup["record_id"].nunique(),
                    "model": "Observed baseline quantile anchor",
                }
            )
    predictions = pd.DataFrame(rows)
    predictions["visit"] = pd.Categorical(predictions["visit"], categories=["T0", "T1", "T2"], ordered=True)
    return predictions.sort_values(["assessment", "severity_reference", "visit"])


def cross_species_stage_summary(human: pd.DataFrame, mouse: pd.DataFrame) -> pd.DataFrame:
    pairings = [
        ("FM-UE / Paw drag", "FM-UE", "C_PawDragPercent"),
        ("FM-UE / Grid walk", "FM-UE", "GW_FootFault"),
        ("FM-LE / Hindlimb drop", "FM-LE", "RB_HindlimbDrop"),
    ]
    human_stages = {"T0": "Acute", "T1": "Early", "T2": "Late"}
    mouse_stages = {3: "Acute", 14: "Early", 56: "Late"}
    rows = []
    for pairing, human_outcome, mouse_outcome in pairings:
        human_values = human.loc[(human["assessment"] == human_outcome) & human["valid_score"] & human["visit"].isin(human_stages)].copy()
        human_values["stage"] = human_values["visit"].astype(str).map(human_stages)
        human_values["standardized_recovery"] = human_values["recovery_z"]
        for stage, group in human_values.groupby("stage"):
            rows.append({"pairing": pairing, "species": "Human", "outcome": human_outcome, "stage": stage, "timepoint": group["visit"].astype(str).iloc[0], "n": len(group), "mean": group["standardized_recovery"].mean(), "sd": group["standardized_recovery"].std(), "sem": group["standardized_recovery"].sem()})

        mouse_values = mouse.loc[(mouse["outcome"] == mouse_outcome) & mouse["valid_value"] & mouse["group"].eq("Stroke") & mouse["day"].isin(mouse_stages)].copy()
        acute = mouse_values.loc[mouse_values["day"] == 3, "value"]
        acute_mean = acute.mean()
        acute_sd = acute.std()
        mouse_values["standardized_recovery"] = (acute_mean - mouse_values["value"]) / acute_sd
        mouse_values["stage"] = mouse_values["day"].map(mouse_stages)
        for stage, group in mouse_values.groupby("stage"):
            rows.append({"pairing": pairing, "species": "Mouse", "outcome": MOUSE_OUTCOMES[mouse_outcome]["label"], "stage": stage, "timepoint": f"Day {int(group['day'].iloc[0])}", "n": len(group), "mean": group["standardized_recovery"].mean(), "sd": group["standardized_recovery"].std(), "sem": group["standardized_recovery"].sem()})
    summary = pd.DataFrame(rows)
    summary["ci_95_low"] = summary["mean"] - 1.96 * summary["sem"]
    summary["ci_95_high"] = summary["mean"] + 1.96 * summary["sem"]
    return summary


def set_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Calibri",
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.grid": False,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".svg"), format="svg", dpi=300)
    figure.savefig(output_stem.with_suffix(".png"), format="png", dpi=300)
    plt.close(figure)


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.grid(False)


def plot_human_trajectories(human: pd.DataFrame, output_stem: Path) -> None:
    set_figure_style()
    order = ["FM-LE", "FM-UE", "BI", "MRS", "NIHSS"]
    positions = {"T0": 0, "T1": 1, "T2": 2}
    valid = human.loc[human["valid_score"]].copy()
    figure, axes = plt.subplots(1, len(order), figsize=(18 / 2.54, 5.7 / 2.54), dpi=300)

    for axis, assessment in zip(axes, order):
        subset = valid.loc[valid["assessment"] == assessment].copy()
        distributions = [subset.loc[subset["visit"] == visit, "score"] for visit in positions]
        boxplot = axis.boxplot(
            distributions,
            positions=list(positions.values()),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            zorder=0,
        )
        for box in boxplot["boxes"]:
            box.set(facecolor="#D9D9D9", edgecolor="#A6A6A6", linewidth=0.6, alpha=0.65)
        for element in ("whiskers", "caps", "medians"):
            for artist in boxplot[element]:
                artist.set(color="#7F7F7F", linewidth=0.6)

        for _, subject in subset.groupby("record_id"):
            subject = subject.sort_values("visit")
            if len(subject) < 2:
                continue
            x_values = [positions[str(visit)] for visit in subject["visit"]]
            axis.plot(
                x_values,
                subject["score"],
                color="#9E9E9E",
                marker="o",
                markersize=1.2,
                linewidth=0.55,
                alpha=0.28,
                zorder=1,
            )

        summary = subset.groupby("visit", observed=True)["score"].agg(["mean", "sem"]).reindex(positions)
        axis.errorbar(
            list(positions.values()),
            summary["mean"],
            yerr=1.96 * summary["sem"],
            color="#0072B2",
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            capsize=2,
            zorder=3,
        )
        axis.set_title(assessment)
        axis.set_xticks(list(positions.values()), list(positions))
        axis.set_xlabel("Time Point")
        axis.set_ylabel("Raw score" if assessment in ("FM-LE", "MRS") else "")
        style_axis(axis)

    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.84, wspace=0.42)
    save_figure(figure, output_stem)


def plot_human_baseline_followup(human: pd.DataFrame, output_stem: Path) -> None:
    set_figure_style()
    order = ["FM-LE", "FM-UE", "BI", "MRS", "NIHSS"]
    colors = {"T1": "#E69F00", "T2": "#0072B2"}
    valid = human.loc[human["valid_score"]].copy()
    figure, axes = plt.subplots(1, len(order), figsize=(18 / 2.54, 5.7 / 2.54), dpi=300)

    for axis, assessment in zip(axes, order):
        subset = valid.loc[valid["assessment"] == assessment]
        baseline = subset.loc[subset["visit"] == "T0", ["record_id", "score"]].rename(columns={"score": "baseline"})
        followup = subset.loc[subset["visit"].isin(["T1", "T2"])].merge(baseline, on="record_id", how="inner")
        floor = HUMAN_SCALES[assessment]["floor"]
        ceiling = HUMAN_SCALES[assessment]["ceiling"]
        axis.plot([floor, ceiling], [floor, ceiling], color="#BDBDBD", linestyle="--", linewidth=0.7, zorder=0)

        for visit in ("T1", "T2"):
            values = followup.loc[followup["visit"] == visit]
            axis.scatter(values["baseline"], values["score"], color=colors[visit], s=12, alpha=0.38, edgecolors="none")
            if values["baseline"].nunique() > 1:
                slope, intercept = np.polyfit(values["baseline"], values["score"], 1)
                x_values = np.linspace(values["baseline"].min(), values["baseline"].max(), 100)
                axis.plot(x_values, slope * x_values + intercept, color=colors[visit], linewidth=1.2)

        padding = 0.03 * (ceiling - floor)
        axis.set_xlim(floor - padding, ceiling + padding)
        axis.set_ylim(floor - padding, ceiling + padding)
        axis.set_title(assessment)
        axis.set_xlabel("T0 score")
        axis.set_ylabel("Follow-up score" if assessment in ("FM-LE", "MRS") else "")
        style_axis(axis)

    legend = [
        Line2D([0], [0], color=colors["T1"], marker="o", markersize=3, linewidth=1.2, label="T1"),
        Line2D([0], [0], color=colors["T2"], marker="o", markersize=3, linewidth=1.2, label="T2"),
    ]
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.78, wspace=0.42)
    save_figure(figure, output_stem)


def plot_human_baseline_severity_trajectories(
    human: pd.DataFrame,
    predictions: pd.DataFrame,
    output_stem: Path,
) -> None:
    set_figure_style()
    order = ["FM-LE", "FM-UE", "BI", "MRS", "NIHSS"]
    visits = ["T0", "T1", "T2"]
    positions = {visit: index for index, visit in enumerate(visits)}
    colors = {"Q25 lower initial deficit": "#E69F00", "Q75 higher initial deficit": "#CC79A7"}
    valid = human.loc[human["valid_score"]].copy()
    figure, axes = plt.subplots(1, len(order), figsize=(18 / 2.54, 5.7 / 2.54), dpi=300)

    for axis, assessment in zip(axes, order):
        observed = valid.loc[valid["assessment"] == assessment]
        for _, subject in observed.groupby("record_id"):
            subject = subject.sort_values("visit")
            if len(subject) < 2:
                continue
            axis.plot(
                [positions[str(visit)] for visit in subject["visit"]],
                subject["score"],
                color="#BDBDBD",
                linewidth=0.45,
                alpha=0.2,
                zorder=0,
            )
        assessment_predictions = predictions.loc[predictions["assessment"] == assessment]
        for reference, group in assessment_predictions.groupby("severity_reference"):
            group = group.sort_values("visit")
            yerr = np.vstack(
                [
                    (group["predicted_score"] - group["ci_95_low"]).fillna(0),
                    (group["ci_95_high"] - group["predicted_score"]).fillna(0),
                ]
            )
            axis.errorbar(
                [positions[str(visit)] for visit in group["visit"]],
                group["predicted_score"],
                yerr=yerr,
                color=colors[reference],
                marker="o",
                markersize=2.8,
                linewidth=1.3,
                capsize=2,
                label=reference,
                zorder=3,
            )
        axis.set_title(assessment)
        axis.set_xticks(list(positions.values()), visits)
        axis.set_xlabel("Time Point")
        axis.set_ylabel("Raw score" if assessment in ("FM-LE", "MRS") else "")
        style_axis(axis)

    legend = [Line2D([0], [0], color=color, marker="o", markersize=3, linewidth=1.3, label=label) for label, color in colors.items()]
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.78, wspace=0.42)
    save_figure(figure, output_stem)


def plot_human_baseline_continuous_relationships(
    human: pd.DataFrame,
    predictions: pd.DataFrame,
    output_stem: Path,
) -> None:
    """Plot observations and predictions from the final baseline-moderation models."""
    set_figure_style()
    order = ["FM-LE", "FM-UE", "BI", "MRS", "NIHSS"]
    colors = {"T1": "#E69F00", "T2": "#0072B2"}
    figure, axes = plt.subplots(1, len(order), figsize=(18 / 2.54, 5.7 / 2.54), dpi=300)

    for axis, assessment in zip(axes, order):
        subset = human.loc[(human["assessment"] == assessment) & human["valid_score"]].copy()
        baseline = subset.loc[subset["visit"] == "T0", ["record_id", "deficit_fraction"]].rename(
            columns={"deficit_fraction": "baseline_deficit"}
        )
        followup = subset.loc[subset["visit"].isin(["T1", "T2"]), ["record_id", "visit", "score"]].merge(
            baseline,
            on="record_id",
            how="inner",
        )
        followup["visit"] = followup["visit"].cat.remove_unused_categories()
        assessment_predictions = predictions.loc[predictions["assessment"] == assessment]
        if assessment_predictions.empty:
            axis.set_visible(False)
            continue
        model_name = assessment_predictions["model"].iloc[0]
        binary_mrs = model_name == "binary_logistic_GEE_mRS_0_2"

        for visit in ("T1", "T2"):
            observed = followup.loc[followup["visit"] == visit]
            observed_outcome = observed["score"].le(2).astype(int) if binary_mrs else observed["score"]
            axis.scatter(
                100 * observed["baseline_deficit"],
                observed_outcome,
                color=colors[visit],
                s=7,
                alpha=0.24,
                linewidths=0,
                zorder=1,
            )
            fitted = assessment_predictions.loc[assessment_predictions["visit"] == visit]
            axis.plot(100 * fitted["baseline_deficit"], fitted["predicted_outcome"], color=colors[visit], linewidth=1.4, zorder=3)
            axis.fill_between(
                100 * fitted["baseline_deficit"],
                fitted["ci_95_low"],
                fitted["ci_95_high"],
                color=colors[visit],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )

        short_model = {
            "random_intercept_LMM": "LMM",
            "Gaussian_GEE_fallback": "GEE",
            "binary_logistic_GEE_mRS_0_2": "binary GEE",
        }.get(model_name, model_name)
        display_assessment = "mRS" if assessment == "MRS" else assessment
        axis.set_title(f"{display_assessment}\n{short_model}")
        axis.set_xlabel("")
        axis.set_ylabel("")
        if binary_mrs:
            axis.set_ylim(-0.05, 1.05)
        style_axis(axis)

    legend = [
        Line2D([0], [0], color=colors[visit], marker="o", markersize=3, linewidth=1.4, label=visit)
        for visit in ("T1", "T2")
    ]
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    figure.supxlabel("Continuous T0 deficit (% of scale)", x=0.53, y=0.03)
    figure.supylabel("Model outcome", x=0.01)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.78, wspace=0.42)
    save_figure(figure, output_stem)


def plot_human_boundaries(human: pd.DataFrame, output_stem: Path) -> None:
    set_figure_style()
    order = ["FM-LE", "FM-UE", "BI", "MRS", "NIHSS"]
    visits = ["T0", "T1", "T2"]
    colors = {"floor": "#D55E00", "ceiling": "#0072B2"}
    summary = boundary_table(human, "assessment", "visit", "score", "valid_score")
    figure, axes = plt.subplots(1, len(order), figsize=(18 / 2.54, 5 / 2.54), dpi=300, sharey=True)

    for axis, assessment in zip(axes, order):
        subset = summary.loc[summary["assessment"] == assessment].set_index("visit").reindex(visits)
        x_values = np.arange(len(visits))
        width = 0.34
        axis.bar(x_values - width / 2, subset["pct_at_floor"], width, color=colors["floor"], label="Floor")
        axis.bar(x_values + width / 2, subset["pct_at_ceiling"], width, color=colors["ceiling"], label="Ceiling")
        axis.set_title(assessment)
        axis.set_xticks(x_values, visits)
        axis.set_xlabel("Time Point")
        axis.set_ylabel("Participants (%)" if assessment == "FM-LE" else "")
        style_axis(axis)

    legend = [Patch(facecolor=colors["floor"], label="Scale floor"), Patch(facecolor=colors["ceiling"], label="Scale ceiling")]
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.75, wspace=0.35)
    save_figure(figure, output_stem)


def plot_mouse_trajectories(mouse: pd.DataFrame, output_stem: Path) -> None:
    set_figure_style()
    valid = mouse.loc[mouse["valid_value"]].copy()
    days = sorted(valid["day"].unique())
    colors = {"Sham": "#4D4D4D", "Stroke": "#D55E00"}
    line_styles = {"Sham": "--", "Stroke": "-"}
    figure, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 5 / 2.54), dpi=300)

    for axis, (outcome, metadata) in zip(axes, MOUSE_OUTCOMES.items()):
        subset = valid.loc[valid["outcome"] == outcome]
        summary = subset.groupby(["group", "day"])["value"].agg(["mean", "sem"]).reset_index()
        for group in ("Sham", "Stroke"):
            values = summary.loc[summary["group"] == group]
            axis.errorbar(
                values["day"],
                values["mean"],
                yerr=1.96 * values["sem"],
                color=colors[group],
                linestyle=line_styles[group],
                marker="o",
                markersize=2.5,
                linewidth=1.2,
                capsize=2,
                label=group,
            )
        axis.set_title(metadata["label"])
        axis.set_xticks(days)
        axis.set_xticklabels([str(day) for day in days], rotation=45, ha="right")
        axis.set_xlabel("Day")
        axis.set_ylabel("Raw outcome" if outcome == "C_PawDragPercent" else "")
        style_axis(axis)

    legend = [
        Line2D([0], [0], color=colors[group], linestyle=line_styles[group], marker="o", markersize=3, linewidth=1.2, label=group)
        for group in ("Sham", "Stroke")
    ]
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.74, wspace=0.36)
    save_figure(figure, output_stem)


def plot_cross_species_standardized(summary: pd.DataFrame, output_stem: Path) -> None:
    set_figure_style()
    pairings = ["FM-UE / Paw drag", "FM-UE / Grid walk", "FM-LE / Hindlimb drop"]
    stages = ["Acute", "Early", "Late"]
    positions = {stage: index for index, stage in enumerate(stages)}
    colors = {"Human": "#0072B2", "Mouse": "#D55E00"}
    line_styles = {"Human": "-", "Mouse": "--"}
    figure, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 5 / 2.54), dpi=300, sharey=True)

    for axis, pairing in zip(axes, pairings):
        subset = summary.loc[summary["pairing"] == pairing]
        for species in ("Human", "Mouse"):
            values = subset.loc[subset["species"] == species].set_index("stage").reindex(stages)
            axis.errorbar(
                list(positions.values()),
                values["mean"],
                yerr=1.96 * values["sem"],
                color=colors[species],
                linestyle=line_styles[species],
                marker="o",
                markersize=2.8,
                linewidth=1.3,
                capsize=2,
                label=species,
            )
        axis.axhline(0, color="#BDBDBD", linewidth=0.7, linestyle=":")
        axis.set_title(pairing)
        axis.set_xticks(list(positions.values()), ["Acute\nT0 / D3", "Early\nT1 / D14", "Late\nT2 / D56"])
        axis.set_xlim(-0.12, 2.12)
        axis.tick_params(axis="x", labelsize=7)
        axis.set_xlabel("Recovery stage")
        axis.set_ylabel("Standardized recovery" if pairing == pairings[0] else "")
        style_axis(axis)

    legend = [Line2D([0], [0], color=colors[species], linestyle=line_styles[species], marker="o", markersize=3, linewidth=1.3, label=species) for species in ("Human", "Mouse")]
    figure.legend(handles=legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    figure.subplots_adjust(left=0.08, right=0.96, bottom=0.3, top=0.72, wspace=0.3)
    save_figure(figure, output_stem)


def write_summary(
    output_dir: Path,
    human: pd.DataFrame,
    mouse: pd.DataFrame,
    human_tests: pd.DataFrame,
    mouse_tests: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    lines = [
        "# Longitudinal Reanalysis Summary",
        "",
        "This analysis models raw repeated outcomes and does not use proportional-recovery regressions, change scores as outcomes, k-means severity groups, or Euclidean PRR distances.",
        "",
        "## Data Use",
        "",
        f"- Human: provisional cohort of {human['record_id'].nunique()} records with {int(human['valid_score'].sum())} valid score observations.",
        f"- Mouse: {mouse['animal_id'].nunique()} eligible animals with {int(mouse['valid_value'].sum())} aggregated valid outcome observations.",
        f"- Human out-of-range observations: {int((~human['valid_score']).sum())}.",
        f"- Mouse out-of-range observations: {int((~mouse['valid_value']).sum())}.",
        "",
        "## Models",
        "",
        "- FM-UE, FM-LE, BI, and NIHSS use random-intercept linear mixed models on raw scores, with Gaussian GEE fallback when the random-effects fit is singular.",
        "- mRS attempts ordinal GEE, then prespecified mRS 0-2 versus 3-6 logistic GEE, and finally Gaussian GEE on raw mRS if sparse categories prevent those models.",
        "- Human visit is categorical; no equal spacing between T0, T1, and T2 is assumed.",
        "- Baseline moderation models use only T1/T2 outcomes and continuous T0 severity, avoiding use of T0 as both outcome and covariate.",
        "- Mouse models use all available days and continuous raw outcomes after averaging replicate animal/day observations.",
        "- Benjamini-Hochberg correction is applied separately to prespecified human and mouse test families.",
        "",
        "## Important Limitations",
        "",
        "- Two FM-LE source values exceed the documented maximum of 86: one is in the provisional cohort and one belongs to the excluded T1-only record. Neither is modeled pending source verification.",
        "- Nominal human visits are used because exact days after stroke are not present in the supplied workbooks.",
        "- Gaussian mixed models may not fully represent bounded or zero-heavy outcomes; diagnostics and alternative outcome-specific models remain necessary before publication.",
        "- The GEE analyses for mRS are population-averaged rather than subject-specific; the binary fallback loses ordinal information.",
        "- Mouse replicate rows are averaged because trial-level identifiers are not supplied.",
        "- Stroke type is summarized descriptively but is not included in primary models because the hemorrhage subgroup is small.",
        "",
        "## Multiplicity-Adjusted Tests",
        "",
    ]
    combined_tests = pd.concat(
        [human_tests.assign(dataset="human"), mouse_tests.assign(dataset="mouse")],
        ignore_index=True,
    )
    if combined_tests.empty:
        lines.append("No models completed successfully.")
    else:
        display_columns = ["dataset", "assessment", "analysis", "contrast", "n_subjects", "p_value", "p_value_fdr_bh"]
        lines.extend([combined_tests[display_columns].to_markdown(index=False, floatfmt=".4g"), ""])
    lines.extend(["## Model Failures", ""])
    if failures.empty:
        lines.append("No model failures were recorded.")
    else:
        lines.extend([failures.to_markdown(index=False), ""])
    lines.extend(
        [
            "",
            "These results are exploratory and require statistical review, residual diagnostics, verification of scale coding, and reconciliation with the manuscript before inferential claims are made.",
        ]
    )
    (output_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_excel_workbook(output_path: Path, tables: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def run(root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    human, human_quality = load_human_inputs(root)
    mouse, mouse_quality = load_mouse_input(root)

    analysis_human = human.loc[human["analysis_eligible"]].copy()
    valid_human = analysis_human.loc[analysis_human["valid_score"]].copy()
    valid_mouse = mouse.loc[mouse["valid_value"]].copy()
    quality_checks = pd.concat([human_quality.assign(dataset="human"), mouse_quality.assign(dataset="mouse")], ignore_index=True)
    human_flow, human_assessment_flow = human_subject_flow(human)
    mouse_flow, mouse_assessment_flow = mouse_subject_flow(mouse, mouse_quality)
    human_availability = availability_table(valid_human, "assessment", "visit", "record_id")
    mouse_availability = availability_table(valid_mouse, "outcome", "day", "animal_id")
    human_boundaries = boundary_table(analysis_human, "assessment", "visit", "score", "valid_score")
    mouse_boundaries = boundary_table(mouse, "outcome", "day", "value", "valid_value")
    human_descriptive = descriptive_table(valid_human, ["assessment", "visit"], "score", "record_id")
    mouse_descriptive = descriptive_table(valid_mouse, ["outcome", "group", "day"], "value", "animal_id")
    standardized_responsiveness, paired_srm = human_responsiveness_tables(analysis_human)
    nominal_slopes, slope_correlations = human_nominal_slopes_and_correlations(analysis_human)
    cross_species_summary = cross_species_stage_summary(analysis_human, mouse)
    human_coefficients, human_tests, human_failures, baseline_predictions = run_human_models(analysis_human)
    mouse_coefficients, mouse_tests, mouse_failures = run_mouse_models(mouse)
    failures = pd.concat([human_failures, mouse_failures], ignore_index=True)

    human.to_csv(output_dir / "human_long_all_available.csv", index=False)
    mouse.to_csv(output_dir / "mouse_long_eligible_aggregated.csv", index=False)
    quality_checks.to_csv(output_dir / "input_quality_checks.csv", index=False)
    human_flow.to_csv(output_dir / "human_subject_flow.csv", index=False)
    human_assessment_flow.to_csv(output_dir / "human_assessment_flow.csv", index=False)
    mouse_flow.to_csv(output_dir / "mouse_subject_flow.csv", index=False)
    mouse_assessment_flow.to_csv(output_dir / "mouse_assessment_flow.csv", index=False)
    human_availability.to_csv(output_dir / "human_availability.csv", index=False)
    mouse_availability.to_csv(output_dir / "mouse_availability.csv", index=False)
    human_boundaries.to_csv(output_dir / "human_boundary_summary.csv", index=False)
    mouse_boundaries.to_csv(output_dir / "mouse_boundary_summary.csv", index=False)
    human_descriptive.to_csv(output_dir / "human_descriptive_mean_sd.csv", index=False)
    mouse_descriptive.to_csv(output_dir / "mouse_descriptive_mean_sd.csv", index=False)
    standardized_responsiveness.to_csv(output_dir / "human_standardized_responsiveness.csv", index=False)
    paired_srm.to_csv(output_dir / "human_paired_standardized_response.csv", index=False)
    nominal_slopes.to_csv(output_dir / "human_nominal_visit_slopes.csv", index=False)
    slope_correlations.to_csv(output_dir / "human_slope_correlations.csv", index=False)
    baseline_predictions.to_csv(output_dir / "human_baseline_continuous_predictions.csv", index=False)
    cross_species_summary.to_csv(output_dir / "cross_species_stage_summary.csv", index=False)
    human.loc[~human["valid_score"]].to_csv(output_dir / "human_range_violations.csv", index=False)
    mouse.loc[~mouse["valid_value"]].to_csv(output_dir / "mouse_range_violations.csv", index=False)
    analysis_human.groupby(["stroke_category", "stroke_type_raw"], dropna=False)["record_id"].nunique().reset_index(name="n_subjects").to_csv(output_dir / "human_stroke_counts.csv", index=False)
    mouse.groupby(["group", "stroke_type"], dropna=False)["animal_id"].nunique().reset_index(name="n_animals").to_csv(output_dir / "mouse_group_counts.csv", index=False)

    human_coefficients.to_csv(output_dir / "human_model_coefficients.csv", index=False)
    human_tests.to_csv(output_dir / "human_model_tests.csv", index=False)
    mouse_coefficients.to_csv(output_dir / "mouse_model_coefficients.csv", index=False)
    mouse_tests.to_csv(output_dir / "mouse_model_tests.csv", index=False)
    failures.to_csv(output_dir / "model_failures.csv", index=False)
    sensitivity_names = [
        "exclude_baseline_boundaries",
        "exclude_boundary_observations",
        "baseline_moderation_exclude_baseline_boundaries",
        "baseline_moderation_exclude_boundary_observations",
    ]
    boundary_sensitivity = human_coefficients.loc[human_coefficients["analysis"].isin(sensitivity_names)]
    boundary_sensitivity_tests = human_tests.loc[human_tests["analysis"].isin(sensitivity_names)]
    boundary_sensitivity.to_csv(output_dir / "human_boundary_sensitivity_coefficients.csv", index=False)
    boundary_sensitivity_tests.to_csv(output_dir / "human_boundary_sensitivity_tests.csv", index=False)
    pd.DataFrame(
        [{"assessment": assessment, **metadata} for assessment, metadata in HUMAN_SCALES.items()]
    ).to_csv(output_dir / "human_scale_definitions.csv", index=False)

    tables = {
        "quality": quality_checks,
        "human_subject_flow": human_flow,
        "human_assessment_flow": human_assessment_flow,
        "mouse_subject_flow": mouse_flow,
        "mouse_assessment_flow": mouse_assessment_flow,
        "human_availability": human_availability,
        "mouse_availability": mouse_availability,
        "human_mean_sd": human_descriptive,
        "mouse_mean_sd": mouse_descriptive,
        "human_boundaries": human_boundaries,
        "mouse_boundaries": mouse_boundaries,
        "standardized_response": standardized_responsiveness,
        "paired_srm": paired_srm,
        "baseline_predictions": baseline_predictions,
        "nominal_slopes": nominal_slopes,
        "slope_correlations": slope_correlations,
        "cross_species": cross_species_summary,
        "human_model_tests": human_tests,
        "human_coefficients": human_coefficients,
        "boundary_sensitivity": boundary_sensitivity,
        "boundary_sens_tests": boundary_sensitivity_tests,
        "mouse_model_tests": mouse_tests,
        "mouse_coefficients": mouse_coefficients,
        "model_failures": failures,
    }
    write_excel_workbook(output_dir / "reanalysis_statistical_tables.xlsx", tables)

    plot_human_trajectories(analysis_human, output_dir / "human_raw_score_trajectories")
    plot_human_baseline_followup(analysis_human, output_dir / "human_baseline_followup")
    plot_human_baseline_continuous_relationships(analysis_human, baseline_predictions, output_dir / "human_baseline_continuous_relationships")
    plot_human_boundaries(analysis_human, output_dir / "human_floor_ceiling")
    plot_mouse_trajectories(mouse, output_dir / "mouse_raw_score_trajectories")
    plot_cross_species_standardized(cross_species_summary, output_dir / "cross_species_standardized_trajectories")
    write_summary(output_dir, analysis_human, mouse, human_tests, mouse_tests, failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: output/reanalysis_reviewer1)")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = arguments.output_dir or root / "output" / "reanalysis_reviewer1"
    run(root, output_dir.resolve())
    print(f"Reanalysis outputs written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
