from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import longitudinal_reanalysis as analysis


def test_human_preparation_retains_available_observations_and_flags_ranges():
    human, quality = analysis.load_human_inputs(ROOT)

    assert not human.duplicated(["record_id", "assessment", "visit"]).any()
    assert set(human["visit"].dropna().astype(str)) == {"T0", "T1", "T2"}
    assert human.loc[~human["valid_score"], "assessment"].tolist() == ["FM-LE", "FM-LE"]
    assert set(human.loc[~human["valid_score"], "score"]) == {95.0, 96.0}
    assert quality.loc[quality["check"] == "Out-of-range human scores", "value"].item() == 2


def test_human_analysis_cohort_is_auditable_from_t0_and_metadata():
    human, quality = analysis.load_human_inputs(ROOT)

    eligible_ids = set(human.loc[human["analysis_eligible"], "record_id"])
    excluded_ids = set(human.loc[~human["analysis_eligible"], "record_id"])
    assert len(eligible_ids) == 120
    assert excluded_ids == {121}
    assert quality.loc[quality["check"] == "Provisional human analysis cohort", "value"].item() == 120


def test_documented_human_scale_ranges_are_explicit():
    assert analysis.HUMAN_SCALES["FM-UE"]["ceiling"] == 126
    assert analysis.HUMAN_SCALES["FM-LE"]["ceiling"] == 86
    assert analysis.HUMAN_SCALES["MRS"]["ceiling"] == 6
    assert all(scale["definition"] for scale in analysis.HUMAN_SCALES.values())


def test_mrs_uses_full_standard_range():
    human, _ = analysis.load_human_inputs(ROOT)

    mrs = human.loc[human["assessment"] == "MRS"]
    assert mrs["ceiling"].eq(6).all()
    assert mrs["valid_score"].all()


def test_stroke_classification_does_not_default_unknown_values_to_infarct():
    assert analysis.classify_stroke("Bihemisphaerischeinfakrte") == "Ischemic"
    assert analysis.classify_stroke("ICB") == "Hemorrhage"
    assert analysis.classify_stroke(None) == "Unknown"
    assert analysis.classify_stroke("unrecognized diagnosis") == "Other/unknown"


def test_mouse_preparation_aggregates_animal_day_replicates():
    mouse, quality = analysis.load_mouse_input(ROOT)

    wide_keys = mouse[["animal_id", "day"]].drop_duplicates()
    aggregated_rows = quality.loc[quality["check"] == "Aggregated animal/day rows", "value"].item()
    assert len(wide_keys) == aggregated_rows
    assert set(mouse["day"].unique()) == {0, 3, 7, 14, 21, 28, 42, 56}
    assert set(mouse["group"].unique()) == {"Sham", "Stroke"}
