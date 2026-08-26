"""Create a Word document containing the reanalysis SVG figures and captions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import win32com.client


FIGURES = [
    (
        "human_raw_score_trajectories.svg",
        2,
        "Transition from static change scores to truly longitudinal models. The "
        "authors should implement linear or generalized linear mixed-effects models "
        "to analyze raw scores across T0, T1, and T2.",
        "The Time x Baseline interaction is still missing from the figure. Show the "
        "trajectory for participants with lower and higher initial deficits and state "
        "which assessment is most sensitive to these differences. Consider nonlinear "
        "time effects or splines and use actual days after stroke if available.",
        "Figure 2 intentionally shows the observed raw trajectories. The requested "
        "Time x Baseline result is now shown separately in Figure 4 using predictions "
        "from models that retain baseline deficit continuously. Human time remains "
        "categorical because the source files provide only T0, T1, and T2, not exact "
        "assessment days; three nominal visits do not support a reliable spline model.",
        "Longitudinal trajectories of clinical outcomes after stroke.",
        "Raw scores are shown for the Fugl-Meyer lower-extremity assessment "
        "(FM-LE), Fugl-Meyer upper-extremity assessment (FM-UE), Barthel Index "
        "(BI), modified Rankin Scale (mRS), and National Institutes of Health "
        "Stroke Scale (NIHSS) at the acute baseline (T0), early rehabilitation "
        "assessment (T1), and six-month follow-up (T2). Gray lines connect "
        "available observations from individual participants, gray boxplots show "
        "the median and interquartile range, and blue lines show the group mean "
        "with approximate 95% confidence intervals (mean +/- 1.96 SEM). Higher FM "
        "and BI scores indicate better performance, whereas lower mRS and NIHSS "
        "scores indicate less disability or neurological impairment. All available "
        "observations were included; complete data at all three visits were not required.",
        "Replace existing Figure 2. The new figure retains the cross-assessment "
        "trajectory comparison but removes recovery-pattern classification and uses "
        "all available raw longitudinal observations.",
    ),
    (
        "human_floor_ceiling.svg",
        3,
        "Account for scale ceiling effects and compression. Report the exact "
        "proportions of participants exhibiting floor or ceiling effects at T0, T1, "
        "and T2 across all clinical scales and conduct sensitivity analyses.",
        "It is unclear whether observations at a floor or ceiling should be corrected "
        "or excluded. Show how the results change when these observations are handled "
        "differently and report the corresponding statistics.",
        "Boundary values are valid scale scores and were not corrected. The primary "
        "models retain them. Two diagnostic analyses exclude participants already at "
        "a boundary at T0 or exclude individual boundary observations; their model "
        "coefficients and tests are supplied in the statistical workbook. Two FM-LE "
        "values above the documented maximum are reported separately and excluded from models.",
        "Floor and ceiling effects across clinical assessments and visits.",
        "Bars show the percentage of available observations located exactly at the "
        "theoretical floor (orange) or ceiling (blue) of each assessment at T0, T1, "
        "and T2. The applied scale ranges were 0-86 for FM-LE, 0-126 for FM-UE, "
        "0-100 for BI, 0-6 for mRS, and 0-42 for NIHSS. Increasing ceiling effects "
        "were observed at T2 for FM-LE, FM-UE, and BI, while NIHSS showed a "
        "prominent floor effect. For NIHSS and mRS, the floor represents better "
        "outcome; for FM and BI, the ceiling represents better outcome. These "
        "boundary effects indicate reduced ability of some scales to differentiate "
        "participants with mild residual deficits at later follow-up.",
        "Replace existing Figure 3. The k-means clustering figure should be removed; "
        "the replacement directly addresses scale-boundary compression across the "
        "five clinical assessments.",
    ),
    (
        "human_baseline_continuous_relationships_prototype.svg",
        4,
        "Avoid mathematical coupling in change-score models. The authors should "
        "abandon change scores (Y - X) as dependent variables and instead model raw "
        "post-stroke outcomes (Y) directly with baseline score (X) as a covariate. "
        "Baseline severity should remain continuous, and a Time x Baseline interaction "
        "should test whether its association with outcome changes over time.",
        "The Q25 and Q75 trajectories appear to be two patient groups, and it is unclear "
        "whether the T1 and T2 points are follow-up percentiles or model predictions. "
        "A direct display of the continuous baseline relationship would make the tested "
        "Time x Baseline interaction easier to understand.",
        "Figure 4 was redesigned to display every observed T1 and T2 score against the "
        "full continuous T0-deficit range. Separate fitted T1 and T2 relationships now "
        "show the interaction directly as a difference in slopes. No percentile cutoff, "
        "severity group, change score, or PRR quantity is used.",
        "Continuous initial deficit and raw follow-up outcomes.",
        "Raw T1 (orange) and T2 (blue) scores are plotted against continuous T0 deficit, "
        "expressed as a percentage of each assessment's theoretical range and oriented "
        "so that larger values indicate worse initial status. Solid lines show the "
        "population-average fitted relationship at each visit, and shaded bands show "
        "approximate 95% confidence intervals for the estimated mean. The Time x Baseline "
        "interaction tests whether the T1 and T2 slopes differ. Thus, a significant "
        "interaction indicates that the association between initial deficit and outcome "
        "changes between visits; it does not test two baseline-severity groups. This "
        "prototype uses Gaussian GEE for a common visual form across assessments; the "
        "final inferential panel should use predictions from the same outcome-specific "
        "models as the accompanying statistical tests.",
        "Replace existing Figure 4. The PRR and best-fit change-score panels should "
        "be removed and replaced by the direct continuous T0-deficit versus raw T1/T2 "
        "outcome relationships.",
    ),
    (
        "cross_species_standardized_trajectories.svg",
        5,
        "Apply uncoupled longitudinal mixed-effects models to continuous raw rodent "
        "performance data, reframing the translational section around cross-species "
        "trajectory comparison rather than proportional recovery.",
        "Keep the full mouse time course, but also show a direct human-mouse comparison "
        "in the same format. Select biologically meaningful mouse stages corresponding "
        "to T0, T1, and T2, explain the selected days, and standardize the outcomes. "
        "Clarify whether observations are matched pairs and do not imply equivalence "
        "between human assessments and mouse tasks.",
        "All mouse days remain in the raw-trajectory output and longitudinal models. "
        "For the conceptual comparison, acute, early, and late landmarks are T0/T1/T2 "
        "for humans and days 3/14/56 for mice. The species contain unrelated subjects, "
        "so these are not matched pairs. Standardization aligns recovery direction and "
        "acute-stage variability but does not make the instruments or times biologically equivalent.",
        "Standardized recovery trajectories across human and experimental stroke.",
        "Human assessments and conceptually related mouse behavioral outcomes are shown "
        "at acute (T0/day 3), early (T1/day 14), and late (T2/day 56) recovery stages. "
        "Scores are oriented so that positive values indicate recovery and standardized "
        "relative to acute-stage variability within each outcome. Points show means and "
        "error bars show approximate 95% confidence intervals. Human and mouse subjects "
        "are independent and are not individually matched. The pairings are conceptual "
        "comparisons of motor constructs; they do not assert measurement or temporal equivalence. "
        "Raw mouse trajectories across days 0, 3, 7, 14, 21, 28, 42, and 56 are supplied separately.",
        "Replace existing Figure 5. The mouse PRR and clustered change-score panels "
        "should be removed. Use the standardized cross-species panel in the main text "
        "and retain the complete raw sham-versus-stroke mouse trajectories as a companion figure.",
    ),
]

TABLES = [
    (
        "human_descriptive_mean_sd.csv",
        "Table 1. Human outcomes by assessment and visit",
        ["assessment", "visit", "n_observations", "mean", "sd"],
        ["Assessment", "Visit", "n", "Mean", "SD"],
    ),
    (
        "mouse_descriptive_mean_sd.csv",
        "Table 2. Mouse outcomes by task, group, and day",
        ["outcome", "group", "day", "n_observations", "mean", "sd"],
        ["Outcome", "Group", "Day", "n", "Mean", "SD"],
    ),
    (
        "human_model_tests.csv",
        "Table 3. Human longitudinal model tests",
        ["assessment", "analysis", "statistic", "df", "p_value", "p_value_fdr_bh"],
        ["Assessment", "Analysis", "Statistic", "df", "p", "FDR p"],
    ),
    (
        "mouse_model_tests.csv",
        "Table 4. Mouse longitudinal model tests",
        ["assessment", "analysis", "statistic", "df", "p_value", "p_value_fdr_bh"],
        ["Outcome", "Analysis", "Statistic", "df", "p", "FDR p"],
    ),
    (
        "human_boundary_summary.csv",
        "Table 5. Human scale-boundary observations",
        ["assessment", "visit", "n", "n_at_floor", "pct_at_floor", "n_at_ceiling", "pct_at_ceiling"],
        ["Assessment", "Visit", "n", "Floor n", "Floor %", "Ceiling n", "Ceiling %"],
    ),
]

POINTS_PER_CM = 72 / 2.54
WD_COLOR_AUTOMATIC = -16777216
WD_COLOR_RED = 255

FIGURE_4_CHANGE_NOTE = (
    "The previous version evaluated one continuous model at the 25th and 75th "
    "percentiles of baseline deficit. Statistically, these were only two reference "
    "inputs, not patient groups. Graphically, however, two colored trajectories labeled "
    "Q25 and Q75 naturally resembled categorized cohorts. Connecting observed T0 anchors "
    "to predicted T1 and T2 values could also suggest that T0 was modeled as an outcome, "
    "and the follow-up points could be mistaken for follow-up percentiles. The revised "
    "figure removes these understandable ambiguities by showing all observed follow-up "
    "values over the complete continuous T0-deficit range. The difference between the "
    "fitted T1 and T2 slopes now corresponds directly to the reviewer's requested "
    "Time x Baseline interaction."
)


def format_table_value(value: object, column: str) -> str:
    if pd.isna(value):
        return ""
    if column in {"p_value", "p_value_fdr_bh"}:
        number = float(value)
        return "<0.001" if number < 0.001 else f"{number:.3f}"
    if column in {"mean", "sd", "statistic", "pct_at_floor", "pct_at_ceiling"}:
        return f"{float(value):.2f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def insert_table(selection, document, dataframe, title, columns, labels) -> None:
    selection.InsertBreak(7)
    selection.ParagraphFormat.Alignment = 0
    selection.Font.Name = "Calibri"
    selection.Font.Size = 11
    selection.Font.Bold = True
    selection.Font.Color = WD_COLOR_AUTOMATIC
    selection.TypeText(title)
    selection.TypeParagraph()

    table = document.Tables.Add(selection.Range, len(dataframe) + 1, len(columns))
    table.Borders.Enable = True
    table.Range.Font.Name = "Calibri"
    table.Range.Font.Size = 8
    for column_index, label in enumerate(labels, start=1):
        cell = table.Cell(1, column_index)
        cell.Range.Text = label
        cell.Range.Font.Bold = True
    for row_index, (_, row) in enumerate(dataframe.iterrows(), start=2):
        for column_index, column in enumerate(columns, start=1):
            table.Cell(row_index, column_index).Range.Text = format_table_value(row[column], column)
    table.AutoFitBehavior(2)
    selection.SetRange(table.Range.End, table.Range.End)
    selection.TypeParagraph()


def create_document(figure_dir: Path, output_path: Path) -> None:
    missing = [item[0] for item in FIGURES + TABLES if not (figure_dir / item[0]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing SVG figures: {', '.join(missing)}")

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Add()
        section = document.Sections(1)
        section.PageSetup.TopMargin = 1.5 * POINTS_PER_CM
        section.PageSetup.BottomMargin = 1.5 * POINTS_PER_CM
        section.PageSetup.LeftMargin = 1.5 * POINTS_PER_CM
        section.PageSetup.RightMargin = 1.5 * POINTS_PER_CM

        selection = word.Selection
        selection.Font.Name = "Calibri"
        selection.Font.Size = 14
        selection.Font.Bold = True
        selection.ParagraphFormat.Alignment = 1
        selection.TypeText("Proposed Replacement Figures and Statistical Tables")
        selection.TypeParagraph()
        selection.TypeParagraph()

        for index, (
            filename,
            figure_number,
            reviewer_comment,
            supervisor_comment,
            response,
            title,
            caption,
            placement,
        ) in enumerate(FIGURES):
            selection.ParagraphFormat.Alignment = 0
            selection.ParagraphFormat.SpaceBefore = 0
            selection.ParagraphFormat.SpaceAfter = 3
            selection.Font.Name = "Calibri"
            selection.Font.Size = 12
            selection.Font.Bold = True
            selection.Font.Italic = False
            selection.TypeText(f"Proposed Figure {figure_number}")
            selection.TypeParagraph()

            selection.ParagraphFormat.Alignment = 0
            selection.ParagraphFormat.SpaceBefore = 0
            selection.ParagraphFormat.SpaceAfter = 3
            selection.Font.Name = "Calibri"
            selection.Font.Size = 10
            selection.Font.Bold = True
            selection.Font.Italic = False
            selection.TypeText("Reviewer 1 comment: ")
            selection.Font.Bold = False
            selection.Font.Italic = True
            selection.TypeText(reviewer_comment)
            selection.Font.Italic = False
            selection.TypeParagraph()

            selection.Font.Color = WD_COLOR_RED
            selection.Font.Bold = True
            selection.TypeText("Supervisor comment (Markus): ")
            selection.Font.Bold = False
            selection.Font.Italic = True
            selection.TypeText(supervisor_comment)
            selection.Font.Italic = False
            selection.TypeParagraph()

            selection.Font.Color = WD_COLOR_AUTOMATIC
            selection.Font.Bold = True
            selection.TypeText("Analysis response: ")
            selection.Font.Bold = False
            selection.TypeText(response)
            selection.TypeParagraph()

            selection.ParagraphFormat.Alignment = 1
            shape = selection.InlineShapes.AddPicture(
                FileName=str((figure_dir / filename).resolve()),
                LinkToFile=False,
                SaveWithDocument=True,
            )
            shape.LockAspectRatio = True
            shape.Width = 18 * POINTS_PER_CM
            selection.TypeParagraph()

            selection.ParagraphFormat.Alignment = 0
            selection.ParagraphFormat.SpaceBefore = 6
            selection.ParagraphFormat.SpaceAfter = 0
            selection.Font.Name = "Calibri"
            selection.Font.Size = 10
            selection.Font.Color = WD_COLOR_AUTOMATIC
            selection.Font.Bold = True
            selection.TypeText(f"Figure {figure_number}. {title} ")
            selection.Font.Bold = False
            selection.TypeText(caption)
            selection.TypeParagraph()

            if figure_number == 4:
                selection.ParagraphFormat.SpaceBefore = 6
                selection.Font.Bold = True
                selection.TypeText("Why Figure 4 was changed: ")
                selection.Font.Bold = False
                selection.TypeText(FIGURE_4_CHANGE_NOTE)
                selection.TypeParagraph()

            selection.ParagraphFormat.SpaceBefore = 6
            selection.Font.Bold = True
            selection.TypeText("Recommended manuscript placement: ")
            selection.Font.Bold = False
            selection.TypeText(placement)

            if index < len(FIGURES) - 1:
                selection.InsertBreak(7)

        selection.TypeParagraph()
        selection.Font.Bold = True
        selection.TypeText("Complete statistical output: ")
        selection.Font.Bold = False
        selection.TypeText(
            "reanalysis_statistical_tables.xlsx contains complete descriptive, model, "
            "sensitivity, standardized-response, prediction, and cross-species tables."
        )

        for filename, title, columns, labels in TABLES:
            dataframe = pd.read_csv(figure_dir / filename)
            insert_table(selection, document, dataframe, title, columns, labels)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        document.SaveAs2(str(output_path.resolve()), FileFormat=16)
    finally:
        if document is not None:
            document.Close(SaveChanges=False)
        word.Quit()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    figure_dir = root / "output" / "reanalysis_reviewer1"
    output_path = figure_dir / "longitudinal_reanalysis_figures_continuous_baseline_revision.docx"
    create_document(figure_dir, output_path)
    print(f"Word document written to {output_path}")


if __name__ == "__main__":
    main()
