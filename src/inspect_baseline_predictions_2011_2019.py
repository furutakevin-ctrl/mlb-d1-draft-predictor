"""Summarize pooled out-of-fold prediction calibration and ranked players."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_cv_predictions_2011_2019.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "baseline_probability_diagnostics_2011_2019.txt"

df = pd.read_csv(PREDICTIONS_PATH)
target = "made_mlb_ever"
model_columns = ["draft_pick_only", "draft_pick_plus_position", "draft_pick_plus_college"]
required = {"player_name", "draft_year", "overall_pick", "position_family", target, *model_columns}
missing = sorted(required - set(df.columns))
if missing:
    raise ValueError(f"Prediction file is missing columns: {missing}")

report = ["2011-2019 BASELINE PROBABILITY DIAGNOSTICS", "=" * 44]
for model_column in model_columns:
    bucket_column = f"{model_column}_bucket"
    df[bucket_column] = pd.qcut(df[model_column], q=5, labels=False, duplicates="drop") + 1
    calibration = (
        df.groupby(bucket_column, observed=False)
        .agg(
            players=("player_name", "size"),
            average_predicted_probability=(model_column, "mean"),
            actual_mlb_rate=(target, "mean"),
        )
        .reset_index()
        .rename(columns={bucket_column: "probability_quintile"})
    )
    report.extend(
        [
            f"\n{model_column.upper()}",
            "-" * 44,
            calibration.to_string(
                index=False,
                formatters={
                    "average_predicted_probability": lambda value: f"{value:.1%}",
                    "actual_mlb_rate": lambda value: f"{value:.1%}",
                },
            ),
        ]
    )

best_model = "draft_pick_only"
year_calibration = (
    df.groupby("draft_year")
    .agg(
        players=("player_name", "size"),
        positive_outcomes=(target, "sum"),
        average_predicted_probability=(best_model, "mean"),
        actual_mlb_rate=(target, "mean"),
    )
    .reset_index()
)
report.extend(
    [
        "\n\nDRAFT-PICK MODEL CALIBRATION BY YEAR",
        "-" * 44,
        year_calibration.to_string(
            index=False,
            formatters={
                "average_predicted_probability": lambda value: f"{value:.1%}",
                "actual_mlb_rate": lambda value: f"{value:.1%}",
            },
        ),
        "\n\nHIGHEST DRAFT-PICK MODEL PROBABILITIES",
        "-" * 44,
    ]
)

columns = ["draft_year", "player_name", "overall_pick", "position_family", target, best_model]
top = df.sort_values(best_model, ascending=False).head(10)[columns]
report.append(top.to_string(index=False, formatters={best_model: lambda value: f"{value:.1%}"}))
report.extend(["\n\nLOWEST DRAFT-PICK MODEL PROBABILITIES", "-" * 44])
bottom = df.sort_values(best_model, ascending=True).head(10)[columns]
report.append(bottom.to_string(index=False, formatters={best_model: lambda value: f"{value:.1%}"}))

REPORT_DIR.mkdir(exist_ok=True)
output = "\n".join(report)
REPORT_PATH.write_text(output, encoding="utf-8")
print(output)
print(f"\nSaved report to: {REPORT_PATH}")
