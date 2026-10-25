from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_cv_predictions_2019.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = (
    REPORT_DIR
    / "baseline_probability_diagnostics_2019.txt"
)

df = pd.read_csv(PREDICTIONS_PATH)

target = "made_mlb_by_2024"

model_columns = [
    "draft_pick_only",
    "draft_pick_plus_position",
    "draft_pick_plus_college",
]

report = []

report.append("2019 BASELINE PROBABILITY DIAGNOSTICS")
report.append("=" * 40)

for model_column in model_columns:
    bucket_column = f"{model_column}_bucket"

    df[bucket_column] = pd.qcut(
        df[model_column],
        q=5,
        labels=False,
        duplicates="drop",
    ) + 1

    calibration = (
        df.groupby(bucket_column)
        .agg(
            players=("player_name", "size"),
            average_predicted_probability=(
                model_column,
                "mean",
            ),
            actual_mlb_rate=(
                target,
                "mean",
            ),
        )
        .reset_index()
        .rename(
            columns={
                bucket_column: "probability_quintile"
            }
        )
    )

    report.append(f"\n{model_column.upper()}")
    report.append("-" * 40)
    report.append(
        calibration.to_string(
            index=False,
            formatters={
                "average_predicted_probability": (
                    lambda value: f"{value:.1%}"
                ),
                "actual_mlb_rate": (
                    lambda value: f"{value:.1%}"
                ),
            },
        )
    )

best_model = "draft_pick_only"

report.append("\n\nHIGHEST DRAFT-PICK MODEL PROBABILITIES")
report.append("-" * 40)

top_players = (
    df.sort_values(best_model, ascending=False)
    .head(10)
    [
        [
            "player_name",
            "overall_pick",
            "position_family",
            target,
            best_model,
        ]
    ]
)

report.append(
    top_players.to_string(
        index=False,
        formatters={
            best_model: lambda value: f"{value:.1%}"
        },
    )
)

report.append("\n\nLOWEST DRAFT-PICK MODEL PROBABILITIES")
report.append("-" * 40)

bottom_players = (
    df.sort_values(best_model, ascending=True)
    .head(10)
    [
        [
            "player_name",
            "overall_pick",
            "position_family",
            target,
            best_model,
        ]
    ]
)

report.append(
    bottom_players.to_string(
        index=False,
        formatters={
            best_model: lambda value: f"{value:.1%}"
        },
    )
)

REPORT_DIR.mkdir(exist_ok=True)
REPORT_PATH.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print("\n".join(report))
print(f"\nSaved report to: {REPORT_PATH}")