from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_2019.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "outcome_exploration_2019.txt"

df = pd.read_csv(DATA_PATH)

df["made_mlb_ever"] = pd.to_numeric(
    df["made_mlb_ever"]
)

df["overall_pick"] = pd.to_numeric(
    df["overall_pick"]
)

df["college_PA"] = pd.to_numeric(
    df["college_PA"]
)

df["college_OPS"] = pd.to_numeric(
    df["college_OPS"]
)

df["position_family"] = df["position"]

df.loc[
    df["position"].isin(["CF", "LF", "RF", "OF"]),
    "position_family",
] = "OF"

df.loc[
    df["position"].isin(["1B", "2B", "3B", "SS", "IF"]),
    "position_family",
] = "IF"

df["pick_bin"] = pd.cut(
    df["overall_pick"],
    bins=[0, 32, 64, 100, 150, 200, 300, 400, 500, 700],
    labels=[
        "1-32",
        "33-64",
        "65-100",
        "101-150",
        "151-200",
        "201-300",
        "301-400",
        "401-500",
        "501-700",
    ],
)

def outcome_table(group_column):
    return (
        df.groupby(group_column, dropna=False)
        .agg(
            players=("player_name", "size"),
            mlb_players=("made_mlb_ever", "sum"),
            mlb_rate=("made_mlb_ever", "mean"),
            average_PA=("college_PA", "mean"),
            median_OPS=("college_OPS", "median"),
        )
        .reset_index()
    )

by_round = outcome_table("draft_round")
by_position = outcome_table("position_family")
by_pick_bin = outcome_table("pick_bin")

by_outcome = (
    df.groupby("made_mlb_ever")
    .agg(
        players=("player_name", "size"),
        average_PA=("college_PA", "mean"),
        median_PA=("college_PA", "median"),
        average_BA=("college_BA", "mean"),
        average_OBP=("college_OBP", "mean"),
        average_SLG=("college_SLG", "mean"),
        average_OPS=("college_OPS", "mean"),
        average_HR=("college_HR", "mean"),
        average_BB=("college_BB", "mean"),
        average_SO=("college_SO", "mean"),
        average_SB=("college_SB", "mean"),
    )
    .reset_index()
)

low_pa = df[
    df["college_PA"] < 100
][
    [
        "player_name",
        "position",
        "overall_pick",
        "college_PA",
        "college_OPS",
        "made_mlb_ever",
    ]
].sort_values("college_PA")

def format_table(table):
    return table.to_string(
        index=False,
        formatters={
            "mlb_rate": lambda value: f"{value:.1%}",
            "average_PA": lambda value: f"{value:.1f}",
            "median_OPS": lambda value: f"{value:.3f}",
        },
    )

report = []

report.append("2019 MLB OUTCOME EXPLORATION")
report.append("=" *  thirty if False else "=" * 30)
report.append("")
report.append(f"Total players: {len(df)}")
report.append(
    f"Players who reached MLB by check date: "
    f"{int(df['made_mlb_ever'].sum())}"
)
report.append(
    f"Overall reach rate: "
    f"{df['made_mlb_ever'].mean():.1%}"
)

report.append("\nOUTCOMES BY DRAFT ROUND")
report.append("-" * 30)
report.append(format_table(by_round))

report.append("\nOUTCOMES BY POSITION FAMILY")
report.append("-" * 30)
report.append(format_table(by_position))

report.append("\nOUTCOMES BY OVERALL PICK BIN")
report.append("-" * 30)
report.append(format_table(by_pick_bin))

report.append("\nCOLLEGE STATISTICS BY OUTCOME")
report.append("-" * 30)
report.append(
    by_outcome.to_string(
        index=False,
        formatters={
            "average_PA": lambda value: f"{value:.1f}",
            "median_PA": lambda value: f"{value:.1f}",
            "average_BA": lambda value: f"{value:.3f}",
            "average_OBP": lambda value: f"{value:.3f}",
            "average_SLG": lambda value: f"{value:.3f}",
            "average_OPS": lambda value: f"{value:.3f}",
            "average_HR": lambda value: f"{value:.1f}",
            "average_BB": lambda value: f"{value:.1f}",
            "average_SO": lambda value: f"{value:.1f}",
            "average_SB": lambda value: f"{value:.1f}",
        },
    )
)

report.append("\nPLAYERS WITH FEWER THAN 100 COLLEGE PA")
report.append("-" * 30)
report.append(low_pa.to_string(index=False))

REPORT_DIR.mkdir(exist_ok=True)
REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"\nSaved report to: {REPORT_PATH}")