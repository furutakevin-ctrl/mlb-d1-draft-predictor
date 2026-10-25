from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "players_2019_master.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "exploration_2019.txt"

df = pd.read_csv(INPUT_PATH)

numeric_columns = [
    "overall_pick",
    "college_PA",
    "college_BA",
    "college_OBP",
    "college_SLG",
    "college_OPS",
    "college_BB_rate",
    "college_K_rate",
    "college_HR_rate",
    "college_ISO",
    "college_XBH_rate",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="raise")

report = []


def add_section(title, content):
    report.append("")
    report.append(title)
    report.append("-" * len(title))
    report.append(str(content))


add_section(
    "Dataset overview",
    {
        "players": len(df),
        "unique_players": df["player_key"].nunique(),
        "draft_years": sorted(df["draft_year"].unique().tolist()),
    },
)

add_section(
    "Position counts",
    df["position"].value_counts().to_string(),
)

add_section(
    "Draft round counts",
    df["draft_round"].value_counts().sort_index().to_string(),
)

add_section(
    "Draft-pick summary",
    df["overall_pick"].describe().round(2).to_string(),
)

add_section(
    "College PA summary",
    df["college_PA"].describe().round(2).to_string(),
)

add_section(
    "Players below PA thresholds",
    {
        "under_100_PA": int((df["college_PA"] < 100).sum()),
        "under_150_PA": int((df["college_PA"] < 150).sum()),
    },
)

position_summary = (
    df.groupby("position")
    .agg(
        players=("player_key", "count"),
        median_PA=("college_PA", "median"),
        median_OPS=("college_OPS", "median"),
        median_BB_rate=("college_BB_rate", "median"),
        median_K_rate=("college_K_rate", "median"),
        median_HR_rate=("college_HR_rate", "median"),
    )
    .sort_values("players", ascending=False)
    .round(3)
)

add_section(
    "Position summary",
    position_summary.to_string(),
)

correlation_columns = [
    "overall_pick",
    "college_OPS",
    "college_OBP",
    "college_SLG",
    "college_BA",
    "college_BB_rate",
    "college_K_rate",
    "college_HR_rate",
    "college_ISO",
    "college_XBH_rate",
]

spearman_correlations = (
    df[correlation_columns]
    .corr(method="spearman")["overall_pick"]
    .drop("overall_pick")
    .sort_values()
    .round(3)
)

add_section(
    "Spearman correlations with overall draft pick",
    spearman_correlations.to_string(),
)

top_players = (
    df[df["college_PA"] >= 100]
    .nlargest(10, "college_OPS")[
        [
            "player_name",
            "position",
            "overall_pick",
            "college_PA",
            "college_OPS",
        ]
    ]
)

add_section(
    "Top OPS performers with at least 100 PA",
    top_players.to_string(index=False),
)

low_PA_players = df[df["college_PA"] < 100][
    [
        "player_name",
        "position",
        "overall_pick",
        "college_PA",
        "college_OPS",
        "college_PA_under_100",
    ]
].sort_values("college_PA")

add_section(
    "Players with fewer than 100 PA",
    low_PA_players.to_string(index=False),
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.write_text("\n".join(report))

print("\n".join(report))
print(f"\nSaved report to: {REPORT_PATH}")