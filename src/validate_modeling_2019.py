from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_2019.csv"

df = pd.read_csv(DATA_PATH)

print(f"Rows: {len(df)}")
print(f"Unique players: {df['player_key'].nunique()}")
print(f"Unique draft picks: {df['overall_pick'].nunique()}")

if len(df) != 185:
    raise ValueError("Expected 185 rows.")

if df["player_key"].nunique() != 185:
    raise ValueError("Duplicate player keys found.")

if df["overall_pick"].nunique() != 185:
    raise ValueError("Duplicate draft picks found.")

required_columns = [
    "player_name",
    "position",
    "draft_year",
    "overall_pick",
    "college_PA",
    "college_OPS",
    "team_SOS",
    "age_at_draft",
    "made_mlb_ever",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

numeric_columns = [
    "draft_year",
    "overall_pick",
    "college_PA",
    "college_OPS",
    "college_BA",
    "college_OBP",
    "college_SLG",
    "college_BB_rate",
    "college_K_rate",
    "college_HR_rate",
    "college_ISO",
    "college_XBH_rate",
    "college_SB_success_rate",
    "team_SOS",
    "age_at_draft",
    "made_mlb_ever",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

print("\nMissing values in core columns:")
core_missing = df[required_columns].isna().sum()
print(core_missing.to_string())

if core_missing.sum() > 0:
    raise ValueError("Missing values found in core modeling columns.")

if not df["made_mlb_ever"].isin([0, 1]).all():
    raise ValueError(
        "Outcome column contains values other than 0 or 1."
    )

if not df["team_SOS"].between(0, 1).all():
    raise ValueError("team_SOS must be a numeric value between 0 and 1.")

if not df["age_at_draft"].between(16, 50).all():
    raise ValueError("age_at_draft contains values outside the expected range.")

print("\nOutcome counts:")
print(
    df["made_mlb_ever"]
    .value_counts()
    .sort_index()
    .to_string()
)

print(
    f"\nOverall MLB rate: "
    f"{df['made_mlb_ever'].mean():.1%}"
)

print("\nPosition counts:")
print(
    df["position"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nPlayers with fewer than 100 college PA:")
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

print(low_pa.to_string(index=False))

pitcher_positions = {"P", "SP", "RP", "RHP", "LHP"}
pitcher_rows = df[
    df["position"]
    .astype(str)
    .str.upper()
    .isin(pitcher_positions)
]

print(
    f"\nRows labeled as pitchers: {len(pitcher_rows)}"
)

if len(pitcher_rows) > 0:
    print(pitcher_rows[
        ["player_name", "position", "overall_pick"]
    ].to_string(index=False))

print("\nValidation complete.")
