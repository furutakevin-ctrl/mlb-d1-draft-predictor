"""Validate the combined provisional MLB D1 cohort for draft years 2011-2019."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_2011_2019.csv"
KEYS = ["draft_year", "overall_pick"]
REQUIRED = [
    "player_name", "position", "draft_year", "overall_pick", "college_PA",
    "college_OPS", "team_SOS", "age_at_draft", "made_mlb_ever",
]

df = pd.read_csv(DATA_PATH)
missing_columns = [column for column in REQUIRED if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

numeric_columns = [
    "draft_year", "overall_pick", "college_PA", "college_OPS", "college_AB",
    "college_BB", "college_SO", "college_HBP", "college_SF", "college_SH",
    "college_BA", "college_OBP", "college_SLG", "college_BB_rate",
    "college_K_rate", "college_HR_rate", "college_ISO", "college_XBH_rate",
    "college_SB_success_rate", "team_SOS", "age_at_draft", "made_mlb_ever",
]
for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

df["draft_year"] = df["draft_year"].astype("Int64")
df["overall_pick"] = df["overall_pick"].astype("Int64")

print(f"Rows: {len(df)}")
print(f"Unique draft-year/pick keys: {df[KEYS].drop_duplicates().shape[0]}")

if df.duplicated(KEYS).any():
    duplicates = df.loc[df.duplicated(KEYS, keep=False), KEYS + ["player_name"]]
    raise ValueError("Duplicate draft-year/pick rows:\n" + duplicates.to_string(index=False))

years = sorted(df["draft_year"].dropna().astype(int).unique().tolist())
expected_years = list(range(2011, 2020))
if years != expected_years:
    raise ValueError(f"Expected draft years {expected_years}; found {years}.")

core_missing = df[REQUIRED].isna().sum()
print("\nMissing values in core columns:")
print(core_missing.to_string())
allowed_ops_missing = df["college_PA"].eq(0)
if df.loc[~allowed_ops_missing, "college_OPS"].isna().any():
    raise ValueError("Missing OPS found for a player with positive college PA.")
other_missing = core_missing.drop(labels=["college_OPS"], errors="ignore")
if other_missing.sum():
    raise ValueError("Missing values found in required core modeling columns.")

if not df["made_mlb_ever"].isin([0, 1]).all():
    raise ValueError("made_mlb_ever must contain only 0 or 1.")
if not df["team_SOS"].between(0, 1).all():
    raise ValueError("team_SOS must be between 0 and 1.")
if not df["age_at_draft"].between(16, 50).all():
    raise ValueError("age_at_draft contains values outside the expected range.")
if df["college_PA"].lt(0).any():
    raise ValueError("college_PA contains negative values.")

if df["college_OPS"].isna().ne(allowed_ops_missing).any():
    raise ValueError("Missing OPS should occur only for zero-PA seasons.")

pa_parts = ["college_AB", "college_BB", "college_HBP", "college_SF", "college_SH"]
if all(column in df.columns for column in pa_parts):
    if df[pa_parts].isna().any().any():
        raise ValueError("Missing PA component values.")
    reconstructed = df[pa_parts].sum(axis=1)
    mismatch = df["college_PA"].ne(reconstructed)
    if mismatch.any():
        cols = KEYS + ["player_name", "college_PA", *pa_parts]
        report = df.loc[mismatch, cols].copy()
        report["reconstructed_PA"] = reconstructed[mismatch]
        raise ValueError("PA totals do not reconcile:\n" + report.head(40).to_string(index=False))

if all(column in df.columns for column in ["college_BB", "college_SO"]):
    count_columns = ["college_BB", "college_SO"]
    if df[count_columns].isna().any().any():
        raise ValueError("Missing college BB or SO counts.")
    for column in ["college_BB", "college_SO"]:
        if df[column].lt(0).any():
            raise ValueError(f"{column} contains negative counts.")
        noninteger = (df[column] - df[column].round()).abs().gt(1e-9)
        if noninteger.any():
            raise ValueError(f"{column} should contain whole-number counts.")

year_summary = (
    df.groupby("draft_year", dropna=False)
    .agg(
        players=("player_name", "size"),
        positive_outcomes=("made_mlb_ever", "sum"),
        mlb_rate=("made_mlb_ever", "mean"),
        zero_pa_players=("college_PA", lambda values: int(values.eq(0).sum())),
    )
    .reset_index()
)
print("\nRows by draft year:")
print(year_summary.to_string(index=False, formatters={"mlb_rate": lambda value: f"{value:.1%}"}))
print(f"\nPositive outcomes: {int(df['made_mlb_ever'].sum())}")
print(f"Overall MLB rate: {df['made_mlb_ever'].mean():.1%}")
print(f"Zero-PA players: {int(df['college_PA'].eq(0).sum())}")

print("\nPosition counts:")
print(df["position"].value_counts(dropna=False).sort_index().to_string())

pitcher_positions = {"P", "SP", "RP", "RHP", "LHP"}
pitchers = df[df["position"].astype(str).str.upper().isin(pitcher_positions)]
print(f"\nRows labeled as pitchers: {len(pitchers)}")
if len(pitchers):
    raise ValueError("Pitcher rows found in the position-player cohort.")

print("\nPlayers with fewer than 100 college PA (first 40):")
low_pa = df[df["college_PA"] < 100][
    ["draft_year", "player_name", "position", "overall_pick", "college_PA", "college_OPS", "made_mlb_ever"]
].sort_values(["draft_year", "college_PA"])
print(low_pa.head(40).to_string(index=False))

print("\nValidation complete.")
