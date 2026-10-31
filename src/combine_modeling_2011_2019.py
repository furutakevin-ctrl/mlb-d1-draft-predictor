"""Combine the provisional 2011-2018 modeling table with the 2019 table."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
HISTORICAL_PATH = PROCESSED / "modeling_2011_2018.csv"
CURRENT_PATH = PROCESSED / "modeling_2019.csv"
OUTPUT_PATH = PROCESSED / "modeling_2011_2019.csv"
YEAR_SUMMARY_PATH = PROCESSED / "modeling_year_summary_2011_2019.csv"
KEYS = ["draft_year", "overall_pick"]
REQUIRED = [
    "draft_year", "draft_round", "overall_pick", "player_name", "position",
    "college_PA", "college_OPS", "age_at_draft", "team_SOS", "made_mlb_ever",
]


def load_table(path, label, expected_years):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Build the source modeling table first.")
    df = pd.read_csv(path)
    if "made_mlb_ever" not in df.columns:
        if "made_mlb_by_2024" in df.columns:
            raise ValueError(
                f"{label} still has made_mlb_by_2024. Rebuild the 2019 outcome/modeling table "
                "using the current made_mlb_ever definition before combining years."
            )
    missing = [column for column in REQUIRED if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required comparable columns: {missing}")

    for column in ["draft_year", "draft_round", "overall_pick"]:
        if column == "draft_round":
            df[column] = df[column].astype("string").str.strip()
        else:
            df[column] = pd.to_numeric(df[column], errors="raise").astype("Int64")
    df["player_name"] = df["player_name"].astype("string").str.strip()
    if df[KEYS].isna().any().any():
        raise ValueError(f"{label} contains missing draft keys.")
    if df.duplicated(KEYS).any():
        raise ValueError(f"{label} contains duplicate draft-year/pick rows.")
    years = set(df["draft_year"].dropna().astype(int))
    if years != set(expected_years):
        raise ValueError(f"{label} has years {sorted(years)}; expected {sorted(expected_years)}.")

    for column in ["college_PA", "college_OPS", "age_at_draft", "team_SOS", "made_mlb_ever"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[["college_PA", "age_at_draft", "team_SOS", "made_mlb_ever"]].isna().any().any():
        raise ValueError(f"{label} has missing values in core modeling fields.")
    if not df["made_mlb_ever"].isin([0, 1]).all():
        raise ValueError(f"{label} has outcomes outside 0/1.")
    if not df["team_SOS"].between(0, 1).all():
        raise ValueError(f"{label} has SOS values outside [0, 1].")
    if df["college_OPS"].isna().ne(df["college_PA"].eq(0)).any():
        raise ValueError(f"{label}: missing OPS must correspond exactly to zero-PA rows.")
    df["made_mlb_ever"] = df["made_mlb_ever"].astype(int)
    # Use one stable key format across all draft years. The original 2019 key
    # was normalized by player name, while the 2011-2018 table uses pick keys.
    df["player_key"] = df["draft_year"].astype(str) + "_" + df["overall_pick"].astype(str)
    return df


def main():
    historical = load_table(HISTORICAL_PATH, "2011-2018 table", range(2011, 2019))
    season_2019 = load_table(CURRENT_PATH, "2019 table", [2019])

    all_columns = list(historical.columns) + [
        column for column in season_2019.columns if column not in historical.columns
    ]
    historical = historical.reindex(columns=all_columns)
    season_2019 = season_2019.reindex(columns=all_columns)
    combined = pd.concat([historical, season_2019], ignore_index=True)
    combined = combined.sort_values(KEYS, kind="stable").reset_index(drop=True)

    if combined.duplicated(KEYS).any():
        raise ValueError("Duplicate draft-year/pick rows found in combined table.")
    if set(combined["draft_year"].dropna().astype(int)) != set(range(2011, 2020)):
        raise ValueError("Combined table does not cover every draft year from 2011 through 2019.")

    combined.to_csv(OUTPUT_PATH, index=False)
    summary = (
        combined.groupby("draft_year", as_index=False)
        .agg(
            players=("overall_pick", "size"),
            positive_outcomes=("made_mlb_ever", "sum"),
            mlb_rate=("made_mlb_ever", "mean"),
            zero_pa_players=("college_PA", lambda values: int(values.eq(0).sum())),
        )
    )
    summary.to_csv(YEAR_SUMMARY_PATH, index=False)

    print(f"2011-2018 rows: {len(historical)}")
    print(f"2019 rows: {len(season_2019)}")
    print(f"Combined rows: {len(combined)}")
    print("Rows by draft year:")
    print(summary.to_string(index=False))
    print(f"Saved combined table: {OUTPUT_PATH}")
    print(f"Saved year summary: {YEAR_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
