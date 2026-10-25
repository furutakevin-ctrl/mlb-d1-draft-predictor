from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "players_2019_master.csv"
OUTCOME_PATH = PROJECT_ROOT / "data" / "processed" / "mlb_outcomes_2019_reviewed.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "modeling_2019.csv"

JOIN_KEYS = ["draft_year", "overall_pick"]


def clean_keys(dataframe):
    dataframe = dataframe.copy()

    dataframe["player_name"] = (
        dataframe["player_name"]
        .astype("string")
        .str.strip()
    )

    for column in JOIN_KEYS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        ).astype("Int64")

    return dataframe


master = pd.read_csv(MASTER_PATH)
outcomes = pd.read_csv(OUTCOME_PATH)

master = clean_keys(master)
outcomes = clean_keys(outcomes)

required_outcome_column = "made_mlb_by_2024"

if required_outcome_column not in outcomes.columns:
    raise ValueError(
        f"Missing required outcome column: {required_outcome_column}"
    )

outcomes[required_outcome_column] = pd.to_numeric(
    outcomes[required_outcome_column],
    errors="coerce",
)

if outcomes[required_outcome_column].isna().any():
    raise ValueError(
        "Some players still have missing MLB outcome labels."
    )

if not outcomes[required_outcome_column].isin([0, 1]).all():
    raise ValueError(
        "The MLB outcome column must contain only 0 or 1."
    )

if master.duplicated(JOIN_KEYS).any():
    raise ValueError(
        "The master dataset contains duplicate draft-year/pick keys."
    )

if outcomes.duplicated(JOIN_KEYS).any():
    raise ValueError(
        "The outcome dataset contains duplicate draft-year/pick keys."
    )

key_check = master[JOIN_KEYS].merge(
    outcomes[JOIN_KEYS],
    on=JOIN_KEYS,
    how="outer",
    indicator=True,
)

missing_outcomes = key_check[
    key_check["_merge"] != "both"
]

if not missing_outcomes.empty:
    raise ValueError(
        "Master and outcome files do not contain the same players:\n"
        + missing_outcomes.to_string(index=False)
    )

name_check = master[JOIN_KEYS + ["player_name"]].merge(
    outcomes[JOIN_KEYS + ["player_name"]],
    on=JOIN_KEYS,
    how="inner",
    suffixes=("_master", "_outcome"),
    validate="one_to_one",
)

name_mismatches = name_check[
    name_check["player_name_master"]
    != name_check["player_name_outcome"]
]

if not name_mismatches.empty:
    raise ValueError(
        "Player-name mismatches found:\n"
        + name_mismatches.to_string(index=False)
    )

outcome_columns = [
    "mlb_id",
    "mlb_debut_year",
    "made_mlb_by_2024",
    "mlb_games_by_2024",
    "outcome_source",
    "outcome_notes",
    "match_status",
    "manual_note",
]

outcome_columns = [
    column
    for column in outcome_columns
    if column in outcomes.columns
]

master_for_merge = master.drop(
    columns=[
        column
        for column in outcome_columns
        if column in master.columns
    ],
    errors="ignore",
)

modeling = master_for_merge.merge(
    outcomes[JOIN_KEYS + outcome_columns],
    on=JOIN_KEYS,
    how="left",
    validate="one_to_one",
)

if len(modeling) != len(master):
    raise ValueError(
        "The merged dataset does not have the expected number of rows."
    )

if modeling[required_outcome_column].isna().any():
    raise ValueError(
        "The merged dataset contains missing outcome labels."
    )

modeling[required_outcome_column] = modeling[
    required_outcome_column
].astype(int)

modeling.to_csv(OUTPUT_PATH, index=False)

print(f"Master rows: {len(master)}")
print(f"Outcome rows: {len(outcomes)}")
print(f"Modeling rows: {len(modeling)}")
print("\nOutcome counts:")
print(
    modeling[required_outcome_column]
    .value_counts()
    .sort_index()
    .to_string()
)
print(f"\nSaved: {OUTPUT_PATH}")