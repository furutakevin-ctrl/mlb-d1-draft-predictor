import json
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import certifi
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIRST_PASS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mlb_outcomes_2019_first_pass.csv"
)

OVERRIDES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mlb_manual_overrides_2019.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mlb_outcomes_2019_reviewed.csv"
)

SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)

PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"


def normalize_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace(".", "")
        .replace(" ", "")
    )


def fetch_profile(mlb_id):
    request = Request(
        f"{PEOPLE_URL}/{int(mlb_id)}",
        headers={"User-Agent": "mlb-d1-draft-predictor"},
    )

    with urlopen(
        request,
        timeout=20,
        context=SSL_CONTEXT,
    ) as response:
        people = json.load(response).get("people", [])

    return people[0] if people else None


outcomes = pd.read_csv(FIRST_PASS_PATH)
overrides = pd.read_csv(OVERRIDES_PATH)

outcomes["player_key"] = outcomes["player_name"].map(
    normalize_name
)
overrides["player_key"] = overrides["player_name"].map(
    normalize_name
)

outcomes["made_mlb_by_2024"] = pd.to_numeric(
    outcomes["made_mlb_by_2024"],
    errors="coerce",
).astype("Int64")

outcomes["mlb_debut_year"] = pd.to_numeric(
    outcomes["mlb_debut_year"],
    errors="coerce",
).astype("Int64")

for column in ["manual_status", "manual_note"]:
    if column not in outcomes.columns:
        outcomes[column] = ""

for _, override in overrides.iterrows():
    player_key = override["player_key"]

    matching_rows = outcomes.index[
        outcomes["player_key"] == player_key
    ]

    if len(matching_rows) != 1:
        print(
            f"Could not uniquely find {override['player_name']}"
        )
        continue

    row_index = matching_rows[0]
    mlb_id = int(override["mlb_id"])
    manual_status = str(override["manual_status"])
    manual_note = str(override["manual_note"])

    profile = fetch_profile(mlb_id)

    if profile is None:
        print(
            f"Could not retrieve MLB profile for "
            f"{override['player_name']}"
        )
        continue

    debut_date = profile.get("mlbDebutDate")
    debut_year = int(debut_date[:4]) if debut_date else pd.NA
    matched_name = profile.get(
        "fullName",
        override["player_name"],
    )
    api_position = (
        profile.get("primaryPosition") or {}
    ).get("abbreviation")

    outcomes.at[row_index, "mlb_id"] = mlb_id
    outcomes.at[row_index, "matched_name"] = matched_name
    outcomes.at[row_index, "mlb_debut_date"] = debut_date
    outcomes.at[row_index, "mlb_debut_year"] = debut_year
    outcomes.at[row_index, "api_position"] = api_position
    outcomes.at[row_index, "manual_status"] = manual_status
    outcomes.at[row_index, "manual_note"] = manual_note
    outcomes.at[row_index, "candidate_summary"] = json.dumps(
        [
            {
                "mlb_id": mlb_id,
                "name": matched_name,
                "debut": debut_date,
                "position": api_position,
            }
        ],
        ensure_ascii=False,
    )
    outcomes.at[row_index, "outcome_source"] = (
        f"{PEOPLE_URL}/{mlb_id}"
    )

    if manual_status == "confirmed":
        if debut_date is None:
            outcomes.at[row_index, "made_mlb_by_2024"] = 0
            outcomes.at[
                row_index, "match_status"
            ] = "manual_confirmed_no_mlb_debut"
        elif debut_date <= "2024-12-31":
            outcomes.at[row_index, "made_mlb_by_2024"] = 1
            outcomes.at[
                row_index, "match_status"
            ] = "manual_confirmed_made_by_cutoff"
        else:
            outcomes.at[row_index, "made_mlb_by_2024"] = 0
            outcomes.at[
                row_index, "match_status"
            ] = "manual_confirmed_after_cutoff"

    elif manual_status == "position_conflict":
        outcomes.at[row_index, "made_mlb_by_2024"] = pd.NA
        outcomes.at[
            row_index, "match_status"
        ] = "manual_position_conflict"


outcomes.to_csv(OUTPUT_PATH, index=False)

remaining_review = outcomes[
    outcomes["made_mlb_by_2024"].isna()
]

print(f"Created: {OUTPUT_PATH}")
print(
    "Automatically labeled:",
    int(outcomes["made_mlb_by_2024"].notna().sum()),
)
print(
    "Still requiring review:",
    len(remaining_review),
)
print("\nOutcome counts:")
print(
    outcomes["made_mlb_by_2024"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nRemaining review cases:")
print(
    remaining_review[
        [
            "player_name",
            "match_status",
            "manual_note",
        ]
    ].to_string(index=False)
)