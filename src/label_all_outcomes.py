import json
import re
import ssl
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEMPLATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mlb_outcomes_2019.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mlb_outcomes_2019_first_pass.csv"
)

SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)

BASE_URL = "https://statsapi.mlb.com/api/v1/people/search"
CUTOFF_DATE = "2024-12-31"


def name_variants(name):
    original = str(name).strip()
    without_periods = re.sub(r"\.", "", original)

    return list(dict.fromkeys([
        original,
        without_periods,
    ]))


def search_player(name):
    for search_name in name_variants(name):
        query = urlencode({"names": search_name})

        request = Request(
            f"{BASE_URL}?{query}",
            headers={"User-Agent": "mlb-d1-draft-predictor"},
        )

        with urlopen(
            request,
            timeout=20,
            context=SSL_CONTEXT,
        ) as response:
            candidates = json.load(response).get("people", [])

        if candidates:
            return candidates, search_name

    return [], None


def api_position(player):
    return (player.get("primaryPosition") or {}).get(
        "abbreviation"
    )


def is_pitcher(position):
    return position in {"P", "LHP", "RHP", "SHP"}


def candidate_summary(candidates):
    summary = []

    for player in candidates:
        summary.append(
            {
                "mlb_id": player.get("id"),
                "name": player.get("fullName"),
                "debut": player.get("mlbDebutDate"),
                "position": api_position(player),
            }
        )

    return json.dumps(summary, ensure_ascii=False)


template = pd.read_csv(TEMPLATE_PATH)
results = []

for _, row in template.iterrows():
    name = row["player_name"]
    roster_position = str(row["position"]).strip()

    candidates, query_used = search_player(name)

    selected = None
    match_status = "manual_review"

    if not candidates:
        match_status = "no_api_match"

    elif len(candidates) == 1:
        selected = candidates[0]
        selected_position = api_position(selected)

        if (
            is_pitcher(selected_position)
            and not is_pitcher(roster_position)
        ):
            match_status = "position_conflict_review"
        else:
            match_status = "unique_match"

    else:
        non_pitchers = [
            player
            for player in candidates
            if not is_pitcher(api_position(player))
        ]

        if len(non_pitchers) == 1:
            selected = non_pitchers[0]
            match_status = "unique_non_pitcher_match"
        else:
            match_status = "manual_review"

    debut_date = None
    debut_year = pd.NA
    made_mlb_by_2024 = pd.NA

    if selected is not None:
        debut_date = selected.get("mlbDebutDate")

        if debut_date:
            debut_year = int(debut_date[:4])

        if match_status not in {
            "position_conflict_review",
            "manual_review",
        }:
            if debut_date is None:
                made_mlb_by_2024 = 0
                match_status = "matched_no_mlb_debut"
            elif debut_date <= CUTOFF_DATE:
                made_mlb_by_2024 = 1
                match_status = "made_by_cutoff"
            else:
                made_mlb_by_2024 = 0
                match_status = "debut_after_cutoff"

    output_row = row.to_dict()

    output_row.update(
        {
            "mlb_id": selected.get("id") if selected else pd.NA,
            "matched_name": (
                selected.get("fullName")
                if selected
                else pd.NA
            ),
            "query_used": query_used,
            "mlb_debut_year": debut_year,
            "mlb_debut_date": debut_date,
            "api_position": (
                api_position(selected)
                if selected
                else pd.NA
            ),
            "made_mlb_by_2024": made_mlb_by_2024,
            "match_status": match_status,
            "candidate_summary": candidate_summary(candidates),
            "outcome_source": BASE_URL,
        }
    )

    results.append(output_row)
    time.sleep(0.20)


results_df = pd.DataFrame(results)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(OUTPUT_PATH, index=False)

manual_review = results_df[
    results_df["made_mlb_by_2024"].isna()
]

print(f"Players checked: {len(results_df)}")
print(
    "Automatically labeled:",
    int(results_df["made_mlb_by_2024"].notna().sum()),
)
print(
    "Manual review needed:",
    len(manual_review),
)
print("\nOutcome counts:")
print(
    results_df["made_mlb_by_2024"]
    .value_counts(dropna=False)
    .to_string()
)

if not manual_review.empty:
    print("\nPlayers requiring review:")
    print(
        manual_review[
            [
                "player_name",
                "position",
                "match_status",
                "candidate_summary",
            ]
        ].to_string(index=False)
    )

print(f"\nCreated: {OUTPUT_PATH}")