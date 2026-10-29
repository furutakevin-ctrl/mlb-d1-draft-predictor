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
PILOT_PATH = PROJECT_ROOT / "data" / "pilot_modeling.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "pilot_api_outcomes.csv"

SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)

BASE_URL = "https://statsapi.mlb.com/api/v1/people/search"


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


def career_mlb_plate_appearances(mlb_id):
    query = urlencode({
        "stats": "yearByYear",
        "group": "hitting",
        "sportIds": 1,
        "gameType": "R",
    })
    request = Request(
        f"https://statsapi.mlb.com/api/v1/people/{int(mlb_id)}/stats?{query}",
        headers={"User-Agent": "mlb-d1-draft-predictor"},
    )
    with urlopen(request, timeout=20, context=SSL_CONTEXT) as response:
        payload = json.load(response)
    splits = [
        split
        for group in payload.get("stats", [])
        for split in group.get("splits", [])
    ]
    return sum(
        int(split.get("stat", {}).get("plateAppearances", 0))
        for split in splits
    )


def position_code(player):
    return (player.get("primaryPosition") or {}).get("abbreviation")


pilot = pd.read_csv(PILOT_PATH)
results = []

for _, row in pilot.iterrows():
    name = row["player_name"]
    candidates, query_used = search_player(name)

    selected = None
    match_status = "manual_review"

    if len(candidates) == 1:
        selected = candidates[0]
        match_status = "unique_match"
    elif len(candidates) > 1:
        non_pitchers = [
            player
            for player in candidates
            if position_code(player) != "P"
        ]

        if len(non_pitchers) == 1:
            selected = non_pitchers[0]
            match_status = "unique_non_pitcher_match"

    debut_date = None
    made_mlb_ever = pd.NA

    if selected is not None:
        debut_date = selected.get("mlbDebutDate")

        try:
            mlb_career_pa = career_mlb_plate_appearances(selected["id"])
            made_mlb_ever = int(mlb_career_pa > 0)
            match_status = (
                "recorded_mlb_pa" if made_mlb_ever
                else "no_mlb_pa_as_of_check"
            )
        except Exception as error:
            mlb_career_pa = pd.NA
            made_mlb_ever = pd.NA
            match_status = "mlb_pa_lookup_error"
            print(f"PA lookup failed for {name}: {error}")
    else:
        mlb_career_pa = pd.NA

    results.append(
        {
            "player_name": name,
            "historical_label_by_2024": row.get("made_mlb_by_2024", pd.NA),
            "query_used": query_used,
            "mlb_id": selected.get("id") if selected else pd.NA,
            "matched_name": selected.get("fullName") if selected else pd.NA,
            "mlb_debut_date": debut_date,
            "api_position": position_code(selected) if selected else pd.NA,
            "made_mlb_ever": made_mlb_ever,
            "mlb_career_pa": mlb_career_pa,
            "match_status": match_status,
        }
    )

    time.sleep(0.25)

results = pd.DataFrame(results)

results["matches_historical_by_2024_label"] = (
    results["historical_label_by_2024"].astype("Int64")
    == results["made_mlb_ever"].astype("Int64")
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_PATH, index=False)

print(results.to_string(index=False))
print()
print(f"Created: {OUTPUT_PATH}")
print(f"Players checked: {len(results)}")
print(
    "Manual reviews needed:",
    int((results["match_status"] == "manual_review").sum()),
)
print(
    "Label disagreements:",
    int((results["label_agrees"] == False).sum()),
)