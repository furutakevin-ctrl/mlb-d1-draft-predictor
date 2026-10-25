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
    made_mlb_by_2024 = pd.NA

    if selected is not None:
        debut_date = selected.get("mlbDebutDate")

        if debut_date is None:
            made_mlb_by_2024 = 0
            match_status = "matched_no_mlb_debut"
        else:
            made_mlb_by_2024 = int(debut_date <= "2024-12-31")

    results.append(
        {
            "player_name": name,
            "pilot_label": row["made_mlb_by_2024"],
            "query_used": query_used,
            "mlb_id": selected.get("id") if selected else pd.NA,
            "matched_name": selected.get("fullName") if selected else pd.NA,
            "mlb_debut_date": debut_date,
            "api_position": position_code(selected) if selected else pd.NA,
            "made_mlb_by_2024": made_mlb_by_2024,
            "match_status": match_status,
        }
    )

    time.sleep(0.25)

results = pd.DataFrame(results)

results["label_agrees"] = (
    results["pilot_label"].astype("Int64")
    == results["made_mlb_by_2024"].astype("Int64")
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