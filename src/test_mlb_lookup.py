import ssl
import re
import certifi
import json
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from pathlib import Path

import pandas as pd

SSL_CONTEXT = ssl.create_default_context(
    cafile=certifi.where()
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = PROJECT_ROOT / "data" / "pilot_modeling.csv"

BASE_URL = "https://statsapi.mlb.com/api/v1/people/search"


def normalize_name(value):
    return str(value).strip().lower()


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
            context=SSL_CONTEXT
        ) as response:
            candidates = json.load(response).get("people", [])

        if candidates:
            return candidates

    return []


pilot = pd.read_csv(PILOT_PATH)

for name in pilot["player_name"]:
    print(f"\n{name}")

    try:
        candidates = search_player(name)

        if not candidates:
            print("  NO RESULTS")
        else:
            for player in candidates[:5]:
                position = player.get("primaryPosition", {})
                print(
                    f"  id={player.get('id')}, "
                    f"name={player.get('fullName')}, "
                    f"debut={player.get('mlbDebutDate')}, "
                    f"position={position.get('abbreviation')}"
                )

    except Exception as error:
        print(f"  ERROR: {error}")

    time.sleep(0.25)