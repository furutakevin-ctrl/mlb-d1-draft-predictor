"""Download MLB Stats API draft records for 2011–2018.

This creates a raw candidate list for review. The API does not reliably encode
historical NCAA division, so this script intentionally does not claim that all
returned college players are Division I or position players.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests


YEARS = range(2018, 2010, -1)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "raw" / "draft_candidates_2011_2018.csv"
API_ROOT = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "mlb-d1-draft-predictor/1.0"}


def as_items(value, *, dictionary_is_one_object=False):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if dictionary_is_one_object:
            return [value]
        return list(value.values())
    return []


def get_json(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def round_number(pick, round_info):
    value = pick.get("pickRound") or pick.get("round")
    if value is None:
        value = round_info.get("roundNumber") or round_info.get("round")
    match = re.search(r"\d+", str(value or ""))
    return int(match.group()) if match else None


def text_or_name(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("fullName") or value.get("displayName") or ""
    return str(value).strip() if value is not None else ""


def position_abbr(value):
    if isinstance(value, dict):
        return value.get("abbreviation") or value.get("code") or value.get("name") or ""
    return str(value).strip() if value is not None else ""


def fetch_year(year):
    url = f"{API_ROOT}/draft/{year}"
    payload = get_json(url)
    rows = []

    # The Stats API's `drafts` member is an object containing `rounds`, rather
    # than an ordinary list. Treat it as one draft object (same shape used by
    # the working age-enrichment script).
    for draft in as_items(payload.get("drafts"), dictionary_is_one_object=True):
        if not isinstance(draft, dict):
            continue
        for round_info in as_items(draft.get("rounds")):
            if not isinstance(round_info, dict):
                continue
            for pick in as_items(round_info.get("picks")):
                if not isinstance(pick, dict):
                    continue
                rnd = round_number(pick, round_info)
                overall_pick = pick.get("pickNumber") or pick.get("overallPickNumber")
                person = pick.get("person") or {}
                school = pick.get("school") or pick.get("college") or {}
                if rnd is None or rnd > 20 or overall_pick is None:
                    continue

                person_id = person.get("id") if isinstance(person, dict) else None
                person_name = text_or_name(person)
                position = (
                    pick.get("primaryPosition")
                    or pick.get("position")
                    or (person.get("primaryPosition") if isinstance(person, dict) else None)
                )
                rows.append(
                    {
                        "draft_year": year,
                        "draft_round": rnd,
                        "overall_pick": int(overall_pick),
                        "player_name": person_name,
                        "mlb_person_id": person_id,
                        "college": text_or_name(school),
                        "position": position_abbr(position),
                        "school_type": text_or_name(school.get("schoolType")) if isinstance(school, dict) else "",
                        "source_draft": f"https://www.mlb.com/draft/tracker/{year}",
                        "source_api": url,
                        "api_record_json": json.dumps(pick, ensure_ascii=False),
                    }
                )

    return rows


def main():
    all_rows = []
    for year in YEARS:
        year_rows = fetch_year(year)
        print(f"{year}: {len(year_rows)} picks through round 20")
        all_rows.extend(year_rows)
        time.sleep(0.25)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise ValueError(
            "No draft picks were parsed. Inspect the API response structure "
            "or the saved per-year diagnostics before retrying."
        )
    if df.duplicated(["draft_year", "overall_pick"]).any():
        duplicates = df.loc[df.duplicated(["draft_year", "overall_pick"], keep=False)]
        raise ValueError("Duplicate year/pick keys found:\n" + duplicates.to_string(index=False))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df)} draft candidates to {OUTPUT}")
    print("This is a source roster, not yet the filtered D1 position-player sample.")
    print("Review missing college/position values and verify NCAA division by draft year before proceeding.")


if __name__ == "__main__":
    main()
