"""Collect made_mlb_ever labels for the provisional 2011–2018 D1 cohort.

The MLB person ID from the draft record is used directly. A positive outcome
requires at least one MLB regular-season hitting plate appearance (sport 1,
game type R) by the check date. Failed lookups remain missing for review.
"""
from datetime import datetime
from pathlib import Path
import json
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "d1_position_candidates_2011_2018.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "mlb_outcomes_2011_2018_first_pass.csv"
REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "mlb_outcomes_2011_2018_review.csv"
API_ROOT = "https://statsapi.mlb.com/api/v1"
USER_AGENT = "mlb-d1-draft-predictor/1.0"
SAVE_EVERY_NEW_IDS = 25
REQUEST_DELAY_SECONDS = 0.15

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def record_person(record_json):
    if pd.isna(record_json) or not str(record_json).strip():
        return {}
    try:
        payload = json.loads(record_json)
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload.get("person") or {}


def career_regular_season_pa(person_id):
    url = f"{API_ROOT}/people/{int(person_id)}/stats"
    params = {
        "stats": "yearByYear",
        "group": "hitting",
        "sportIds": 1,
        "gameType": "R",
    }
    last_error = None
    for attempt in range(4):
        try:
            response = SESSION.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            splits = [
                split
                for group in payload.get("stats", [])
                for split in group.get("splits", [])
            ]
            return sum(
                int(float(split.get("stat", {}).get("plateAppearances", 0) or 0))
                for split in splits
            ), url
        except (requests.RequestException, ValueError, TypeError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(last_error))


def load_success_cache():
    if not OUTPUT_PATH.exists():
        return {}
    prior = pd.read_csv(OUTPUT_PATH)
    if "mlb_person_id" not in prior or "made_mlb_ever" not in prior:
        return {}
    prior["made_mlb_ever"] = pd.to_numeric(prior["made_mlb_ever"], errors="coerce")
    prior["mlb_career_pa"] = pd.to_numeric(prior["mlb_career_pa"], errors="coerce")
    valid = prior[prior["made_mlb_ever"].isin([0, 1]) & prior["mlb_career_pa"].notna()]
    cache = {}
    for _, row in valid.drop_duplicates("mlb_person_id", keep="last").iterrows():
        cache[int(row["mlb_person_id"])] = {
            "mlb_career_pa": int(row["mlb_career_pa"]),
            "made_mlb_ever": int(row["made_mlb_ever"]),
            "outcome_checked_date": row.get("outcome_checked_date", ""),
            "outcome_source": row.get("outcome_source", ""),
            "match_status": row.get("match_status", ""),
        }
    return cache


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run the 2011–2018 SOS filter first."
        )

    candidates = pd.read_csv(INPUT_PATH)
    required = {"draft_year", "overall_pick", "player_name", "mlb_person_id", "api_record_json"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidate file is missing columns: {sorted(missing)}")
    if candidates.duplicated(["draft_year", "overall_pick"]).any():
        raise ValueError("Duplicate draft-year/pick keys found in candidate file.")

    cache = load_success_cache()
    results = []
    new_ids = 0
    checked_date = datetime.now().astimezone().date().isoformat()

    for _, row in candidates.iterrows():
        record = record_person(row.get("api_record_json"))
        raw_id = row.get("mlb_person_id")
        api_id = record.get("id")
        status = "pending"
        output = row.to_dict()
        output.update({
            "api_player_name": record.get("fullName", ""),
            "mlb_debut_date": record.get("mlbDebutDate", ""),
            "api_position": (record.get("primaryPosition") or {}).get("abbreviation", ""),
            "mlb_career_pa": pd.NA,
            "made_mlb_ever": pd.NA,
            "outcome_checked_date": checked_date,
            "match_status": status,
            "outcome_source": "",
        })

        if pd.isna(raw_id):
            output["match_status"] = "missing_mlb_person_id"
            results.append(output)
            continue

        person_id = int(raw_id)
        if api_id is not None and int(api_id) != person_id:
            output["match_status"] = "id_conflict_review"
            results.append(output)
            continue

        if person_id in cache:
            cached = cache[person_id]
            output.update(cached)
            results.append(output)
            continue

        try:
            pa, source_url = career_regular_season_pa(person_id)
            output.update({
                "mlb_career_pa": pa,
                "made_mlb_ever": int(pa > 0),
                "outcome_checked_date": checked_date,
                "match_status": "recorded_mlb_pa" if pa > 0 else "no_mlb_pa_as_of_check",
                "outcome_source": source_url,
            })
            cache[person_id] = {
                "mlb_career_pa": pa,
                "made_mlb_ever": int(pa > 0),
                "outcome_checked_date": checked_date,
                "outcome_source": source_url,
                "match_status": output["match_status"],
            }
            new_ids += 1
            time.sleep(REQUEST_DELAY_SECONDS)
        except Exception as error:
            output["match_status"] = "lookup_error"
            output["lookup_error"] = str(error)

        results.append(output)
        if new_ids and new_ids % SAVE_EVERY_NEW_IDS == 0:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
            print(f"Saved progress: {len(results)} of {len(candidates)} candidate rows")

    result = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    review = result[result["made_mlb_ever"].isna()].copy()
    review.to_csv(REVIEW_PATH, index=False)

    print(f"Candidate rows: {len(result)}")
    print(f"Unique MLB IDs with successful lookup: {len(cache)}")
    print("Outcome counts:")
    print(result["made_mlb_ever"].value_counts(dropna=False).sort_index().to_string())
    print(f"Review rows: {len(review)}")
    print(f"Saved outcomes: {OUTPUT_PATH}")
    print(f"Saved review file: {REVIEW_PATH}")
    print("These labels cover the provisional SOS-matched cohort; finalize school and draft-day position review before modeling.")


if __name__ == "__main__":
    main()
