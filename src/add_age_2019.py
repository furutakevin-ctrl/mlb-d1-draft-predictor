"""Fetch draft-day ages for the 2019 player master and attach them."""

from datetime import date
from pathlib import Path
import re
import time
import unicodedata

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "players_2019_master.csv"
ROSTER_PATH = PROJECT_ROOT / "data" / "raw" / "draft_roster_2019.csv"
AGE_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "player_ages_2019.csv"
UNRESOLVED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "player_ages_unresolved_2019.csv"
)

API_ROOT = "https://statsapi.mlb.com/api/v1"
DRAFT_URL = f"{API_ROOT}/draft/2019"
PEOPLE_URL = f"{API_ROOT}/people"
HEADERS = {"User-Agent": "mlb-d1-draft-predictor/1.0"}


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def get_json(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def draft_day(overall_pick, round_label):
    # Day 1 includes the top 78 selections, including both competitive-balance rounds.
    if overall_pick <= 78:
        return date(2019, 6, 3)

    match = re.search(r"\d+", str(round_label))
    if match and int(match.group()) <= 10:
        return date(2019, 6, 4)
    return date(2019, 6, 5)


draft_data = get_json(DRAFT_URL)
draft_picks = {}
drafts = draft_data.get("drafts", [])
if isinstance(drafts, dict):
    drafts = [drafts]

for draft in drafts:
    if not isinstance(draft, dict):
        continue
    rounds = draft.get("rounds", [])
    if isinstance(rounds, dict):
        rounds = list(rounds.values())
    for round_info in rounds:
        if not isinstance(round_info, dict):
            continue
        for pick in round_info.get("picks", []):
            if not isinstance(pick, dict):
                continue
            overall_pick = pick.get("pickNumber")
            person = pick.get("person") or {}
            person_id = person.get("id")
            if overall_pick is None or person_id is None:
                continue
            overall_pick = int(overall_pick)
            round_label = pick.get("pickRound") or round_info.get("roundNumber", "")
            draft_picks[overall_pick] = {
                "mlb_person_id": int(person_id),
                "draft_api_name": person.get("fullName", ""),
                "draft_round_api": str(round_label),
            }

roster = pd.read_csv(ROSTER_PATH)
master = pd.read_csv(MASTER_PATH)
age_rows = []
unresolved = []

for _, row in roster.iterrows():
    overall_pick = int(row["overall_pick"])
    pick = draft_picks.get(overall_pick)
    if pick is None:
        unresolved.append(
            {
                "player_name": row["player_name"],
                "overall_pick": overall_pick,
                "reason": "draft API pick not found",
            }
        )
        continue

    if normalize_name(row["player_name"]) != normalize_name(pick["draft_api_name"]):
        unresolved.append(
            {
                "player_name": row["player_name"],
                "overall_pick": overall_pick,
                "reason": f"draft API name mismatch: {pick['draft_api_name']}",
            }
        )
        continue

    profile_url = f"{PEOPLE_URL}/{pick['mlb_person_id']}"
    try:
        profile = get_json(profile_url).get("people", [])
        birth_date_text = profile[0].get("birthDate") if profile else None
        if not birth_date_text:
            raise ValueError("birth date missing from player profile")
        birth_date = date.fromisoformat(birth_date_text[:10])
        day = draft_day(overall_pick, pick["draft_round_api"])
        age_years = round((day - birth_date).days / 365.2425, 3)
        if age_years <= 0:
            raise ValueError("birth date is after the draft date")
    except Exception as error:
        unresolved.append(
            {
                "player_name": row["player_name"],
                "overall_pick": overall_pick,
                "reason": str(error),
            }
        )
        continue

    age_rows.append(
        {
            "draft_year": 2019,
            "overall_pick": overall_pick,
            "player_name": row["player_name"],
            "mlb_person_id": pick["mlb_person_id"],
            "birth_date": birth_date.isoformat(),
            "draft_date": day.isoformat(),
            "age_at_draft": age_years,
            "age_source_url": profile_url,
        }
    )
    time.sleep(0.15)

if unresolved:
    pd.DataFrame(unresolved).to_csv(UNRESOLVED_PATH, index=False)
    print(f"Ages retrieved: {len(age_rows)} of {len(roster)}")
    print(f"Unresolved players: {len(unresolved)}")
    print(pd.DataFrame(unresolved).to_string(index=False))
    raise ValueError("Resolve the listed age records before modeling.")

ages = pd.DataFrame(age_rows)
if ages.duplicated(["draft_year", "overall_pick"]).any():
    raise ValueError("Duplicate draft picks found in age data.")

ages.to_csv(AGE_OUTPUT_PATH, index=False)
master = master.drop(
    columns=[
        "age_at_draft",
        "birth_date",
        "draft_date",
        "draft_person_id",
        "age_source_url",
    ],
    errors="ignore",
)
master = master.merge(
    ages[
        [
            "draft_year",
            "overall_pick",
            "age_at_draft",
            "birth_date",
            "draft_date",
            "mlb_person_id",
            "age_source_url",
        ]
    ].rename(columns={"mlb_person_id": "draft_person_id"}),
    on=["draft_year", "overall_pick"],
    how="left",
    validate="one_to_one",
)
if master["age_at_draft"].isna().any():
    raise ValueError("Some master rows did not receive an age value.")

master.to_csv(MASTER_PATH, index=False)
UNRESOLVED_PATH.unlink(missing_ok=True)
print(f"Players with age data: {len(ages)}")
print(f"Age range: {ages['age_at_draft'].min():.3f}–{ages['age_at_draft'].max():.3f}")
print(f"Saved age source data: {AGE_OUTPUT_PATH}")
print(f"Updated master data: {MASTER_PATH}")
