"""Create draft-day age records for the provisional 2011-2018 D1 cohort."""

from datetime import date
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "d1_position_candidates_2011_2018.csv"
OUTPUT = ROOT / "data" / "raw" / "player_ages_2011_2018.csv"
REVIEW = ROOT / "data" / "processed" / "player_ages_review_2011_2018.csv"

# First day of the applicable draft segment. The historical MLB schedules used
# rounds 1-2 / 3-10 / 11+ on separate days, with 2011 and 2012 exceptions.
DRAFT_DAYS = {
    2011: {1: date(2011, 6, 6), 2: date(2011, 6, 7)},
    2012: {1: date(2012, 6, 4), 2: date(2012, 6, 5), 16: date(2012, 6, 6)},
    2013: {1: date(2013, 6, 6), 3: date(2013, 6, 7), 11: date(2013, 6, 8)},
    2014: {1: date(2014, 6, 5), 3: date(2014, 6, 6), 11: date(2014, 6, 7)},
    2015: {1: date(2015, 6, 8), 3: date(2015, 6, 9), 11: date(2015, 6, 10)},
    2016: {1: date(2016, 6, 9), 3: date(2016, 6, 10), 11: date(2016, 6, 11)},
    2017: {1: date(2017, 6, 12), 3: date(2017, 6, 13), 11: date(2017, 6, 14)},
    2018: {1: date(2018, 6, 4), 3: date(2018, 6, 5), 11: date(2018, 6, 6)},
}


def pick_date(year, round_number):
    segments = DRAFT_DAYS[int(year)]
    applicable = [threshold for threshold in segments if threshold <= int(round_number)]
    return segments[max(applicable)]


def main():
    candidates = pd.read_csv(INPUT)
    rows, review = [], []
    for _, row in candidates.iterrows():
        key = {"draft_year": row.get("draft_year"), "overall_pick": row.get("overall_pick"),
               "player_name": row.get("player_name"), "mlb_person_id": row.get("mlb_person_id")}
        try:
            api_record = json.loads(row["api_record_json"])
            person = api_record.get("person") or {}
            birth_text = person.get("birthDate")
            if not birth_text:
                raise ValueError("birthDate missing from embedded MLB draft record")
            birth = date.fromisoformat(str(birth_text)[:10])
            draft = pick_date(row["draft_year"], row["draft_round"])
            age = round((draft - birth).days / 365.2425, 3)
            if age <= 0:
                raise ValueError("birth date is after draft date")
            rows.append({**key, "birth_date": birth.isoformat(), "draft_date": draft.isoformat(),
                         "age_at_draft": age,
                         "age_source_url": f"https://statsapi.mlb.com/api/v1/people/{int(row['mlb_person_id'])}"})
        except Exception as exc:
            review.append({**key, "reason": str(exc)})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    pd.DataFrame(review, columns=["draft_year", "overall_pick", "player_name", "mlb_person_id", "reason"]).to_csv(REVIEW, index=False)
    print(f"Candidates: {len(candidates)}")
    print(f"Ages created: {len(rows)}")
    print(f"Needs review: {len(review)}")
    print(f"Saved ages: {OUTPUT}")
    print(f"Saved review: {REVIEW}")


if __name__ == "__main__":
    main()
