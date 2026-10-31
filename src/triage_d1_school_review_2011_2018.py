"""Create a compact school-level triage sheet for unmatched 2011–2018 picks.

This script does not decide D1 eligibility or exclude candidates. It groups the
current unmatched player file by draft year and school and labels only school
names with explicit secondary-school or JUCO/CC wording. Everything else is left for
D1/division and school-name verification.
"""
from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "d1_school_match_review_2011_2018.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "d1_school_review_triage_2011_2018.csv"


def classify_school_label(value: str) -> tuple[str, str]:
    text = "" if pd.isna(value) else str(value)
    lower = text.lower()
    if re.search(r"\b(high school|hs|prep|secondary school)\b", lower):
        return "explicit_high_school_or_prep_label", "Likely not an NCAA college program; verify if unclear."
    if re.search(r"\b(community college|junior college|juco|cc|jc)\b", lower):
        return "explicit_junior_or_community_college_label", "Likely not an NCAA D1 program; verify if unclear."
    if "academy" in lower:
        return "academy_label_needs_school_type_check", "Academy wording is ambiguous; verify whether this was a college or secondary school."
    return "division_or_school_name_review", "Check D1 status for this draft year and resolve any school-name variant."


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_PATH}. Run filter_d1_candidates_add_sos_2011_2018.py first."
        )
    review = pd.read_csv(INPUT_PATH)
    required = {"draft_year", "college", "player_name", "position"}
    missing = required.difference(review.columns)
    if missing:
        raise ValueError(f"Review file is missing required columns: {sorted(missing)}")

    rows = []
    for (year, school), group in review.groupby(["draft_year", "college"], dropna=False, sort=True):
        category, note = classify_school_label(school)
        rows.append({
            "draft_year": int(year),
            "college_label": "" if pd.isna(school) else str(school),
            "players_to_review": len(group),
            "player_names": "; ".join(sorted(group["player_name"].dropna().astype(str).unique())),
            "api_positions": "; ".join(sorted(group["position"].dropna().astype(str).unique())),
            "triage_category": category,
            "review_note": note,
            "manual_decision": "",
            "verified_d1_school_name": "",
            "verification_source": "",
        })

    output = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"School-year rows: {len(output)}")
    print(f"Players represented: {output['players_to_review'].sum()}")
    print("Triage counts by player:")
    print(output.groupby("triage_category")["players_to_review"].sum().to_string())
    print(f"Saved: {OUTPUT_PATH}")
    print("No records were auto-included or excluded from the D1 cohort.")


if __name__ == "__main__":
    main()
