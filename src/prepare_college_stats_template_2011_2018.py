"""Prepare a row-aligned worksheet for final college-season batting stats."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "d1_position_candidates_2011_2018.csv"
OUTPUT = ROOT / "data" / "raw" / "college_stats_2011_2018_template.csv"

STAT_COLUMNS = [
    "college_games", "college_PA", "college_AB", "college_R", "college_H",
    "college_2B", "college_3B", "college_HR", "college_RBI", "college_TB",
    "college_BB", "college_SO", "college_HBP", "college_GDP", "college_SF",
    "college_SH", "college_SB", "college_CS", "college_BA", "college_OBP",
    "college_SLG", "college_OPS", "stat_source", "notes",
]


def main():
    candidates = pd.read_csv(INPUT)
    identity = ["draft_year", "draft_round", "overall_pick", "player_name",
                "mlb_person_id", "college", "position", "team_SOS", "team_SOS_rank"]
    template = candidates[[column for column in identity if column in candidates.columns]].copy()
    # For college players, the season played during the draft year is the
    # final completed college season used by the 2019 pilot workflow.
    template["college_season"] = template["draft_year"]
    for column in STAT_COLUMNS:
        template[column] = pd.NA
    template = template[[*identity, "college_season", *STAT_COLUMNS]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(OUTPUT, index=False)
    print(f"Template rows: {len(template)}")
    print(f"Saved: {OUTPUT}")
    print("Fill stats from the player's official college season line and retain its source URL.")


if __name__ == "__main__":
    main()
