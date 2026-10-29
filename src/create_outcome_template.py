from pathlib import Path
import re
import unicodedata
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSTER_PATH = PROJECT_ROOT / "data" / "raw" / "draft_roster_2019.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "mlb_outcomes_2019.csv"


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


roster = pd.read_csv(ROSTER_PATH)

outcomes = roster[
    [
        "player_name",
        "draft_year",
        "overall_pick",
        "position",
    ]
].copy()

outcomes.insert(
    1,
    "player_key",
    outcomes["player_name"].map(normalize_name),
)

outcomes["mlb_debut_year"] = pd.Series(
    pd.NA,
    index=outcomes.index,
    dtype="Int64",
)

outcomes["mlb_career_pa"] = pd.Series(
    pd.NA,
    index=outcomes.index,
    dtype="Int64",
)

outcomes["made_mlb_ever"] = pd.Series(
    pd.NA,
    index=outcomes.index,
    dtype="Int64",
)

outcomes["outcome_checked_date"] = ""
outcomes["outcome_source"] = ""
outcomes["outcome_notes"] = ""

outcomes.to_csv(OUTPUT_PATH, index=False)

print(f"Created: {OUTPUT_PATH}")
print(f"Players: {len(outcomes)}")
print(f"Unlabeled outcomes: {outcomes['made_mlb_ever'].isna().sum()}")