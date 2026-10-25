from pathlib import Path
import re
import unicodedata
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ROSTER_PATH = RAW_DIR / "draft_roster_2019.csv"
STATS_PATH = RAW_DIR / "college_stats_2019.csv"
OUTPUT_PATH = PROCESSED_DIR / "players_2019_master.csv"


def normalize_name(value):
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


roster = pd.read_csv(ROSTER_PATH)
stats = pd.read_csv(STATS_PATH)

required_roster_columns = {
    "player_name",
    "college",
    "position",
    "draft_year",
    "draft_round",
    "overall_pick",
    "source_draft",
}

required_stats_columns = {
    "player_name",
    "college",
    "college_season",
    "college_games",
    "college_PA",
    "college_AB",
    "college_R",
    "college_H",
    "college_2B",
    "college_3B",
    "college_HR",
    "college_RBI",
    "college_TB",
    "college_BB",
    "college_SO",
    "college_HBP",
    "college_GDP",
    "college_SF",
    "college_SH",
    "college_SB",
    "college_CS",
    "college_BA",
    "college_OBP",
    "college_SLG",
    "college_OPS",
    "stat_source",
    "notes",
}

missing_roster = required_roster_columns - set(roster.columns)
missing_stats = required_stats_columns - set(stats.columns)

if missing_roster:
    raise ValueError(f"Missing roster columns: {sorted(missing_roster)}")

if missing_stats:
    raise ValueError(f"Missing stats columns: {sorted(missing_stats)}")


roster["player_key"] = roster["player_name"].map(normalize_name)
stats["player_key"] = stats["player_name"].map(normalize_name)

if roster["player_key"].duplicated().any():
    raise ValueError("Duplicate player names found in the draft roster.")

if stats["player_key"].duplicated().any():
    raise ValueError("Duplicate player names found in the college stats file.")


numeric_stats = [
    "college_season",
    "college_games",
    "college_PA",
    "college_AB",
    "college_R",
    "college_H",
    "college_2B",
    "college_3B",
    "college_HR",
    "college_RBI",
    "college_TB",
    "college_BB",
    "college_SO",
    "college_HBP",
    "college_GDP",
    "college_SF",
    "college_SH",
    "college_SB",
    "college_CS",
    "college_BA",
    "college_OBP",
    "college_SLG",
    "college_OPS",
]

for column in numeric_stats:
    stats[column] = pd.to_numeric(stats[column], errors="raise")


# Validate plate appearances before calculating derived features.
stats["pa_reconstructed"] = (
    stats["college_AB"]
    + stats["college_BB"]
    + stats["college_HBP"]
    + stats["college_SF"]
    + stats["college_SH"]
)

bad_pa = stats[stats["college_PA"] != stats["pa_reconstructed"]]

if not bad_pa.empty:
    print("PLATE APPEARANCE CHECK FAILED:")
    print(
        bad_pa[
            [
                "player_name",
                "college_PA",
                "college_AB",
                "college_BB",
                "college_HBP",
                "college_SF",
                "college_SH",
                "pa_reconstructed",
            ]
        ].to_string(index=False)
    )
    raise ValueError("Fix the rows listed above before building the master dataset.")


# Derived offensive features.
stats["college_BB_rate"] = stats["college_BB"] / stats["college_PA"]
stats["college_K_rate"] = stats["college_SO"] / stats["college_PA"]
stats["college_HR_rate"] = stats["college_HR"] / stats["college_PA"]
stats["college_ISO"] = stats["college_SLG"] - stats["college_BA"]
stats["college_XBH_rate"] = (
    stats["college_2B"] + stats["college_3B"] + stats["college_HR"]
) / stats["college_PA"]

stats["college_SB_attempts"] = stats["college_SB"] + stats["college_CS"]
stats["college_SB_success_rate"] = stats["college_SB"] / stats[
    "college_SB_attempts"
].where(stats["college_SB_attempts"] > 0)

# Sample-size flags. These do not remove players.
stats["college_PA_under_100"] = (stats["college_PA"] < 100).astype(int)
stats["college_PA_under_150"] = (stats["college_PA"] < 150).astype(int)

stats = stats.drop(columns=["pa_reconstructed"])


# Rename overlapping stats fields so both source values remain traceable.
stats_for_join = stats.rename(
    columns={
        "player_name": "stats_player_name",
        "college": "stats_college",
    }
)

master = roster.merge(
    stats_for_join,
    on="player_key",
    how="outer",
    indicator=True,
    validate="one_to_one",
)

unmatched = master[master["_merge"] != "both"]

if not unmatched.empty:
    print("JOIN CHECK FAILED:")
    print(unmatched.to_string(index=False))
    raise ValueError("Every roster player must have exactly one stats record.")


master = master.drop(columns=["_merge"])

draft_columns = [
    "player_name",
    "player_key",
    "college",
    "position",
    "draft_year",
    "draft_round",
    "overall_pick",
    "source_draft",
]

stats_columns = [
    column
    for column in stats_for_join.columns
    if column not in {"player_key"}
]

output_columns = draft_columns + stats_columns
master = master[output_columns]

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
master.to_csv(OUTPUT_PATH, index=False)

print(f"Created: {OUTPUT_PATH}")
print(f"Rows: {len(master)}")
print(f"Unique players: {master['player_key'].nunique()}")
print("Plate appearance checks: passed")
print("Roster-to-stats join: passed")
print(
    "Players with fewer than 100 PA:",
    int(master["college_PA_under_100"].sum()),
)
print(
    "Players with fewer than 150 PA:",
    int(master["college_PA_under_150"].sum()),
)