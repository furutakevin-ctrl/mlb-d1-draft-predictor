from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

roster_path = PROJECT_ROOT / "data" / "pilot_roster.csv"
college_path = PROJECT_ROOT / "data" / "pilot_college_stats.csv"
output_path = PROJECT_ROOT / "data" / "pilot_modeling.csv"

roster = pd.read_csv(roster_path)
college = pd.read_csv(college_path)

merged = roster.merge(
    college,
    on=["player_name", "college"],
    how="outer",
    indicator=True,
    suffixes=("_draft", "_college"),
    validate="one_to_one",
)

print(f"Roster rows: {len(roster)}")
print(f"College-stat rows: {len(college)}")
print(f"Merged rows: {len(merged)}")

unmatched = merged[merged["_merge"] != "both"]

if len(unmatched) > 0:
    print("\nUnmatched records:")
    print(unmatched[["player_name", "college", "_merge"]])
    raise SystemExit("Fix unmatched records before continuing.")

merged = merged.drop(columns=["_merge"])
merged.to_csv(output_path, index=False)

print(f"\nSaved combined dataset to: {output_path}")
print("\nMissing values:")
print(merged.isna().sum())
