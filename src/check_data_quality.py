from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_path = PROJECT_ROOT / "data" / "pilot_modeling.csv"

df = pd.read_csv(data_path)

print("DATASET SHAPE")
print(df.shape)

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nDUPLICATE PLAYER NAMES")
print(df["player_name"].duplicated().sum())

print("\nMISSING VALUES")
print(df.isna().sum())

print("\nOUTCOME COUNTS")
print(df["made_mlb_by_2024"].value_counts().sort_index())

print("\nDATA TYPES")
print(df.dtypes)

print("\nRANGE CHECKS")

checks = {
    "draft_year_valid": df["draft_year"].between(2011, 2019).all(),
    "draft_round_valid": df["draft_round"].between(1, 40).all(),
    "overall_pick_positive": (df["overall_pick"] > 0).all(),
    "batting_average_valid": df["college_BA"].between(0, 1).all(),
    "on_base_percentage_valid": df["college_OBP"].between(0, 1).all(),
    "slugging_percentage_valid": df["college_SLG"].between(0, 2).all(),
    "outcome_valid": df["made_mlb_by_2024"].isin([0, 1]).all(),
}

for check_name, passed in checks.items():
    print(f"{check_name}: {passed}")

print("\nPLAYERS MISSING COLLEGE PA")
print(df[df["college_PA"].isna()][["player_name", "college"]])

print("\nDATA QUALITY CHECK COMPLETE")
