from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_path = PROJECT_ROOT / "data" / "pilot_modeling.csv"

df = pd.read_csv(data_path)

print("=== PILOT DATASET SUMMARY ===")
print(f"Players: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\n=== MLB OUTCOME COUNTS ===")
outcomes = (
    df["made_mlb_ever"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "Did not reach MLB",
        1: "Reached MLB"
    })
)
print(outcomes)

print("\n=== OVERALL MLB RATE ===")
mlb_rate = df["made_mlb_ever"].mean() * 100
print(f"{mlb_rate:.1f}%")

print("\n=== DRAFT PICK BY OUTCOME ===")
print(
    df.groupby("made_mlb_ever")["overall_pick"]
    .agg(["count", "mean", "median", "min", "max"])
    .rename(index={
        0: "Did not reach MLB",
        1: "Reached MLB"
    })
)

print("\n=== COLLEGE OPS BY OUTCOME ===")
print(
    df.groupby("made_mlb_ever")["college_OPS"]
    .agg(["count", "mean", "median", "min", "max"])
    .rename(index={
        0: "Did not reach MLB",
        1: "Reached MLB"
    })
)

print("\n=== COLLEGE OBP BY OUTCOME ===")
print(
    df.groupby("made_mlb_ever")["college_OBP"]
    .agg(["count", "mean", "median", "min", "max"])
    .rename(index={
        0: "Did not reach MLB",
        1: "Reached MLB"
    })
)

print("\n=== COLLEGE SLG BY OUTCOME ===")
print(
    df.groupby("made_mlb_ever")["college_SLG"]
    .agg(["count", "mean", "median", "min", "max"])
    .rename(index={
        0: "Did not reach MLB",
        1: "Reached MLB"
    })
)

print("\n=== PLAYER-LEVEL SUMMARY ===")
columns = [
    "player_name",
    "college",
    "position",
    "overall_pick",
    "college_OPS",
    "college_OBP",
    "college_SLG",
    "college_HR",
    "college_SB",
    "made_mlb_ever",
]

print(
    df[columns]
    .sort_values("overall_pick")
    .to_string(index=False)
)