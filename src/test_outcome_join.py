from pathlib import Path
import re
import unicodedata
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "players_2019_master.csv"
)

PILOT_PATH = PROJECT_ROOT / "data" / "pilot_modeling.csv"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pilot_modeling_from_master.csv"
)


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


master = pd.read_csv(MASTER_PATH)
pilot = pd.read_csv(PILOT_PATH)

master["player_key"] = master["player_name"].map(normalize_name)
pilot["player_key"] = pilot["player_name"].map(normalize_name)

if master["player_key"].duplicated().any():
    raise ValueError("Duplicate players found in master dataset.")

if pilot["player_key"].duplicated().any():
    raise ValueError("Duplicate players found in pilot dataset.")

pilot_labels = pilot[
    [
        "player_key",
        "made_mlb_by_2024",
        "mlb_debut_year",
        "source_outcome",
    ]
].copy()

pilot_labels["made_mlb_by_2024"] = pd.to_numeric(
    pilot_labels["made_mlb_by_2024"],
    errors="raise",
)

unmatched_pilot = pilot_labels[
    ~pilot_labels["player_key"].isin(master["player_key"])
]

if not unmatched_pilot.empty:
    print("Pilot players missing from master dataset:")
    print(unmatched_pilot.to_string(index=False))
    raise ValueError("Pilot-to-master join failed.")

joined = master.merge(
    pilot_labels,
    on="player_key",
    how="left",
    validate="one_to_one",
)

joined["pilot_outcome_available"] = (
    joined["made_mlb_by_2024"].notna().astype(int)
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joined.to_csv(OUTPUT_PATH, index=False)

print(f"Master rows: {len(master)}")
print(f"Pilot rows: {len(pilot)}")
print(
    "Successfully joined pilot outcomes:",
    int(joined["pilot_outcome_available"].sum()),
)
print(
    "Outcome distribution among pilot players:",
    joined.loc[
        joined["pilot_outcome_available"] == 1,
        "made_mlb_by_2024",
    ].value_counts().to_dict(),
)
print(f"Created: {OUTPUT_PATH}")