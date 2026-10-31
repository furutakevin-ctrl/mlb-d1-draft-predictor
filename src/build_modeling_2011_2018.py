"""Join the provisional 2011-2018 cohort, stats, ages, and MLB outcomes."""
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"

CANDIDATES_PATH = PROCESSED / "d1_position_candidates_2011_2018.csv"
STATS_PATH = RAW / "college_stats_2011_2018_first_pass.csv"
AGES_PATH = RAW / "player_ages_2011_2018.csv"
OUTCOMES_PATH = PROCESSED / "mlb_outcomes_2011_2018_first_pass.csv"
MASTER_PATH = PROCESSED / "players_2011_2018_master.csv"
MODELING_PATH = PROCESSED / "modeling_2011_2018.csv"
SCHOOL_REVIEW_PATH = PROCESSED / "d1_school_match_review_2011_2018.csv"

KEYS = ["draft_year", "overall_pick"]
COUNTING_STATS = [
    "college_games", "college_PA", "college_AB", "college_R", "college_H",
    "college_2B", "college_3B", "college_HR", "college_RBI", "college_TB",
    "college_BB", "college_SO", "college_HBP", "college_GDP", "college_SF",
    "college_SH", "college_SB", "college_CS",
]
RATE_STATS = ["college_BA", "college_OBP", "college_SLG", "college_OPS"]


def read_keyed(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    frame = pd.read_csv(path)
    missing = set(KEYS).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing join columns: {sorted(missing)}")
    frame = frame.copy()
    for key in KEYS:
        frame[key] = pd.to_numeric(frame[key], errors="raise").astype("Int64")
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{label} contains missing join keys")
    if frame.duplicated(KEYS).any():
        duplicates = frame.loc[frame.duplicated(KEYS, keep=False), KEYS]
        raise ValueError(f"{label} has duplicate draft-year/pick keys:\n{duplicates.to_string(index=False)}")
    if "player_name" in frame:
        frame["player_name"] = frame["player_name"].astype("string").str.strip()
    return frame


def align_to_candidates(candidates, source, label, keep_columns):
    columns = KEYS + [c for c in keep_columns if c in source.columns and c not in KEYS]
    selected = source[columns].copy()
    merged = candidates[KEYS].merge(
        selected, on=KEYS, how="left", validate="one_to_one", indicator=True
    )
    missing = merged[merged["_merge"] != "both"]
    if not missing.empty:
        raise ValueError(
            f"{label} is missing {len(missing)} cohort rows:\n"
            + missing[KEYS].to_string(index=False)
        )
    return merged.drop(columns="_merge")


def assert_same_names(candidates, aligned, name_column, label):
    if name_column not in aligned:
        return
    names = aligned[name_column].astype("string").str.strip()
    candidate_names = candidates["player_name"].astype("string").str.strip()
    mismatch = names.ne(candidate_names)
    if mismatch.any():
        check = candidates.loc[mismatch, KEYS + ["player_name"]].copy()
        check[label + "_player_name"] = names[mismatch].values
        raise ValueError(f"Player-name mismatch in {label}:\n{check.to_string(index=False)}")


def add_derived_features(frame):
    pa = frame["college_PA"]
    positive_pa = pa.gt(0)
    frame["college_BB_rate"] = (frame["college_BB"] / pa).where(positive_pa)
    frame["college_K_rate"] = (frame["college_SO"] / pa).where(positive_pa)
    frame["college_HR_rate"] = (frame["college_HR"] / pa).where(positive_pa)
    frame["college_ISO"] = frame["college_SLG"] - frame["college_BA"]
    frame["college_XBH_rate"] = (
        (frame["college_2B"] + frame["college_3B"] + frame["college_HR"]) / pa
    ).where(positive_pa)
    frame["college_SB_attempts"] = frame["college_SB"] + frame["college_CS"]
    attempts = frame["college_SB_attempts"]
    frame["college_SB_success_rate"] = (frame["college_SB"] / attempts).where(attempts.gt(0))
    frame["college_PA_under_100"] = pa.lt(100).astype(int)
    frame["college_PA_under_150"] = pa.lt(150).astype(int)
    return frame


def main():
    candidates = read_keyed(CANDIDATES_PATH, "candidate file")
    stats = read_keyed(STATS_PATH, "college-stats file")
    ages = read_keyed(AGES_PATH, "age file")
    outcomes = read_keyed(OUTCOMES_PATH, "outcomes file")

    candidate_required = {
        "player_name", "mlb_person_id", "draft_round", "college", "position",
        "team_SOS", "team_SOS_rank", "team_SOS_source_team",
    }
    stats_required = set(COUNTING_STATS + RATE_STATS + ["player_name", "stat_source", "notes"])
    age_required = {"player_name", "age_at_draft", "birth_date", "draft_date"}
    outcome_required = {"player_name", "made_mlb_ever", "mlb_career_pa", "match_status"}
    for label, frame, required in [
        ("candidate file", candidates, candidate_required),
        ("college-stats file", stats, stats_required),
        ("age file", ages, age_required),
        ("outcomes file", outcomes, outcome_required),
    ]:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing required columns: {sorted(missing)}")

    stat_columns = [
        "player_name", "mlb_person_id", "college", "college_season", *COUNTING_STATS,
        *RATE_STATS, "tbc_player_name", "stat_source", "notes",
    ]
    stats_aligned = align_to_candidates(candidates, stats, "college-stats file", stat_columns)
    assert_same_names(candidates, stats_aligned, "player_name", "stats")
    stats_aligned = stats_aligned.rename(columns={
        "player_name": "stats_player_name",
        "mlb_person_id": "stats_mlb_person_id",
        "college": "stats_college",
    })

    age_columns = ["player_name", "mlb_person_id", "birth_date", "draft_date", "age_at_draft", "age_source_url"]
    ages_aligned = align_to_candidates(candidates, ages, "age file", age_columns)
    assert_same_names(candidates, ages_aligned, "player_name", "age")
    ages_aligned = ages_aligned.rename(columns={
        "player_name": "age_player_name",
        "mlb_person_id": "age_mlb_person_id",
    })

    outcome_columns = [
        "player_name", "mlb_person_id", "mlb_career_pa", "made_mlb_ever",
        "mlb_debut_date", "api_position", "outcome_checked_date", "match_status",
        "outcome_source", "lookup_error", "manual_note",
    ]
    outcomes_aligned = align_to_candidates(candidates, outcomes, "outcomes file", outcome_columns)
    assert_same_names(candidates, outcomes_aligned, "player_name", "outcomes")
    outcomes_aligned = outcomes_aligned.rename(columns={
        "player_name": "outcome_player_name",
        "mlb_person_id": "outcome_mlb_person_id",
    })

    master = candidates.drop(columns=["api_record_json"], errors="ignore").merge(
        stats_aligned, on=KEYS, how="left", validate="one_to_one"
    ).merge(ages_aligned, on=KEYS, how="left", validate="one_to_one")

    for other_id in ["stats_mlb_person_id", "age_mlb_person_id"]:
        if other_id in master:
            left = pd.to_numeric(master["mlb_person_id"], errors="coerce")
            right = pd.to_numeric(master[other_id], errors="coerce")
            if left.isna().any() or right.isna().any() or not left.eq(right).all():
                raise ValueError(f"MLB person IDs disagree between candidates and {other_id}.")

    master = master.drop(columns=["stats_mlb_person_id", "age_mlb_person_id"], errors="ignore")
    master["player_key"] = (
        master["draft_year"].astype(str) + "_" + master["overall_pick"].astype(str)
    )
    master = add_derived_features(master)

    model = master.merge(outcomes_aligned, on=KEYS, how="left", validate="one_to_one")
    model = model.drop(columns=["outcome_mlb_person_id"], errors="ignore")

    # Core identity and outcome integrity checks.
    for column in ["age_at_draft", "team_SOS", "college_PA", "made_mlb_ever"]:
        if model[column].isna().any():
            raise ValueError(f"Missing values in required modeling column {column}.")
    model["made_mlb_ever"] = pd.to_numeric(model["made_mlb_ever"], errors="raise")
    if not model["made_mlb_ever"].isin([0, 1]).all():
        raise ValueError("made_mlb_ever must contain only 0 or 1.")
    model["made_mlb_ever"] = model["made_mlb_ever"].astype(int)
    if not model["team_SOS"].between(0, 1).all():
        raise ValueError("team_SOS values must be between 0 and 1.")
    if model["college_OPS"].isna().ne(model["college_PA"].eq(0)).any():
        raise ValueError("Missing OPS should occur only for zero-PA seasons, and all zero-PA OPS should be missing.")

    # Check that plate appearances reconcile to their components.
    pa_parts = ["college_AB", "college_BB", "college_HBP", "college_SF", "college_SH"]
    if model[pa_parts].isna().any().any():
        raise ValueError("Missing PA component values; review college-stat records before modeling.")
    reconstructed_pa = model[pa_parts].sum(axis=1)
    mismatch = model["college_PA"].ne(reconstructed_pa)
    if mismatch.any():
        check = model.loc[mismatch, ["draft_year", "overall_pick", "player_name", "college_PA", *pa_parts]].copy()
        check["reconstructed_PA"] = reconstructed_pa[mismatch]
        raise ValueError("Plate appearances do not reconcile:\n" + check.to_string(index=False))

    if len(model) != len(candidates):
        raise ValueError("Joined modeling row count does not match the candidate cohort.")
    if model.duplicated(KEYS).any():
        raise ValueError("Duplicate draft-year/pick rows found after joining.")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_PATH, index=False)
    model.to_csv(MODELING_PATH, index=False)

    print(f"Cohort rows: {len(candidates)}")
    print(f"Master rows (stats + age): {len(master)}")
    print(f"Modeling rows (plus MLB outcome): {len(model)}")
    print("Outcome counts:")
    print(model["made_mlb_ever"].value_counts().sort_index().to_string())
    print(f"Zero-PA players: {int(model['college_PA'].eq(0).sum())}")
    print(f"Missing OPS rows (zero PA): {int(model['college_OPS'].isna().sum())}")
    print(f"Saved master: {MASTER_PATH}")
    print(f"Saved modeling data: {MODELING_PATH}")
    if SCHOOL_REVIEW_PATH.exists():
        school_review = pd.read_csv(SCHOOL_REVIEW_PATH)
        print(f"School-match review rows still outside this provisional cohort: {len(school_review)}")


if __name__ == "__main__":
    main()
