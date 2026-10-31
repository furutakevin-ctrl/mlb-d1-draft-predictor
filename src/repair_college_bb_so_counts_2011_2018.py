"""Repair BB and SO count fields in the earlier TBC export.

The earlier collector collapsed TBC's BB/SO count headers with BB%/SO%
headers. This script reads the original export, checks the rate evidence,
recovers TBC BB counts from PA components and SO counts from SO% * PA,
then writes a corrected CSV beside the original. The original is not changed.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "college_stats_2011_2018_first_pass.csv"
OUTPUT = ROOT / "data" / "raw" / "college_stats_2011_2018_first_pass_corrected.csv"

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing original stats file: {SOURCE}")
    frame = pd.read_csv(SOURCE)
    required = {
        "stat_source", "college_PA", "college_AB", "college_BB", "college_SO",
        "college_HBP", "college_SF", "college_SH",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tbc = frame["stat_source"].astype("string").str.contains(
        "thebaseballcube.com/content/stats_college", na=False
    )
    if not tbc.any():
        raise ValueError("No TBC college-season rows found; refusing to alter the file.")

    # Preserve the values currently occupying these columns for audit. TBC's
    # two-decimal BB%/SO% values were written there by the old header parser.
    frame["college_BB_pct_source"] = pd.NA
    frame["college_SO_pct_source"] = pd.NA
    frame.loc[tbc, "college_BB_pct_source"] = frame.loc[tbc, "college_BB"]
    frame.loc[tbc, "college_SO_pct_source"] = frame.loc[tbc, "college_SO"]

    numeric_cols = [
        "college_PA", "college_AB", "college_BB", "college_SO",
        "college_HBP", "college_SF", "college_SH",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise")

    pa = frame.loc[tbc, "college_PA"]
    ab = frame.loc[tbc, "college_AB"]
    hbp = frame.loc[tbc, "college_HBP"]
    sf = frame.loc[tbc, "college_SF"]
    sh = frame.loc[tbc, "college_SH"]
    bb_pct = pd.to_numeric(frame.loc[tbc, "college_BB_pct_source"], errors="raise")
    so_pct = pd.to_numeric(frame.loc[tbc, "college_SO_pct_source"], errors="raise")

    # PA = AB + BB + HBP + SF + SH, so the official source PA and counting
    # components provide the exact walk count. Confirm the source BB% agrees.
    bb_count = pa - ab - hbp - sf - sh
    if bb_count.lt(0).any() or bb_count.mod(1).ne(0).any():
        raise ValueError("Recovered BB counts are negative or non-integer.")
    bb_pct_rebuilt = (100 * bb_count / pa.where(pa.gt(0))).round(2)
    if not bb_pct_rebuilt.eq(bb_pct).all():
        bad = frame.loc[tbc].loc[bb_pct_rebuilt.ne(bb_pct),
            ["draft_year", "overall_pick", "player_name"]].copy()
        bad["source_BB_pct"] = bb_pct[bb_pct_rebuilt.ne(bb_pct)].values
        bad["recovered_BB_pct"] = bb_pct_rebuilt[bb_pct_rebuilt.ne(bb_pct)].values
        raise ValueError("Recovered BB counts do not match TBC BB%:\n" + bad.to_string(index=False))

    # TBC percentages are displayed to two decimals; rounding the implied
    # count recovers each integer strikeout count within the source precision.
    so_count = (so_pct * pa / 100).round()
    so_pct_rebuilt = 100 * so_count / pa.where(pa.gt(0))
    if ((so_pct_rebuilt - so_pct).abs() > 0.0051).any():
        bad_mask = (so_pct_rebuilt - so_pct).abs() > 0.0051
        bad = frame.loc[tbc].loc[bad_mask, ["draft_year", "overall_pick", "player_name"]].copy()
        bad["source_SO_pct"] = so_pct[bad_mask].values
        bad["recovered_SO_pct"] = so_pct_rebuilt[bad_mask].values
        raise ValueError("Recovered SO counts do not match TBC SO% within rounding precision:\n" + bad.to_string(index=False))
    if so_count.lt(0).any() or (so_count - so_count.round()).abs().gt(1e-9).any():
        raise ValueError("Recovered SO counts are negative or non-integer.")

    frame.loc[tbc, "college_BB"] = bb_count
    frame.loc[tbc, "college_SO"] = so_count

    # Check PA reconciliation for all TBC rows after the repair.
    reconstructed = (
        frame.loc[tbc, "college_AB"] + frame.loc[tbc, "college_BB"]
        + frame.loc[tbc, "college_HBP"] + frame.loc[tbc, "college_SF"]
        + frame.loc[tbc, "college_SH"]
    )
    if not reconstructed.eq(frame.loc[tbc, "college_PA"]).all():
        raise ValueError("PA totals still fail to reconcile after repair.")

    frame.to_csv(OUTPUT, index=False)
    print(f"Rows: {len(frame)}")
    print(f"TBC rows repaired: {int(tbc.sum())}")
    print(f"Non-TBC manual rows retained: {int((~tbc).sum())}")
    print("TBC BB and SO counts are integer-valued.")
    print("All TBC PA values reconcile to AB + BB + HBP + SF + SH.")
    print(f"Original left unchanged: {SOURCE}")
    print(f"Corrected file: {OUTPUT}")

if __name__ == "__main__":
    main()
