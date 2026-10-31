"""Collect final-draft-year D1 batting lines from The Baseball Cube (TBC).

The collector matches on school + season + player name. It never auto-assigns
fuzzy school or player matches; those are written to review files instead.
Fetched school-season pages are cached so reruns resume without refetching.
"""

from difflib import SequenceMatcher
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "processed" / "d1_position_candidates_2011_2018.csv"
SCHOOL_DIR_URL = "https://thebaseballcube.com/content/schools/ncaa-1/"
OVERRIDES = ROOT / "data" / "raw" / "tbc_school_id_overrides_2011_2018.csv"
OUT = ROOT / "data" / "raw" / "college_stats_2011_2018_first_pass.csv"
REVIEW = ROOT / "data" / "processed" / "college_stats_review_2011_2018.csv"
SCHOOL_REVIEW = ROOT / "data" / "processed" / "college_stats_school_map_review_2011_2018.csv"
MANUAL_OVERRIDES = ROOT / "data" / "raw" / "college_stats_manual_overrides_2011_2018.csv"
CACHE_DIR = ROOT / "data" / "raw" / "tbc_stats_cache_2011_2018"

BASE = "https://thebaseballcube.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MLB-D1-Draft-Predictor/1.0)"}
SLEEP_SECONDS = 0.7

TBC_TO_PROJECT = {
    "g": "college_games", "pa": "college_PA", "ab": "college_AB",
    "r": "college_R", "h": "college_H", "2b": "college_2B",
    "3b": "college_3B", "hr": "college_HR", "rbi": "college_RBI",
    "tb": "college_TB", "bb": "college_BB", "so": "college_SO",
    "hbp": "college_HBP", "gdp": "college_GDP", "sf": "college_SF",
    "sh": "college_SH", "sb": "college_SB", "cs": "college_CS",
    "avg": "college_BA", "obp": "college_OBP", "slg": "college_SLG",
    "ops": "college_OPS",
}
STAT_COLUMNS = list(TBC_TO_PROJECT.values())


def normalize(value):
    value = unicodedata.normalize("NFKD", str(value or ""))
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    parts = re.findall(r"[a-z0-9]+", value)
    # TBC often omits suffixes that appear in the MLB draft roster.
    while parts and parts[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        parts.pop()
    return "".join(parts)


def normalize_header(value):
    """Normalize TBC headers while keeping rate columns distinct from counts."""
    raw = str(value or "").strip().lower()
    if raw.endswith("%"):
        return normalize(raw[:-1]) + "_pct"
    return normalize(raw)


# Confirmed spelling variant in the cited TBC team-season table.
PLAYER_NAME_ALIASES = {
    "danieldelacalle": {"dannydelacalle"},
    "robertkral": {"robkral"},
    "michaelgarza": {"mikegarza"},
    "michaelbenjamin": {"mikebenjamin"},
    "dylaburdeaux": {"dylanburdeaux"},
}


def read_manual_overrides():
    if not MANUAL_OVERRIDES.exists():
        return {}
    frame = pd.read_csv(MANUAL_OVERRIDES, dtype=str).fillna("")
    needed = {"draft_year", "player_name", "stat_source", "notes", *STAT_COLUMNS}
    if not needed.issubset(frame.columns):
        raise ValueError(f"{MANUAL_OVERRIDES} must contain columns {sorted(needed)}")
    overrides = {}
    for _, row in frame.iterrows():
        key = (int(row["draft_year"]), normalize(row["player_name"]))
        overrides[key] = {
            "stats": {column: numeric(row[column]) for column in STAT_COLUMNS},
            "stat_source": row["stat_source"],
            "notes": row["notes"],
        }
    return overrides


def numeric(value):
    text = str(value or "").strip().replace(",", "")
    if text in {"", "-", "--", "n/a", "N/A", "None"}:
        return None
    text = text.rstrip("*%")
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return None


def fetch_html(url):
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {error}")


def school_directory():
    soup = BeautifulSoup(fetch_html(SCHOOL_DIR_URL), "html.parser")
    schools, aliases = {}, {}
    id_pattern = re.compile(r"/content/college_history/(\d+)/?")
    for row in soup.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        for anchor in row.find_all("a", href=True):
            match = id_pattern.search(anchor["href"])
            if not match:
                continue
            tbc_id = match.group(1)
            short_name = anchor.get_text(" ", strip=True)
            cell_text = [cell.get_text(" ", strip=True) for cell in cells]
            full_name = cell_text[-2] if len(cell_text) >= 2 else short_name
            schools[tbc_id] = {"tbc_id": tbc_id, "short_name": short_name,
                               "full_name": full_name}
            # The directory row also contains nickname, conference, location,
            # and year; only school-name fields are valid aliases.
            for label in {short_name, full_name}:
                key = normalize(label)
                if key:
                    aliases.setdefault(key, set()).add(tbc_id)
    if not schools:
        raise RuntimeError("Could not read school IDs from the TBC Division I directory.")
    return schools, aliases


def read_overrides():
    if not OVERRIDES.exists():
        return {}
    frame = pd.read_csv(OVERRIDES, dtype=str).fillna("")
    needed = {"candidate_college", "tbc_school_id"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"{OVERRIDES} must contain columns {sorted(needed)}")
    return {normalize(row.candidate_college): row.tbc_school_id.strip()
            for row in frame.itertuples() if row.candidate_college.strip() and row.tbc_school_id.strip()}


def parse_batting_table(html):
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        header = None
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            names = [normalize_header(cell.get_text(" ", strip=True)) for cell in cells]
            if "player" in names and "pa" in names and "ab" in names:
                header = names
                break
        if not header:
            continue
        players = []
        for tr in table.find_all("tr"):
            cells = tr.find_all("td", recursive=False)
            if len(cells) != len(header):
                continue
            raw = {header[i]: cells[i].get_text(" ", strip=True) for i in range(len(header))}
            player_name = raw.get("player", "").strip()
            # Skip totals and other non-player table rows.
            if not player_name or normalize(player_name) in {"totals", "average"}:
                continue
            players.append({"player_name": player_name, **raw})
        if players:
            return players
    raise RuntimeError("No TBC batting table with player, PA, and AB columns was found.")


def load_team_season(year, tbc_id):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{year}_{tbc_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"{BASE}/content/stats_college/{year}~{tbc_id}/"
    players = parse_batting_table(fetch_html(url))
    payload = {"source_url": url, "players": players}
    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    time.sleep(SLEEP_SECONDS)
    return payload


def main():
    candidates = pd.read_csv(CANDIDATES).fillna("")
    schools, directory_aliases = school_directory()
    overrides = read_overrides()
    manual_player_overrides = read_manual_overrides()
    college_values = sorted(candidates["college"].astype(str).unique())
    school_ids = {}
    school_map_review = []

    for college in college_values:
        key = normalize(college)
        if key in overrides:
            school_ids[key] = overrides[key]
            continue
        ids = directory_aliases.get(key, set())
        if len(ids) == 1:
            school_ids[key] = next(iter(ids))
            continue
        suggestions = sorted(
            ((max(SequenceMatcher(None, key, normalize(s["short_name"])).ratio(),
                  SequenceMatcher(None, key, normalize(s["full_name"])).ratio()), sid, s)
             for sid, s in schools.items()), reverse=True
        )[:5]
        school_map_review.append({
            "candidate_college": college,
            "reason": "no_exact_directory_match" if not ids else "multiple_exact_matches",
            "suggested_tbc_school_ids": "; ".join(item[1] for item in suggestions),
            "suggested_tbc_school_names": "; ".join(item[2]["short_name"] for item in suggestions),
            "tbc_school_id_override": "",
        })

    if school_map_review:
        SCHOOL_REVIEW.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(school_map_review).to_csv(SCHOOL_REVIEW, index=False)
        print(f"School names needing mapping: {len(school_map_review)}")
        print(f"Review: {SCHOOL_REVIEW}")
        print(f"Add confirmed mappings to: {OVERRIDES}")

    result_rows, review_rows = [], []
    groups = candidates.groupby(["draft_year", "college"], sort=True, dropna=False)
    total_groups = len(groups)
    for group_number, ((year, college), group) in enumerate(groups, start=1):
        key = normalize(college)
        tbc_id = school_ids.get(key)
        if not tbc_id:
            for _, row in group.iterrows():
                review_rows.append({**row.to_dict(), "review_reason": "school_requires_mapping", "stats_source_url": ""})
            continue
        url = f"{BASE}/content/stats_college/{int(year)}~{tbc_id}/"
        try:
            payload = load_team_season(int(year), tbc_id)
            player_rows = payload["players"]
        except Exception as exc:
            for _, row in group.iterrows():
                review_rows.append({**row.to_dict(), "review_reason": f"source_fetch_or_parse_error: {exc}", "stats_source_url": url})
            continue

        by_name = {}
        for player in player_rows:
            by_name.setdefault(normalize(player["player_name"]), []).append(player)
        for _, candidate in group.iterrows():
            candidate_key = normalize(candidate["player_name"])
            manual_key = (int(year), candidate_key)
            if manual_key in manual_player_overrides:
                manual = manual_player_overrides[manual_key]
                result_rows.append({
                    "draft_year": int(year), "draft_round": candidate.get("draft_round"),
                    "overall_pick": candidate.get("overall_pick"), "player_name": candidate["player_name"],
                    "mlb_person_id": candidate.get("mlb_person_id"), "college": college,
                    "college_season": int(year), "tbc_player_name": "manual source record",
                    **manual["stats"], "stat_source": manual["stat_source"],
                    "notes": manual["notes"],
                })
                continue
            candidate_keys = {candidate_key, *PLAYER_NAME_ALIASES.get(candidate_key, set())}
            matches = [player for key in candidate_keys for player in by_name.get(key, [])]
            conflicted_fields = set()
            # Collapse repeated source rows only when their batting data agree.
            if len(matches) > 1:
                values_by_column = {
                    column: {numeric(player.get(column, "")) for player in matches}
                    for column in TBC_TO_PROJECT
                }
                signatures = {tuple(numeric(player.get(column, "")) for column in TBC_TO_PROJECT)
                              for player in matches}
                positive_pa = [player for player in matches if (numeric(player.get("pa", "")) or 0) > 0]
                if len(signatures) == 1:
                    matches = matches[:1]
                elif all(len(values) == 1 or column == "gdp" for column, values in values_by_column.items()):
                    conflicted_fields = {column for column, values in values_by_column.items() if len(values) > 1}
                    matches = matches[:1]
                elif len(positive_pa) == 1 and all(
                    (numeric(player.get("pa", "")) or 0) == 0 for player in matches if player is not positive_pa[0]
                ):
                    matches = positive_pa
            if len(matches) != 1:
                reason = "player_name_not_found" if not matches else "duplicate_player_name_in_team_table"
                review_rows.append({**candidate.to_dict(), "review_reason": reason, "stats_source_url": url})
                continue
            source_player = matches[0]
            stats = {project_col: numeric(source_player.get(source_col, ""))
                     for source_col, project_col in TBC_TO_PROJECT.items()}
            for source_column in conflicted_fields:
                stats[TBC_TO_PROJECT[source_column]] = None
            note = "TBC team-season batting table; matched after punctuation, diacritic, suffix, and confirmed alias normalization."
            if conflicted_fields:
                note += " Duplicate TBC rows disagreed on " + ", ".join(sorted(conflicted_fields)) + "; conflicting value left missing."
            result_rows.append({
                "draft_year": int(year), "draft_round": candidate.get("draft_round"),
                "overall_pick": candidate.get("overall_pick"), "player_name": candidate["player_name"],
                "mlb_person_id": candidate.get("mlb_person_id"), "college": college,
                "college_season": int(year), "tbc_player_name": source_player["player_name"],
                **stats, "stat_source": payload["source_url"],
                "notes": note,
            })
        if group_number % 25 == 0 or group_number == total_groups:
            print(f"School-seasons processed: {group_number}/{total_groups}; matched players: {len(result_rows)}; review rows: {len(review_rows)}")
            OUT.parent.mkdir(parents=True, exist_ok=True)
            REVIEW.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result_rows).to_csv(OUT, index=False)
            pd.DataFrame(review_rows).to_csv(REVIEW, index=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(OUT, index=False)
    pd.DataFrame(review_rows).to_csv(REVIEW, index=False)
    print(f"Candidates: {len(candidates)}")
    print(f"Stats matched: {len(result_rows)}")
    print(f"Rows needing review: {len(review_rows)}")
    print(f"Saved stats: {OUT}")
    print(f"Saved review: {REVIEW}")
    print("Review school aliases and player-name mismatches before using these stats in modeling.")


if __name__ == "__main__":
    main()
