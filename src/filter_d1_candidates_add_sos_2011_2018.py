"""Match 2011–2018 draft candidates to each season's D1 SOS table.

Matched player records are confirmed D1 position-player candidates with SOS.
Unmatched non-pitcher schools are written for review because they can be
non-D1 schools or naming variants; they are not silently discarded.
"""

from pathlib import Path
from datetime import datetime
import shutil
import re
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = PROJECT_ROOT / "data" / "raw" / "draft_candidates_2011_2018.csv"
SOS_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "team_sos_2011_2018.csv"
MATCHED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "d1_position_candidates_2011_2018.csv"
UNMATCHED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "d1_school_match_review_2011_2018.csv"
EXCLUDED_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "d1_cohort_exclusions_2011_2018.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; mlb-d1-draft-predictor/1.0)"}

# Equivalences among common school labels. Values are normalized by
# normalize_name(), so punctuation and spaces do not matter.
SCHOOL_ALIASES = {
    "alabamabirmingham": ["uab"],
    "brighamyoung": ["byu"],
    "floridaatlantic": ["fau", "floridaatlanticuniversity"],
    "floridainternational": ["fiu"],
    "georgiatech": ["georgiainstituteoftechnology"],
    "illinoischicago": ["uic", "universityofillinoisatchicago"],
    "louisianalafayette": ["louisiana", "ulafayette", "universityoflouisianaatlafayette"],
    "louisianamonroe": ["ulm", "universityoflouisianamonroe"],
    "mississippi": ["olemiss", "universityofmississippi"],
    "ncstate": ["northcarolinastate", "northcarolinastateuniversity"],
    "uncwilmington": ["uncw", "universityofnorthcarolinawilmington"],
    "penn": ["pennsylvania", "universityofpennsylvania"],
    "southernmississippi": ["southernmiss", "universityofsouthernmississippi"],
    "southerncalifornia": ["usc", "universityofsoutherncalifornia"],
    "texaschristian": ["tcu"],
    "ucirvine": ["universityofcaliforniairvine"],
    "ucriverside": ["universityofcaliforniariverside"],
    "ucsb": ["ucsantabarbara", "universityofcaliforniasantabarbara"],
    "ucsd": ["ucsanDiego", "universityofcaliforniasandiego"],
    "ucdavis": ["universityofcaliforniadavis"],
    "ucberkeley": ["california", "universityofcaliforniaberkeley"],
    "ucLosAngeles": ["ucla", "universityofcalifornialosangeles"],
    "ucSantaCruz": ["universityofcaliforniasantacruz"],
    "universityofnorthflorida": ["northflorida", "unf"],
    "virginiacommonwealth": ["vcu"],
    "wrightstateuniversity": ["wrightstate"],
    "louisianastate": ["lsu"],
    "calpolysanluisobispo": ["calpoly"],
    "centralflorida": ["ucf"],
    "uncchapelhill": ["northcarolina"],
    "universityofsouthalabama": ["southalabama"],
    "middletennesseestate": ["middletennessee"],
    "austinpeaystate": ["austinpeay"],
    "sunystonybrook": ["stonybrook"],
    "stjohnsny": ["saintjohns"],
    "stjohns": ["saintjohns"],
    "stbonaventure": ["saintbonaventure"],
    "stjosephspa": ["saintjosephs"],
    "stjosephs": ["saintjosephs"],
    "collegeofwilliamandmary": ["williamandmary"],
    "colofwilliamandmary": ["williamandmary"],
    "unccharlotte": ["charlotte"],
    "uncgreensboro": ["uncg"],
    "nevadalasvegas": ["unlv"],
    "nevadareno": ["nevada"],
    "southcarolinacolumbia": ["southcarolina"],
    "calstatesacramento": ["sacramentostate"],
    "calstlongbeach": ["longbeachstate"],
    "calstatelongbeach": ["longbeachstate"],
    "arkansasfayetteville": ["arkansas"],
    "uarkansasfayetteville": ["arkansas"],
    "illinoischampaign": ["illinois"],
    "universityofillinois": ["illinois"],
    "universityofmaineorono": ["maine"],
    "maineorono": ["maine"],
    "collegeofcharleston": ["charleston"],
    "jacksonvilleuniversity": ["jacksonville"],
    "belmontuniversity": ["belmont"],
    "bryantuniversity": ["bryant"],
    "libertyuniversity": ["liberty"],
    "troyuniversity": ["troy"],
    "saintlouisuniversity": ["saintlouis"],
    "santaclarauniversity": ["santaclara"],
    "sacredheartuniversity": ["sacredheart"],
    "loyolamarymountuniversity": ["loyolamarymount"],
    "universityofsandiego": ["sandiego"],
    "longislandu": ["longisland"],
    "universityofbuffalo": ["buffalo"],
    "ubuffalo": ["buffalo"],
    "semissouristate": ["southeastmissouristate"],
    "southernuaandm": ["southern"],
    "southernuam": ["southern"],
    "stephenfaustinstate": ["stephenfaustin"],
    "texaspanamerican": ["texaspanamerican", "utpa"],
    "universityoftexaspanamerican": ["texaspanamerican", "utpa"],
    "texasarlington": ["texasarlington", "uta"],
    "universityoftexasarlington": ["texasarlington", "uta"],
    "marylandbaltimorecounty": ["umbc"],
    "virginiamilitaryinstitute": ["vmi"],
    "massachusettslowell": ["umasslowell"],
    "floridagulfcoast": ["fgcu"],
    "miami": ["miamifl"],
    "stmarys": ["saintmaryscollege"],
    "saintmarys": ["saintmaryscollege"],
    "seattle": ["seattleuniversity"],
    # Common MLB draft tracker labels for D1 schools in the historical SOS tables.
    "alabamatuscaloosa": ["alabama"],
    "bowlinggreenstate": ["bowlinggreen"],
    "nebraskalincoln": ["nebraska"],
    "universityofnebraskalincoln": ["nebraska"],
    "sunybinghamton": ["binghamton"],
    "usmilitaryacademy": ["army", "unitedstatesmilitaryacademy", "westpoint", "usma"],
    "unitedstatesmilitaryacademy": ["army", "usmilitaryacademy", "westpoint", "usma"],
    "usnavalacademy": ["navy", "unitedstatesnavalacademy", "annapolis", "usna"],
    "unitedstatesnavalacademy": ["navy", "usnavalacademy", "annapolis", "usna"],
    "usairforceacademy": ["airforce", "unitedstatesairforceacademy", "usafa"],
    "unitedstatesairforceacademy": ["airforce", "usairforceacademy", "usafa"],
}

# The MLB draft feed's school/position fields are incorrect for these
# candidates. Keep their raw draft records, but exclude them from this
# D1 position-player cohort based on the verified draft-season school or role.
COHORT_EXCLUSIONS = {
    (2012, "yoenny gonzalez"): {
        "verified_college": "College of Central Florida",
        "verified_division": "NJCAA",
        "reason": "Draft feed says Central Florida, but player history identifies College of Central Florida (NJCAA), not UCF.",
        "verification_sources": "https://thebaseballcube.com/content/player/171321/; https://thebaseballcube.com/content/stats_college/2024~20208/",
    },
    (2012, "melvin mercedes"): {
        "verified_college": "College of Central Florida",
        "verified_division": "NJCAA",
        "reason": "Draft feed says Central Florida, but player history identifies College of Central Florida (NJCAA), not UCF.",
        "verification_sources": "https://thebaseballcube.com/content/player/171322/; https://thebaseballcube.com/content/stats_college/2024~20208/",
    },
    (2012, "terrell joyce"): {
        "verified_college": "Florida State College at Jacksonville",
        "verified_division": "NJCAA",
        "reason": "Draft feed says Florida State; verified school is Florida State College at Jacksonville (NJCAA).",
        "verification_sources": "https://www.baseball-reference.com/register/player.fcgi?id=joyce-000ter; https://www.thebaseballcube.com/content/college_alumni/21987/",
    },
    (2015, "ryan uhl"): {
        "verified_college": "Indiana University of Pennsylvania",
        "verified_division": "NCAA Division II",
        "reason": "Draft feed says Indiana; verified school is Indiana University of Pennsylvania (NCAA Division II).",
        "verification_sources": "https://www.thebaseballcube.com/content/player/197424/",
    },
    (2016, "shane carrier"): {
        "verified_college": "Fullerton College",
        "verified_division": "NJCAA",
        "reason": "Draft feed says Cal State Fullerton; verified school is Fullerton College (junior college/NJCAA).",
        "verification_sources": "https://www.milb.com/player/shane-carrier-669427; https://thebaseballcube.com/content/college_alumni/20330/",
    },
    (2011, "jack armstrong"): {
        "verified_college": "Vanderbilt University",
        "verified_division": "NCAA Division I; pitcher",
        "reason": "Draft records identify Armstrong as a right-handed pitcher; excluded from the position-player cohort despite the MLB Stats API 1B label.",
        "verification_sources": "https://www.baseball-reference.com/draft/index.fcgi?draft_type=junreg&query_type=franch_year&team_ID=HOU&year_ID=2011; https://vucommodores.com/sports/baseball/roster/player/jack-armstrong",
    },
}


def normalize_name(value):
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def simplify_school_name(value):
    """Remove common institution wording for unique exact-name matches."""
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\bst\.?\b", "saint", text)
    text = re.sub(r"\b(university|univ|college|col|of|the|at|u)\b", " ", text)
    text = text.replace("&", " and ")
    return re.sub(r"[^a-z0-9]", "", text)


def get_sos_table(year):
    url = f"https://www.warrennolan.com/baseball/{year}/sos"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        output = []
        for row in rows:
            cells = row.find_all(["td", "th"], recursive=False)
            if not cells:
                continue

            # Historical Warren Nolan pages use more than one header layout.
            # Identify data rows by their linked team name and decimal SOS value
            # instead of requiring a specific header row.
            team_cell = next(
                (
                    cell
                    for cell in cells
                    if cell.find("a")
                    and cell.find("a").get_text(" ", strip=True)
                ),
                None,
            )
            if team_cell is None:
                continue
            team = team_cell.find("a").get_text(" ", strip=True)
            trailing = cells[cells.index(team_cell) + 1 :]
            values = [cell.get_text(" ", strip=True) for cell in trailing]
            decimals = [
                (index, value)
                for index, value in enumerate(values)
                if re.fullmatch(r"0?\.\d+", value)
            ]
            if not team or not decimals:
                continue

            sos_index, sos_text = decimals[0]
            if not 0 <= float(sos_text) <= 1:
                continue

            rank = pd.NA
            rank_index = sos_index + 1
            if rank_index < len(values) and values[rank_index].isdigit():
                rank = int(values[rank_index])

            # Some seasons render opponent wins and losses in separate cells;
            # others combine them into a single W-L cell.
            after_rank = values[rank_index + 1 :]
            opponent_record = ""
            if after_rank:
                record_match = re.fullmatch(r"(\d{1,4})\s*-\s*(\d{1,4})", after_rank[0])
                if record_match:
                    opponent_record = f"{record_match.group(1)}-{record_match.group(2)}"
                elif len(after_rank) > 1 and after_rank[0].isdigit() and after_rank[1].isdigit():
                    opponent_record = f"{after_rank[0]}-{after_rank[1]}"

            opponent_win_pct = pd.NA
            if len(decimals) > 1:
                opponent_win_pct = float(decimals[1][1])

            output.append({
                "season": year,
                "source_team": team,
                "team_SOS": float(sos_text),
                "team_SOS_rank": rank,
                "opponent_record": opponent_record,
                "opponent_win_pct": opponent_win_pct,
                "source_url": url,
            })
        if output:
            return pd.DataFrame(output).drop_duplicates("source_team")

    # Fallback for pages whose visible rows are not in a standard HTML table.
    visible_text = soup.get_text(" ", strip=True)
    pattern = re.compile(
        r"(?P<team>[A-Za-zÀ-ÿ&().’'/-]+(?:\s+[A-Za-zÀ-ÿ&().’'/-]+){0,5})"
        r"\s+(?P<sos>0\.\d{4})\s+(?P<rank>\d{1,3})"
        r"\s+(?P<record>\d{1,4}\s*-\s*\d{1,4})\s+(?P<pct>0\.\d{4})"
    )
    output = []
    for match in pattern.finditer(visible_text):
        output.append({
            "season": year,
            "source_team": match.group("team").strip(),
            "team_SOS": float(match.group("sos")),
            "team_SOS_rank": int(match.group("rank")),
            "opponent_record": match.group("record").replace(" ", ""),
            "opponent_win_pct": float(match.group("pct")),
            "source_url": url,
        })
    if output:
        return pd.DataFrame(output).drop_duplicates("source_team")

    title = soup.title.get_text(" ", strip=True) if soup.title else "no title"
    raise ValueError(f"Could not parse {year} SOS table (page title: {title!r}).")


def make_alias_lookup(sos):
    lookup = {normalize_name(team): team for team in sos["source_team"]}
    for alias_key, alias_values in SCHOOL_ALIASES.items():
        group = [alias_key, *alias_values]
        matched_team = next((lookup[normalize_name(x)] for x in group if normalize_name(x) in lookup), None)
        if matched_team:
            for value in group:
                lookup[normalize_name(value)] = matched_team

    simplified = {}
    for team in sos["source_team"]:
        simplified.setdefault(simplify_school_name(team), set()).add(team)
    for key, teams in simplified.items():
        if key and len(teams) == 1:
            lookup[f"_simple_{key}"] = next(iter(teams))
    return lookup


def source_match(school, source_lookup):
    normalized = normalize_name(school)
    candidates = [normalized, *SCHOOL_ALIASES.get(normalized, [])]
    match = next((source_lookup[key] for key in candidates if key in source_lookup), None)
    if match:
        return match
    return source_lookup.get(f"_simple_{simplify_school_name(school)}")


def main():
    candidates = pd.read_csv(CANDIDATES_PATH)
    candidates["draft_year"] = pd.to_numeric(candidates["draft_year"], errors="raise").astype(int)
    candidates["draft_round"] = pd.to_numeric(candidates["draft_round"], errors="raise").astype(int)
    candidates["overall_pick"] = pd.to_numeric(candidates["overall_pick"], errors="raise").astype(int)

    position_players = candidates[
        ~candidates["position"].astype(str).str.upper().isin({"P", "LHP", "RHP", "SHP"})
    ].copy()
    position_players["position_source_note"] = (
        "MLB Stats API pick/person position; verify against draft-day position before final inclusion"
    )
    excluded_rows = []
    keep_mask = []
    for _, candidate in position_players.iterrows():
        exclusion = COHORT_EXCLUSIONS.get(
            (int(candidate["draft_year"]), str(candidate["player_name"]).strip().lower())
        )
        keep_mask.append(exclusion is None)
        if exclusion is not None:
            excluded_rows.append({**candidate.to_dict(), **exclusion})
    position_players = position_players.loc[keep_mask].copy()
    if MATCHED_OUTPUT_PATH.exists():
        backup_path = MATCHED_OUTPUT_PATH.with_name(
            f"{MATCHED_OUTPUT_PATH.stem}.before_cohort_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}{MATCHED_OUTPUT_PATH.suffix}"
        )
        shutil.copy2(MATCHED_OUTPUT_PATH, backup_path)
        print(f"Previous matched candidate file backed up: {backup_path}")
    sos_tables = []
    matched_parts = []
    unmatched_parts = []

    for year in range(2011, 2019):
        sos = get_sos_table(year)
        sos_tables.append(sos)
        year_players = position_players[position_players["draft_year"] == year].copy()
        source_lookup = make_alias_lookup(sos)
        year_players["team_SOS_source_team"] = year_players["college"].map(
            lambda school: source_match(school, source_lookup)
        )
        year_players["team_SOS"] = year_players["team_SOS_source_team"].map(
            sos.set_index("source_team")["team_SOS"]
        )
        year_players["team_SOS_rank"] = year_players["team_SOS_source_team"].map(
            sos.set_index("source_team")["team_SOS_rank"]
        )
        year_players["team_SOS_source_url"] = f"https://www.warrennolan.com/baseball/{year}/sos"

        matched = year_players[year_players["team_SOS"].notna()].copy()
        unmatched = year_players[year_players["team_SOS"].isna()].copy()
        matched_parts.append(matched)
        unmatched_parts.append(unmatched)
        print(
            f"{year}: {len(matched)} matched D1 candidates with non-pitcher API position; "
            f"{len(unmatched)} school/player records need review"
        )

        # Persist after each completed season so later source-page issues do
        # not discard prior years' successful results.
        partial_sos = pd.concat(sos_tables, ignore_index=True)
        partial_matched = pd.concat(matched_parts, ignore_index=True)
        partial_unmatched = pd.concat(unmatched_parts, ignore_index=True)
        SOS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        MATCHED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial_sos.to_csv(SOS_OUTPUT_PATH, index=False)
        partial_matched.to_csv(MATCHED_OUTPUT_PATH, index=False)
        partial_unmatched.to_csv(UNMATCHED_OUTPUT_PATH, index=False)

    all_sos = pd.concat(sos_tables, ignore_index=True)
    all_matched = pd.concat(matched_parts, ignore_index=True)
    all_unmatched = pd.concat(unmatched_parts, ignore_index=True)
    all_excluded = pd.DataFrame(excluded_rows)
    SOS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATCHED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_sos.to_csv(SOS_OUTPUT_PATH, index=False)
    all_matched.to_csv(MATCHED_OUTPUT_PATH, index=False)
    all_unmatched.to_csv(UNMATCHED_OUTPUT_PATH, index=False)
    EXCLUDED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_excluded.to_csv(EXCLUDED_OUTPUT_PATH, index=False)
    print(f"\nSOS rows saved: {len(all_sos)} -> {SOS_OUTPUT_PATH}")
    print(f"Matched candidates saved: {len(all_matched)} -> {MATCHED_OUTPUT_PATH}")
    print(f"School names for review: {all_unmatched['college'].nunique()} unique, {len(all_unmatched)} players -> {UNMATCHED_OUTPUT_PATH}")
    print(f"Verified cohort exclusions (school/role): {len(all_excluded)} -> {EXCLUDED_OUTPUT_PATH}")
    print("Review unmatched schools before treating the matched file as the complete D1 cohort.")
    print("The position field comes from the MLB Stats API record and still needs draft-day verification.")


if __name__ == "__main__":
    main()
