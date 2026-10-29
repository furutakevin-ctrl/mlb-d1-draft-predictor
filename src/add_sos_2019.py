"""Attach the 2019 Division I team strength-of-schedule value to each player."""

from pathlib import Path
import re
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "players_2019_master.csv"
SOS_OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "team_sos_2019.csv"
UNMATCHED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "team_sos_unmatched_2019.csv"
)
SOS_URL = "https://www.warrennolan.com/baseball/2019/sos"


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", text)


# Alternate school labels used by the player-stat source.
SCHOOL_ALIASES = {
    "alabamabirmingham": ["uab"],
    "brighamyoung": ["byu"],
    "campbelluniversity": ["campbell"],
    "floridaatlantic": ["fau", "floridaatlanticuniversity"],
    "floridainternational": ["fiu"],
    "georgewashingtonuniversity": ["georgewashington"],
    "grandcanyonuniversity": ["grandcanyon"],
    "illinoischicago": ["uic"],
    "universityofillinoisatchicago": ["uic"],
    "louisianalafayette": ["louisiana"],
    "universityoflouisianalafayette": ["louisiana"],
    "louisianamonroe": ["ulm"],
    "universityoflouisianamonroe": ["ulm"],
    "libertyuniversity": ["liberty"],
    "loyolamarymount": ["loyolamarymount"],
    "middletennesseestate": ["middletennessee"],
    "mississippi": ["olemiss"],
    "ncstate": ["northcarolinastate"],
    "uncwilmington": ["uncw"],
    "penn": ["pennsylvania"],
    "saintmaryscollegeofcalifornia": ["saintmaryscollege"],
    "stmarys": ["saintmaryscollege"],
    "southernmississippi": ["southernmiss"],
    "southerncalifornia": ["usc"],
    "texaschristian": ["tcu"],
    "universityofcaliforniairvine": ["ucirvine"],
    "universityofcaliforniariverside": ["ucriverside"],
    "universityofcaliforniasantabarbara": ["ucsantabarbara"],
    "universityofnorthflorida": ["northflorida"],
    "universityofsandiego": ["sandiego"],
    "virginiacommonwealth": ["vcu"],
    "wrightstateuniversity": ["wrightstate"],
}


def fetch_sos_table():
    response = requests.get(
        SOS_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; mlb-d1-draft-predictor/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for header_row in rows:
            headers = [
                cell.get_text(" ", strip=True).lower()
                for cell in header_row.find_all(["th", "td"], recursive=False)
            ]
            if "team" not in headers or "sos" not in headers:
                continue

            output = []
            for row in rows:
                cells = row.find_all(["td", "th"], recursive=False)
                if not cells:
                    continue

                team_cell = next((cell for cell in cells if cell.find("a")), None)
                if team_cell is None:
                    continue
                team_link = team_cell.find("a")
                team = team_link.get_text(" ", strip=True)
                team_cell_index = cells.index(team_cell)
                trailing = cells[team_cell_index + 1 :]
                decimal_cells = []
                for index, cell in enumerate(trailing):
                    value = cell.get_text(" ", strip=True)
                    if re.fullmatch(r"0?\.\d+", value):
                        decimal_cells.append((index, value))
                if not decimal_cells:
                    continue
                sos_index, sos_text = decimal_cells[0]
                sos_value = float(sos_text)
                if not team or not 0 <= sos_value <= 1:
                    continue

                rank = pd.NA
                rank_index = sos_index + 1
                if rank_index < len(trailing):
                    rank_text = trailing[rank_index].get_text(" ", strip=True)
                    if rank_text.isdigit():
                        rank = int(rank_text)

                record = ""
                record_index = rank_index + 1
                if record_index < len(trailing):
                    record = trailing[record_index].get_text(" ", strip=True)

                opponent_pct = pd.NA
                opponent_pct_index = record_index + 1
                if opponent_pct_index < len(trailing):
                    try:
                        opponent_pct = float(
                            trailing[opponent_pct_index].get_text(" ", strip=True)
                        )
                    except ValueError:
                        pass

                output.append(
                    {
                        "season": 2019,
                        "source_team": team,
                        "team_SOS": sos_value,
                        "team_SOS_rank": rank,
                        "opponent_record": record,
                        "opponent_win_pct": opponent_pct,
                        "source_url": SOS_URL,
                    }
                )

            if output:
                return pd.DataFrame(output)

    # Some versions of the site render the rows without a conventional table.
    # Parse the same visible columns from the page text as a fallback.
    visible_text = soup.get_text(" ", strip=True)
    header_match = re.search(
        r"Opp\s+Win\s+Percent",
        visible_text,
        flags=re.IGNORECASE,
    )
    if header_match:
        visible_text = visible_text[header_match.end() :]

    row_pattern = re.compile(
        r"(?P<team>[A-Za-zÀ-ÿ&().’'/-]+(?:\s+[A-Za-zÀ-ÿ&().’'/-]+){0,5})"
        r"\s+(?P<sos>0\.\d{4})\s+(?P<rank>\d{1,3})"
        r"\s+(?P<record>\d{1,4}\s*-\s*\d{1,4})"
        r"\s+(?P<opponent_pct>0\.\d{4})"
    )
    text_rows = []
    for match in row_pattern.finditer(visible_text):
        text_rows.append(
            {
                "season": 2019,
                "source_team": match.group("team").strip(),
                "team_SOS": float(match.group("sos")),
                "team_SOS_rank": int(match.group("rank")),
                "opponent_record": match.group("record").replace(" ", ""),
                "opponent_win_pct": float(match.group("opponent_pct")),
                "source_url": SOS_URL,
            }
        )
    if text_rows:
        return pd.DataFrame(text_rows)

    page_title = soup.title.get_text(" ", strip=True) if soup.title else "no title"
    raise ValueError(
        "Could not parse the 2019 SOS rows from the source page "
        f"(received page title: {page_title!r})."
    )


master = pd.read_csv(MASTER_PATH)
school_column = "stats_college" if "stats_college" in master else "college"
if school_column not in master:
    raise ValueError("Master data is missing a college name column.")

sos = fetch_sos_table()
sos.to_csv(SOS_OUTPUT_PATH, index=False)
source_lookup = {
    normalize_name(row.source_team): row
    for row in sos.itertuples(index=False)
}


def source_match(school):
    normalized = normalize_name(school)
    candidates = [normalized, *SCHOOL_ALIASES.get(normalized, [])]
    return next(
        (source_lookup[key] for key in candidates if key in source_lookup),
        None,
    )


master["_sos_match"] = master[school_column].map(source_match)
missing = (
    master.loc[master["_sos_match"].isna(), ["player_name", school_column]]
    .drop_duplicates()
)

if not missing.empty:
    missing.to_csv(UNMATCHED_PATH, index=False)
    print(f"SOS table rows downloaded: {len(sos)}")
    print(f"Unmatched colleges: {missing[school_column].nunique()}")
    print(missing.to_string(index=False))
    raise ValueError("Resolve the listed college-name matches before modeling.")

for column in [
    "team_SOS",
    "team_SOS_rank",
    "team_SOS_source_team",
    "team_SOS_source_url",
]:
    master = master.drop(columns=column, errors="ignore")

master["_sos_match"] = master[school_column].map(source_match)
master["team_SOS"] = master["_sos_match"].map(lambda row: row.team_SOS)
master["team_SOS_rank"] = master["_sos_match"].map(
    lambda row: row.team_SOS_rank
)
master["team_SOS_source_team"] = master["_sos_match"].map(
    lambda row: row.source_team
)
master["team_SOS_source_url"] = SOS_URL
master = master.drop(columns=["_sos_match"])
master.to_csv(MASTER_PATH, index=False)
UNMATCHED_PATH.unlink(missing_ok=True)

print(f"Players enriched: {len(master)}")
print(f"Unique colleges enriched: {master[school_column].nunique()}")
print(f"Saved team SOS source table: {SOS_OUTPUT_PATH}")
print(f"Updated master data: {MASTER_PATH}")
