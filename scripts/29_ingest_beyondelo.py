"""
29_ingest_beyondelo.py — Pull Elo history from beyondELO.com.

beyondELO has a free API at api.beyondelo.com with:
- /api/teams — list of all teams (1271)
- /api/teams/<id>/matches — per-team match history (paginated, 100/page)
  Each match has home_elo_before, away_elo_before, home_elo_after, away_elo_after
  This gives us the FULL Elo history per team per match!

For our 5 leagues × ~25 teams/league = ~125 teams, we need ~1500 API calls
(100 matches per page × ~12 pages per team = ~12 calls/team).

Data goes back to at least 2016 (based on early tests), giving us the full
8-season backtest coverage we need.

We pull all matches for each team, extract the (team_id, match_date, elo_before)
as our point-in-time Elo rating per team per date.

beyondELO uses a sophisticated Glicko-2-style variant of Elo with:
- Match-importance-weighted K factor (UCL=30, UEL=20, etc.)
- Home-field advantage as +HFA on home team's rating
- Goal-margin scaling
- Rating deviation (uncertainty) — teams with high RD move faster
- Cross-league play included (UCL, UEL, UECL)

This is a much richer signal than our self-computed Elo which only saw
domestic-league matches.
"""
import os
import sys
import json
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from soccer_model.config import LEAGUES, ELO_DIR, DATA_PROCESSED

BEYONDELO_API = "https://api.beyondelo.com"
MATCHES_PER_PAGE = 100
MAX_PAGES_PER_TEAM = 50  # 50 pages × 100 = 5000 matches per team (plenty)


def get_teams():
    """Get list of all teams from beyondELO."""
    cache_path = Path("/tmp/beyondelo_teams.json")
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)["data"]
    print("Fetching team list from beyondELO...")
    r = requests.get(f"{BEYONDELO_API}/api/teams", timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    with open(cache_path, "w") as f:
        json.dump({"data": data}, f)
    print(f"  Got {len(data)} teams")
    return data


def get_team_matches(team_id, max_pages=MAX_PAGES_PER_TEAM):
    """Get all matches for a team, paginated."""
    all_matches = []
    for page in range(1, max_pages + 1):
        url = f"{BEYONDELO_API}/api/teams/{team_id}/matches?limit=1000&page={page}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                break
            matches = r.json().get("matches", [])
            if not matches:
                break
            all_matches.extend(matches)
            if len(matches) < 100:
                break
            time.sleep(0.1)  # be polite
        except Exception as e:
            print(f"    [error] page {page}: {e}")
            break
    return all_matches


# Map our canonical team names → beyondELO team names
# We'll match by checking each team in our matches_enriched against beyondELO's list
def build_team_name_map(beyondelo_teams, our_teams):
    """Match our canonical team names to beyondELO team names."""
    # Build lookup by lowercased name (with Unicode normalization for accents)
    import unicodedata
    def normalize(s):
        """Normalize string: lowercase + strip accents for matching."""
        s = s.lower().strip()
        # NFD decomposition then remove combining marks
        s = unicodedata.normalize('NFD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
        return s

    be_by_name = {normalize(t["name"]): t for t in beyondelo_teams}
    be_by_slug = {normalize(t.get("slug", "")): t for t in beyondelo_teams if t.get("slug")}

    # Common abbreviation → full name mappings for EPL teams
    # These are common in football-data.co.uk but expanded in beyondELO
    # Common abbreviation → full name mappings
    # Built from thorough audit of all 154 teams in our dataset
    # Handles: abbreviations (Man City → Manchester City), accented chars
    # (Alaves → Alavés), suffixes (Hull → Hull City), prefixes, etc.
    abbreviation_map = {
        # EPL — abbreviations and expanded names
        "man city": "manchester city",
        "man united": "manchester united",
        "newcastle": "newcastle united",
        "nott'm forest": "nottingham forest",
        "nottm forest": "nottingham forest",
        "west brom": "west bromwich albion",
        "wolves": "wolverhampton wanderers",
        "tottenham": "tottenham hotspurs",
        "luton": "luton town",
        "leicester": "leicester city",
        "bournemouth": "afc bournemouth",
        "brentford": "brentford fc",
        "huddersfield": "huddersfield town",
        "hull": "hull city",
        "cardiff": "cardiff city",
        "norwich": "norwich city",
        "sunderland": "sunderland afc",
        "stoke": "stoke city",
        "leeds": "leeds united",
        # Serie A
        "milan": "ac milan",
        "parma": "parma calcio 1913",
        "spal": "spal 2013",
        "spal 2013": "spal 2013",
        "benevento": "benevento calcio",
        "frosinone": "frosinone calcio",
        "crotone": "fc crotone",
        "lecce": "us lecce",
        "monza": "ac monza",
        "salernitana": "us salernitana 1919",
        "spezia": "spezia calcio",
        "cremonese": "us cremonese",
        "venezia": "venezia fc",
        # Bundesliga
        "bielefeld": "arminia bielefeld",
        "dortmund": "borussia dortmund",
        "ein frankfurt": "frankfurt",
        "fc koln": "köln",
        "fortuna dusseldorf": "fortuna duesseldorf",
        "greuther furth": "spvgg greuther fürth",
        "hamburg": "hamburger sv",
        "hannover": "hannover 96",
        "heidenheim": "fc heidenheim",
        "hertha": "hertha berlin",
        "leverkusen": "bayer leverkusen",
        "m'gladbach": "mönchengladbach",
        "mgladbach": "mönchengladbach",
        "mainz": "mainz 05",
        "nurnberg": "nürnberg",
        "rb leipzig": "rasenballsport leipzig",
        "stuttgart": "vfb stuttgart",
        "werder bremen": "bremen",
        "darmstadt": "sv darmstadt 98",
        "ingolstadt": "fc ingolstadt 04",
        "paderborn": "sc paderborn 07",
        # La Liga — accented characters
        "ath bilbao": "athletic club",
        "ath madrid": "atletico madrid",
        "betis": "real betis",
        "celta": "celta vigo",
        "espanol": "espanyol",
        "huesca": "sd huesca",
        "la coruna": "deportivo",  # may not have data — falls back to self-comp
        "sociedad": "real sociedad",
        "sp gijon": "sporting gijón",
        "valladolid": "real valladolid",
        "vallecano": "rayo vallecano",
        "alaves": "alavés",
        "almeria": "ud almería",
        "cadiz": "cádiz cf",
        "eibar": "sd eibar",
        "elche": "elche cf",
        "leganes": "cd leganés",
        "malaga": "málaga",
        # Ligue 1 — accented characters
        "bastia": "sc bastia",
        "clermont": "clermont foot 63",
        "paris sg": "paris saint-germain",
        "st etienne": "st-étienne",
        "nimes": "nîmes",
        "dijon": "dijon fco",
        "amiens": "amiens sc",
        "ajaccio": "ac ajaccio",
        "le havre": "le havre ac",
        "lorient": "lorient",  # exists but may have 0 matches
    }

    mapping = {}
    unmatched = []
    for our_team in our_teams:
        our_norm = normalize(our_team)
        # Try abbreviation map first (also normalized)
        if our_norm in {normalize(k): v for k, v in abbreviation_map.items()}:
            # Find the expanded form
            expanded = None
            for k, v in abbreviation_map.items():
                if normalize(k) == our_norm:
                    expanded = v
                    break
            if expanded:
                exp_norm = normalize(expanded)
                if exp_norm in be_by_name:
                    mapping[our_team] = be_by_name[exp_norm]
                    continue
                if exp_norm in be_by_slug:
                    mapping[our_team] = be_by_slug[exp_norm]
                    continue
        # Direct match (normalized)
        if our_norm in be_by_name:
            mapping[our_team] = be_by_name[our_norm]
            continue
        # Slug match (normalized)
        if our_norm in be_by_slug:
            mapping[our_team] = be_by_slug[our_norm]
            continue
        # Fuzzy: try removing "FC", "AFC", "CF", common prefixes
        for prefix in ["fc ", "afc ", "cf ", "1. ", "club ", "sc ", "ac "]:
            cleaned = our_norm.replace(prefix, "").strip()
            if cleaned in be_by_name:
                mapping[our_team] = be_by_name[cleaned]
                break
        else:
            # Try suffix
            for suffix in [" fc", " cf", " afc", " club", " city", " united", " town"]:
                cleaned = our_norm.replace(suffix, "").strip()
                if cleaned in be_by_name:
                    mapping[our_team] = be_by_name[cleaned]
                    break
            else:
                unmatched.append(our_team)
    return mapping, unmatched


def main():
    print("=" * 70)
    print("Phase 0 — Step 29: Pull Elo history from beyondELO.com")
    print("=" * 70)

    # Get all beyondELO teams
    be_teams = get_teams()
    print(f"beyondELO teams available: {len(be_teams)}")

    # Get our team list from matches_enriched
    matches = pd.read_csv(DATA_PROCESSED / "matches_enriched.csv", low_memory=False)
    our_teams = sorted(set(matches["home_team"].unique()) | set(matches["away_team"].unique()))
    our_teams = [t for t in our_teams if isinstance(t, str) and t]
    print(f"Our teams (from matches_enriched): {len(our_teams)}")

    # Build team name map
    mapping, unmatched = build_team_name_map(be_teams, our_teams)
    print(f"\nMapped: {len(mapping)}/{len(our_teams)}")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for t in unmatched:
            # Try partial match
            partials = [bt["name"] for bt in be_teams if t.lower().split()[0] in bt["name"].lower()]
            print(f"  '{t}' — possible matches: {partials[:3]}")

    # Save mapping
    mapping_path = ELO_DIR / "beyondelo_team_map.json"
    with open(mapping_path, "w") as f:
        json.dump({k: {"id": v["id"], "name": v["name"]} for k, v in mapping.items()}, f, indent=2)
    print(f"\nTeam mapping saved to: {mapping_path}")

    # For each mapped team, pull all matches
    print(f"\nPulling matches for {len(mapping)} teams...")
    all_elo_records = []
    for i, (our_team, be_team) in enumerate(mapping.items()):
        team_id = be_team["id"]
        team_name = be_team["name"]
        print(f"  [{i+1}/{len(mapping)}] {team_name} ({team_id[:8]}...) ", end="", flush=True)
        matches_data = get_team_matches(team_id)
        print(f"{len(matches_data)} matches")

        # Extract per-match Elo
        for m in matches_data:
            # For home matches, our team is home; for away, our team is away
            is_home = m["home_team_id"] == team_id
            if is_home:
                our_elo_before = m.get("home_elo_before")
                our_elo_after = m.get("home_elo_after")
                opp_id = m["away_team_id"]
                opp_name = m.get("away_team", {}).get("name", "")
            else:
                our_elo_before = m.get("away_elo_before")
                our_elo_after = m.get("away_elo_after")
                opp_id = m["home_team_id"]
                opp_name = m.get("home_team", {}).get("name", "")

            if our_elo_before is None:
                continue

            all_elo_records.append({
                "team": our_team,
                "team_id": team_id,
                "team_name_be": team_name,
                "date": m["match_date"],
                "is_home": is_home,
                "elo_before": our_elo_before,
                "elo_after": our_elo_after,
                "opp_id": opp_id,
                "opp_name": opp_name,
                "competition": m.get("competition", {}).get("code", ""),
                "season": m.get("season", ""),
                "rd_before": m.get("home_rd" if is_home else "away_rd"),
                "hfa": m.get("home_hfa" if is_home else "away_hfa"),
            })
        time.sleep(0.05)  # be polite

    if not all_elo_records:
        print("[fatal] No Elo data collected")
        return

    df = pd.DataFrame(all_elo_records)
    df["date"] = pd.to_datetime(df["date"])

    # Sort by team and date
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    # Save
    out_csv = ELO_DIR / "beyondelo_history.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("BEYONDELO INGESTION COMPLETE")
    print(f"  Total Elo records: {len(df):,}")
    print(f"  Teams: {df['team'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Output: {out_csv}")

    # Coverage per team
    print(f"\nRecords per team (sample):")
    for team in sorted(df["team"].unique())[:20]:
        sub = df[df["team"] == team]
        print(f"  {team:25s}: {len(sub):4d} matches, "
              f"{sub['date'].min().date()} → {sub['date'].max().date()}, "
              f"Elo range {sub['elo_before'].min():.0f}-{sub['elo_before'].max():.0f}")


if __name__ == "__main__":
    main()
