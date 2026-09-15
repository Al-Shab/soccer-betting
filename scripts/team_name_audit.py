"""
team_name_audit.py — Comprehensive test that team names from ALL data sources
correctly normalize to teams that exist in our internal model.

Sources tested:
1. Internal model (DC state attack dict) — the "ground truth" team names
2. The Odds API — live odds for upcoming matches (5 leagues)
3. SharpAPI — DraftKings + FanDuel odds (5 leagues × 2 books)
4. football-data.org — match results API (5 leagues)
5. beyondELO — Elo history (per-team records)
6. Understat — NPxG per-match data (via soccerdata)
7. football-data.co.uk — historical CSV files (5 leagues × 3 seasons)

For each source, we:
- Fetch the actual team names from the live API/source
- Run each name through LiveModel._normalize_team()
- Check if the normalized name exists in our model's attack dict
- Report any teams that DON'T match (these would silently produce wrong predictions)

Usage:
    python scripts/team_name_audit.py
"""
import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("/home/z/my-project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# === CONFIG ===
ODDS_API_KEYS = ["5da093bcd251f14f01da76e937c1996c", "305cab3b39fad0292f2f6bf23c705aed"]
SHARP_API_KEY = "sk_live_2CAdrXx97qM8YACRbgLM7C"
FD_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "525d8ca1e2a449d48f8dbbe2a7c652ca")

LEAGUES_ODDS_API = {
    "EPL": "soccer_epl", "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a", "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
}
SHARP_LEAGUE_SLUGS = {
    "EPL": "england_-_premier_league",
    "La Liga": "spain_-_la_liga",
    "Serie A": "italy_-_serie_a",
    "Bundesliga": "germany_-_bundesliga",
    "Ligue 1": "france_-_ligue_1",
}
FD_LEAGUE_CODES = {
    "EPL": "PL", "La Liga": "PD", "Serie A": "SA",
    "Bundesliga": "BL1", "Ligue 1": "FL1",
}

# SharpAPI market types to fetch
SHARP_MARKETS = ["moneyline", "total_goals", "point_spread"]


def get_internal_teams():
    """Load the internal model and return the set of teams it knows about."""
    from soccer_model.live_model import LiveModel
    model = LiveModel()
    internal_teams = set(model.attack.keys()) | set(model.team_features.keys())
    return model, internal_teams


# =============================================================================
# SOURCE 1: The Odds API
# =============================================================================
def fetch_odds_api_teams():
    """Fetch all team names from The Odds API across 5 leagues.
    
    Returns: dict {league: set of team names}
    """
    print("\n[1] Fetching team names from The Odds API...")
    teams_by_league = {}
    for league, sport_key in LEAGUES_ODDS_API.items():
        all_teams = set()
        for key in ODDS_API_KEYS:
            try:
                r = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                    params={"apiKey": key, "regions": "eu", "markets": "h2h",
                            "oddsFormat": "decimal", "bookmakers": "pinnacle,williamhill"},
                    timeout=20
                )
                if r.status_code == 200:
                    for event in r.json():
                        all_teams.add(event.get("home_team", ""))
                        all_teams.add(event.get("away_team", ""))
                    print(f"  {league}: {len(all_teams)} teams from The Odds API")
                    teams_by_league[league] = all_teams
                    break
                elif r.status_code == 429:
                    continue
            except Exception as e:
                print(f"  {league}: error - {e}")
                continue
        if league not in teams_by_league:
            teams_by_league[league] = set()
        time.sleep(0.5)
    return teams_by_league


# =============================================================================
# SOURCE 2: SharpAPI (DraftKings + FanDuel)
# =============================================================================
def fetch_sharp_api_teams():
    """Fetch all team names from SharpAPI across 5 leagues × 2 sportsbooks.
    
    Returns: dict {league: set of team names}
    """
    print("\n[2] Fetching team names from SharpAPI...")
    teams_by_league = {}
    for league, slug in SHARP_LEAGUE_SLUGS.items():
        all_teams = set()
        for sportsbook in ["draftkings", "fanduel"]:
            for market in SHARP_MARKETS:
                try:
                    r = requests.get(
                        "https://api.sharpapi.io/api/v1/odds",
                        headers={"X-API-Key": SHARP_API_KEY},
                        params={"sport": "soccer", "league": slug,
                                "sportsbook": sportsbook, "market": market, "limit": 200},
                        timeout=15
                    )
                    if r.status_code == 200:
                        for record in r.json().get("data", []):
                            all_teams.add(record.get("home_team", ""))
                            all_teams.add(record.get("away_team", ""))
                except Exception as e:
                    pass
                time.sleep(0.5)  # rate limit
            time.sleep(0.5)
        all_teams.discard("")  # remove empty strings
        teams_by_league[league] = all_teams
        print(f"  {league}: {len(all_teams)} teams from SharpAPI")
    return teams_by_league


# =============================================================================
# SOURCE 3: football-data.org
# =============================================================================
def fetch_football_data_org_teams():
    """Fetch all team names from football-data.org across 5 leagues.
    
    Returns: dict {league: set of team names}
    """
    print("\n[3] Fetching team names from football-data.org...")
    teams_by_league = {}
    headers = {"X-Auth-Token": FD_API_KEY}
    for league, code in FD_LEAGUE_CODES.items():
        all_teams = set()
        try:
            r = requests.get(
                f"https://api.football-data.org/v4/competitions/{code}/teams",
                headers=headers, timeout=15
            )
            if r.status_code == 200:
                teams = r.json().get("teams", [])
                for t in teams:
                    all_teams.add(t.get("name", ""))
                print(f"  {league}: {len(all_teams)} teams from football-data.org")
            else:
                print(f"  {league}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  {league}: error - {e}")
        all_teams.discard("")
        teams_by_league[league] = all_teams
        time.sleep(7)  # 10 req/min rate limit
    return teams_by_league


# =============================================================================
# SOURCE 4: beyondELO
# =============================================================================
def fetch_beyondelo_teams():
    """Fetch team names from the beyondELO history CSV.
    
    Returns: set of team names
    """
    print("\n[4] Loading team names from beyondELO history...")
    be_path = PROJECT_ROOT / "data" / "raw" / "elo" / "beyondelo_history.csv"
    if not be_path.exists():
        print(f"  [!] {be_path} not found")
        return set()
    df = pd.read_csv(be_path)
    teams = set(df["team"].unique())
    print(f"  {len(teams)} teams in beyondELO history")
    return teams


# =============================================================================
# SOURCE 5: Understat (via soccerdata)
# =============================================================================
def fetch_understat_teams():
    """Fetch team names from the Understat fresh data CSV.
    
    Returns: set of team names
    """
    print("\n[5] Loading team names from Understat...")
    us_path = PROJECT_ROOT / "data" / "raw" / "xg" / "understat_fresh.csv"
    if not us_path.exists():
        print(f"  [!] {us_path} not found")
        return set()
    df = pd.read_csv(us_path)
    teams = set(df["team"].unique())
    print(f"  {len(teams)} teams in Understat fresh data")
    return teams


# =============================================================================
# SOURCE 6: football-data.co.uk CSVs (historical + fresh)
# =============================================================================
def fetch_footballdata_csv_teams():
    """Fetch team names from football-data.co.uk CSVs.
    
    Returns: set of team names
    """
    print("\n[6] Loading team names from football-data.co.uk CSVs...")
    teams = set()
    # Historical
    hist_path = PROJECT_ROOT / "data" / "raw" / "matches" / "ALL_matches.csv"
    if hist_path.exists():
        df = pd.read_csv(hist_path, low_memory=False)
        teams.update(df["home_team"].unique())
        teams.update(df["away_team"].unique())
    # Fresh
    fresh_dir = PROJECT_ROOT / "data" / "raw" / "matches" / "fresh"
    if fresh_dir.exists():
        for csv_file in fresh_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, encoding="utf-8-sig", low_memory=False)
                if "HomeTeam" in df.columns:
                    teams.update(df["HomeTeam"].unique())
                    teams.update(df["AwayTeam"].unique())
            except:
                pass
    teams.discard("")
    print(f"  {len(teams)} teams in football-data.co.uk CSVs")
    return teams


# =============================================================================
# RUN THE AUDIT
# =============================================================================
def run_audit():
    """Run the comprehensive team name audit across all sources."""
    print("=" * 70)
    print("TEAM NAME NORMALIZATION AUDIT")
    print("=" * 70)
    print(f"Time: {datetime.now()}")
    
    # Load internal model
    print("\n[0] Loading internal model...")
    model, internal_teams = get_internal_teams()
    print(f"  Internal model has {len(internal_teams)} teams")
    print(f"  Sample: {sorted(internal_teams)[:10]}")
    
    # Fetch from all sources
    odds_api_teams = fetch_odds_api_teams()
    sharp_teams = fetch_sharp_api_teams()
    fd_org_teams = fetch_football_data_org_teams()
    beyondelo_teams = fetch_beyondelo_teams()
    understat_teams = fetch_understat_teams()
    fd_csv_teams = fetch_footballdata_csv_teams()
    
    # === TEST EACH SOURCE ===
    all_results = {}
    
    # Test 1: The Odds API
    print("\n" + "=" * 70)
    print("TEST 1: The Odds API → internal model")
    print("=" * 70)
    odds_api_results = test_source(odds_api_teams, model, internal_teams, "The Odds API")
    all_results["odds_api"] = odds_api_results
    
    # Test 2: SharpAPI
    print("\n" + "=" * 70)
    print("TEST 2: SharpAPI → internal model")
    print("=" * 70)
    sharp_results = test_source(sharp_teams, model, internal_teams, "SharpAPI")
    all_results["sharp_api"] = sharp_results
    
    # Test 3: football-data.org
    print("\n" + "=" * 70)
    print("TEST 3: football-data.org → internal model")
    print("=" * 70)
    fd_org_results = test_source(fd_org_teams, model, internal_teams, "football-data.org")
    all_results["fd_org"] = fd_org_results
    
    # Test 4: beyondELO
    print("\n" + "=" * 70)
    print("TEST 4: beyondELO → internal model")
    print("=" * 70)
    be_dict = {"all": beyondelo_teams}
    be_results = test_source(be_dict, model, internal_teams, "beyondELO")
    all_results["beyondelo"] = be_results
    
    # Test 5: Understat
    print("\n" + "=" * 70)
    print("TEST 5: Understat → internal model")
    print("=" * 70)
    us_dict = {"all": understat_teams}
    us_results = test_source(us_dict, model, internal_teams, "Understat")
    all_results["understat"] = us_results
    
    # Test 6: football-data.co.uk CSVs
    print("\n" + "=" * 70)
    print("TEST 6: football-data.co.uk CSVs → internal model")
    print("=" * 70)
    # These ARE the internal names (football-data.co.uk is our source for canonical names)
    # So they should match directly
    fd_csv_dict = {"all": fd_csv_teams}
    fd_csv_results = test_source(fd_csv_dict, model, internal_teams, "football-data CSVs", 
                                  expect_direct_match=True)
    all_results["fd_csv"] = fd_csv_results
    
    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_unmatched = 0
    for source, results in all_results.items():
        all_unmatched = []
        for league, unmatched in results.items():
            all_unmatched.extend(unmatched)
        total = sum(len(v) for v in results.values())
        unmatched_count = len(all_unmatched)
        matched_count = total - unmatched_count
        pct = (matched_count / total * 100) if total > 0 else 0
        print(f"  {source:20s}: {matched_count}/{total} ({pct:.1f}%) matched, {unmatched_count} unmatched")
        total_unmatched += unmatched_count
    
    print(f"\n  TOTAL unmatched across all sources: {total_unmatched}")
    if total_unmatched == 0:
        print("\n  ✅ ALL TEAM NAMES NORMALIZE CORRECTLY")
    else:
        print(f"\n  ⚠️  {total_unmatched} team names need fixing")
    
    # Save detailed results
    report_path = PROJECT_ROOT / "team_name_audit_report.json"
    with open(report_path, "w") as f:
        # Convert sets to lists for JSON
        json_results = {}
        for source, results in all_results.items():
            json_results[source] = {league: list(unmatched) for league, unmatched in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n  Detailed report saved to: {report_path}")
    
    return total_unmatched


def test_source(teams_by_league, model, internal_teams, source_name, 
                expect_direct_match=False):
    """Test that team names from a source normalize to internal teams.
    
    Returns dict {league: list of unmatched team names}
    """
    results = {}
    for league, teams in teams_by_league.items():
        if not teams:
            results[league] = []
            continue
        unmatched = []
        matched = 0
        for team in teams:
            if not team or pd.isna(team):
                continue
            normalized = model._normalize_team(team)
            if normalized in internal_teams:
                matched += 1
            else:
                unmatched.append(team)
        total = matched + len(unmatched)
        pct = (matched / total * 100) if total > 0 else 0
        status = "✅" if not unmatched else "⚠️ "
        print(f"  {status} {league:15s}: {matched}/{total} ({pct:.1f}%) matched")
        if unmatched:
            print(f"      UNMATCHED: {sorted(unmatched)}")
        results[league] = unmatched
    return results


if __name__ == "__main__":
    unmatched = run_audit()
    sys.exit(0 if unmatched == 0 else 1)
