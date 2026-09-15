"""
39_daily_update.py — Daily incremental data refresh + model retrain.

Runs all the data refresh steps:
1. Fetch fresh Understat data (soccerdata.Understat loader — the canonical method)
   for the latest seasons (currently 2024/25 + 2025/26)
2. Re-pull beyondELO data (full refresh — API is free, ~10 minutes)
3. Refresh football-data.co.uk CSVs (download new season files if available)
4. Merge new matches into matches_with_features.csv
5. Recompute rolling features + Elo features + rest/congestion
6. Retrain LGBM model + DC state + calibrators + team_features
7. Save all artifacts for the live Streamlit app

Designed to run once per day (e.g., 09:00 Iran time via cron / APScheduler).

Outputs:
- data/raw/xg/understat_fresh.csv
- data/raw/elo/beyondelo_history.csv
- data/raw/elo/beyondelo_team_map.json
- data/processed/matches_with_features.csv
- models/lgbm_home.txt, lgbm_away.txt, dc_state.pkl, calibrators.pkl,
  team_features.pkl, lgbm_features.pkl
"""
import os
import sys
import time
import json
import shutil
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

# Set up paths
PROJECT_ROOT = Path("/home/z/my-project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from soccer_model.config import DATA_PROCESSED, ELO_DIR, MODELS_DIR, XG_DIR, DC_MAX_GOALS
from soccer_model.dixon_coles import init_dynamic_from_elo, predict_dynamic_dc, update_dynamic_dc, scoreline_matrix
from soccer_model.lgbm_head import LGBM_FEATURES
from soccer_model.features import per_team_long_format, compute_team_rolling_features, wide_format_features

# === CONFIG ===
LEAGUES = {
    "EPL":        "ENG-Premier League",
    "La_Liga":    "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie_A":    "ITA-Serie A",
    "Ligue_1":    "FRA-Ligue 1",
}
# Latest seasons to fetch fresh data for
# 2024/25, 2025/26, 2026/27 (we're a few weeks into 2627 as of 2026-09-13)
FRESH_SEASONS = ["2425", "2526", "2627"]

BEYONDELO_API = "https://api.beyondelo.com"
MATCHES_PER_PAGE = 1000
MAX_PAGES_PER_TEAM = 50

# Understat → internal team name mapping (used by 38_refresh_model.py)
UNDERSTAT_TO_INTERNAL = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom", "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspurs": "Tottenham", "Brighton and Hove Albion": "Brighton",
    "Luton Town": "Luton", "Leicester City": "Leicester",
    "Leeds United": "Leeds", "Huddersfield Town": "Huddersfield",
    "Cardiff City": "Cardiff", "Hull City": "Hull",
    "Norwich City": "Norwich", "Stoke City": "Stoke",
    "Sunderland AFC": "Sunderland", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Liverpool": "Liverpool",
    "Arsenal": "Arsenal", "Chelsea": "Chelsea",
    "West Ham": "West Ham", "Watford": "Watford",
    "Aston Villa": "Aston Villa", "Southampton": "Southampton",
    "Bristol City": "Bristol City", "QPR": "QPR",
    # Serie A
    "AC Milan": "Milan", "Parma Calcio 1913": "Parma", "SPAL 2013": "Spal",
    "Benevento Calcio": "Benevento", "Frosinone Calcio": "Frosinone",
    "FC Crotone": "Crotone", "US Lecce": "Lecce",
    "AC Monza": "Monza", "US Salernitana 1919": "Salernitana",
    "Spezia Calcio": "Spezia", "US Cremonese": "Cremonese",
    "Venezia FC": "Venezia", "Genoa": "Genoa",
    "Hellas Verona": "Verona", "Chievo": "Chievo",
    "Brescia": "Brescia", "Palermo": "Palermo",
    "Pescara": "Pescara", "Empoli": "Empoli",
    "Bologna": "Bologna", "Sassuolo": "Sassuolo",
    "Cagliari": "Cagliari", "Udinese": "Udinese",
    "Torino": "Torino", "Sampdoria": "Sampdoria",
    "Fiorentina": "Fiorentina", "Atalanta": "Atalanta",
    "Napoli": "Napoli", "Roma": "Roma", "Inter": "Inter",
    "Juventus": "Juventus", "Lazio": "Lazio",
    "Como": "Como", "Cristhian Stuani": "Stuani",
    # Bundesliga
    "Arminia Bielefeld": "Bielefeld", "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt", "FC Cologne": "FC Koln",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "SpVgg Greuther Fuerth": "Greuther Furth",
    "Hamburger SV": "Hamburg", "Hannover 96": "Hannover",
    "FC Heidenheim": "Heidenheim", "Hertha Berlin": "Hertha",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia M.Gladbach": "M'gladbach", "Mainz 05": "Mainz",
    "Nuernberg": "Nurnberg", "RasenBallsport Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart", "Werder Bremen": "Werder Bremen",
    "Bayern Munich": "Bayern Munich", "Wolfsburg": "Wolfsburg",
    "Freiburg": "Freiburg", "Hoffenheim": "Hoffenheim",
    "Augsburg": "Augsburg", "Schalke 04": "Schalke 04",
    "Bochum": "Bochum", "Union Berlin": "Union Berlin",
    "Darmstadt 98": "Darmstadt", "Paderborn": "Paderborn",
    "SV Darmstadt 98": "Darmstadt", "St. Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    # La Liga
    "Athletic Club": "Ath Bilbao", "Atletico Madrid": "Ath Madrid",
    "Real Betis": "Betis", "Celta Vigo": "Celta",
    "Espanyol": "Espanol", "SD Huesca": "Huesca",
    "Deportivo": "La Coruna", "Real Sociedad": "Sociedad",
    "Sporting Gijon": "Sp Gijon", "Real Valladolid": "Valladolid",
    "Rayo Vallecano": "Vallecano", "Alaves": "Alaves",
    "UD Almeria": "Almeria", "Cadiz CF": "Cadiz",
    "SD Eibar": "Eibar", "Elche CF": "Elche",
    "CD Leganes": "Leganes", "Malaga": "Malaga",
    "Real Madrid": "Real Madrid", "Barcelona": "Barcelona",
    "Sevilla": "Sevilla", "Valencia": "Valencia",
    "Villarreal": "Villarreal", "Girona": "Girona",
    "Las Palmas": "Las Palmas", "Getafe": "Getafe",
    "Mallorca": "Mallorca", "Osasuna": "Osasuna",
    "Granada": "Granada", "Levante": "Levante",
    "Almería": "Almeria", "Cádiz": "Cadiz", "Málaga": "Malaga",
    "Leganes": "Leganes", "Alaves": "Alaves", "Cordoba": "Cordoba",
    "Leganes": "Leganes",
    # Ligue 1
    "SC Bastia": "Bastia", "Clermont Foot 63": "Clermont",
    "Paris Saint-Germain": "Paris SG", "Saint-Etienne": "St Etienne",
    "Nimes": "Nimes", "Nîmes": "Nimes",
    "Dijon FCO": "Dijon", "Amiens SC": "Amiens",
    "AC Ajaccio": "Ajaccio", "Le Havre AC": "Le Havre",
    "Troyes": "Troyes", "Metz": "Metz",
    "Lorient": "Lorient", "Auxerre": "Auxerre",
    "Clermont Foot": "Clermont", "Lens": "Lens",
    "Toulouse": "Toulouse", "Brest": "Brest",
    "Reims": "Reims", "Strasbourg": "Strasbourg",
    "Montpellier": "Montpellier", "Rennes": "Rennes",
    "Nice": "Nice", "Bordeaux": "Bordeaux",
    "Nantes": "Nantes", "Marseille": "Marseille",
    "Monaco": "Monaco", "Lille": "Lille", "Lyon": "Lyon",
    "Caen": "Caen", "Guingamp": "Guingamp",
    "Angers": "Angers", "Dijon": "Dijon",
    "Nancy": "Nancy", "Laval": "Laval",
    "Evian": "Evian", "Paris FC": "Paris FC",
}

# === LOGGING ===
LOG_FILE = PROJECT_ROOT / "logs" / "daily_update.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def step(msg):
    log.info(f"--- {msg} ---")


def normalize_team(name):
    return UNDERSTAT_TO_INTERNAL.get(name, name)


# beyondELO abbreviation/alias map (same as 29_ingest_beyondelo.py)
BE_ABBREVIATION_MAP = {
    "man city": "manchester city", "man united": "manchester united",
    "newcastle": "newcastle united", "nott'm forest": "nottingham forest",
    "nottm forest": "nottingham forest", "west brom": "west bromwich albion",
    "wolves": "wolverhampton wanderers", "tottenham": "tottenham hotspurs",
    "luton": "luton town", "leicester": "leicester city",
    "bournemouth": "afc bournemouth", "brentford": "brentford fc",
    "huddersfield": "huddersfield town", "hull": "hull city",
    "cardiff": "cardiff city", "norwich": "norwich city",
    "sunderland": "sunderland afc", "stoke": "stoke city",
    "leeds": "leeds united",
    "milan": "ac milan", "parma": "parma calcio 1913", "spal": "spal 2013",
    "benevento": "benevento calcio", "frosinone": "frosinone calcio",
    "crotone": "fc crotone", "lecce": "us lecce", "monza": "ac monza",
    "salernitana": "us salernitana 1919", "spezia": "spezia calcio",
    "cremonese": "us cremonese", "venezia": "venezia fc",
    "bielefeld": "arminia bielefeld", "dortmund": "borussia dortmund",
    "ein frankfurt": "frankfurt", "fc koln": "köln",
    "fortuna dusseldorf": "fortuna duesseldorf",
    "greuther furth": "spvgg greuther fürth", "hamburg": "hamburger sv",
    "hannover": "hannover 96", "heidenheim": "fc heidenheim",
    "hertha": "hertha berlin", "leverkusen": "bayer leverkusen",
    "m'gladbach": "mönchengladbach", "mgladbach": "mönchengladbach",
    "mainz": "mainz 05", "nurnberg": "nürnberg",
    "rb leipzig": "rasenballsport leipzig", "stuttgart": "vfb stuttgart",
    "werder bremen": "bremen", "darmstadt": "sv darmstadt 98",
    "ingolstadt": "fc ingolstadt 04", "paderborn": "sc paderborn 07",
    "ath bilbao": "athletic club", "ath madrid": "atletico madrid",
    "betis": "real betis", "celta": "celta vigo", "espanol": "espanyol",
    "huesca": "sd huesca", "la coruna": "deportivo",
    "sociedad": "real sociedad", "sp gijon": "sporting gijón",
    "valladolid": "real valladolid", "vallecano": "rayo vallecano",
    "alaves": "alavés", "almeria": "ud almería", "cadiz": "cádiz cf",
    "eibar": "sd eibar", "elche": "elche cf", "leganes": "cd leganés",
    "malaga": "málaga", "bastia": "sc bastia",
    "clermont": "clermont foot 63", "paris sg": "paris saint-germain",
    "st etienne": "st-étienne", "nimes": "nîmes", "dijon": "dijon fco",
    "amiens": "amiens sc", "ajaccio": "ac ajaccio",
    "le havre": "le havre ac", "lorient": "lorient",
}


def _normalize_be_name(s):
    """Normalize string: lowercase + strip accents for matching."""
    import unicodedata
    s = s.lower().strip()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    return s


def _build_beyondelo_team_map(beyondelo_teams, our_teams):
    """Match our canonical team names to beyondELO team names.
    
    Same logic as 29_ingest_beyondelo.py's build_team_name_map().
    Returns (mapping, unmatched).
    mapping: {our_team: {"id": ..., "name": ...}}
    unmatched: list of team names we couldn't match
    """
    be_by_name = {_normalize_be_name(t["name"]): t for t in beyondelo_teams}
    be_by_slug = {_normalize_be_name(t.get("slug", "")): t for t in beyondelo_teams if t.get("slug")}
    
    mapping = {}
    unmatched = []
    for our_team in our_teams:
        our_norm = _normalize_be_name(our_team)
        # Try abbreviation map first
        if our_norm in {_normalize_be_name(k): v for k, v in BE_ABBREVIATION_MAP.items()}:
            for k, v in BE_ABBREVIATION_MAP.items():
                if _normalize_be_name(k) == our_norm:
                    exp_norm = _normalize_be_name(v)
                    if exp_norm in be_by_name:
                        mapping[our_team] = be_by_name[exp_norm]
                        break
                    if exp_norm in be_by_slug:
                        mapping[our_team] = be_by_slug[exp_norm]
                        break
                    break
            else:
                pass
            if our_team in mapping:
                continue
        # Direct match
        if our_norm in be_by_name:
            mapping[our_team] = be_by_name[our_norm]
            continue
        if our_norm in be_by_slug:
            mapping[our_team] = be_by_slug[our_norm]
            continue
        # Fuzzy: try removing common prefixes
        for prefix in ["fc ", "afc ", "cf ", "1. ", "club ", "sc ", "ac "]:
            cleaned = our_norm.replace(prefix, "").strip()
            if cleaned in be_by_name:
                mapping[our_team] = be_by_name[cleaned]
                break
        else:
            for suffix in [" fc", " cf", " afc", " club", " city", " united", " town"]:
                cleaned = our_norm.replace(suffix, "").strip()
                if cleaned in be_by_name:
                    mapping[our_team] = be_by_name[cleaned]
                    break
            else:
                unmatched.append(our_team)
    return mapping, unmatched


# =============================================================================
# STEP 1: Fetch fresh Understat data
# =============================================================================
def fetch_fresh_understat():
    """Pull fresh Understat per-team match stats via soccerdata.
    
    This is the CANONICAL method — see DATA_PIPELINE.md.
    Uses soccerdata.Understat which leverages tls_requests (no Selenium).
    """
    step("STEP 1: Fetching fresh Understat data via soccerdata")
    import soccerdata  # import here so logging works first
    
    all_data = []
    for league_key, sd_league in LEAGUES.items():
        for season in FRESH_SEASONS:
            log.info(f"  {league_key} {season} (soccerdata: {sd_league} {season})")
            try:
                us = soccerdata.Understat(leagues=[sd_league], seasons=[season])
                df = us.read_team_match_stats()
                if df.empty:
                    log.warning(f"  [empty] {league_key} {season}")
                    continue
                df = df.reset_index()
                df["league"] = league_key
                # Derive canonical season key from date
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], utc=True)
                    df["season"] = df["date"].apply(
                        lambda d: f"{(d.year if d.month >= 8 else d.year - 1) % 100:02d}"
                                  f"{(d.year + 1 if d.month >= 8 else d.year) % 100:02d}"
                    )
                # Soccerdata returns wide format with home_*/away_* cols —
                # we need per-team long format to match the historical schema
                # The data in fresh file should have columns:
                # league, season, date, team, h_a, xg, xga, npxg, npxga, deep, ppda_att, ppda_def, scored, missed
                rows = []
                for _, r in df.iterrows():
                    # Home team row
                    rows.append({
                        "league": league_key, "season": r["season"],
                        "date": r["date"], "team": r["home_team"], "h_a": "h",
                        "xg": r.get("home_xg"), "xga": r.get("away_xg"),
                        "npxg": r.get("home_np_xg"), "npxga": r.get("away_np_xg"),
                        "deep": r.get("home_deep_completions"),
                        "ppda_att": r.get("home_ppda"), "ppda_def": r.get("away_ppda"),
                        "scored": r.get("home_goals"), "missed": r.get("away_goals"),
                    })
                    # Away team row
                    rows.append({
                        "league": league_key, "season": r["season"],
                        "date": r["date"], "team": r["away_team"], "h_a": "a",
                        "xg": r.get("away_xg"), "xga": r.get("home_xg"),
                        "npxg": r.get("away_np_xg"), "npxga": r.get("home_np_xg"),
                        "deep": r.get("away_deep_completions"),
                        "ppda_att": r.get("away_ppda"), "ppda_def": r.get("home_ppda"),
                        "scored": r.get("away_goals"), "missed": r.get("home_goals"),
                    })
                per_team = pd.DataFrame(rows)
                log.info(f"  [ok] {league_key} {season}: {len(per_team)//2} matches, "
                         f"date {per_team['date'].min().date()} → {per_team['date'].max().date()}")
                all_data.append(per_team)
                time.sleep(1.0)  # be polite
            except Exception as e:
                log.error(f"  [error] {league_key} {season}: {type(e).__name__}: {e}")
                continue
    
    if not all_data:
        log.error("[FATAL] No Understat data fetched")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True)
    # Deduplicate
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["date", "league", "team", "h_a"], keep="last"
    ).reset_index(drop=True)
    if before != len(combined):
        log.info(f"  Dedup: {before:,} → {len(combined):,}")
    combined = combined.sort_values(["league", "date"]).reset_index(drop=True)
    
    out_path = XG_DIR / "understat_fresh.csv"
    combined.to_csv(out_path, index=False)
    log.info(f"  Saved {len(combined):,} rows to {out_path}")
    return combined


# =============================================================================
# STEP 2: Re-pull beyondELO
# =============================================================================
def fetch_beyondelo():
    """Re-pull all Elo history from beyondELO. ~10 minutes."""
    step("STEP 2: Re-pulling beyondELO data")
    
    # Get team list
    cache_path = Path("/tmp/beyondelo_teams.json")
    if cache_path.exists() and (datetime.now().timestamp() - cache_path.stat().st_mtime) < 86400:
        with open(cache_path) as f:
            be_teams = json.load(f)["data"]
        log.info(f"  Using cached team list ({len(be_teams)} teams, age <1d)")
    else:
        log.info("  Fetching team list from beyondELO...")
        r = requests.get(f"{BEYONDELO_API}/api/teams", timeout=30)
        r.raise_for_status()
        be_teams = r.json()["data"]
        with open(cache_path, "w") as f:
            json.dump({"data": be_teams}, f)
        log.info(f"  Got {len(be_teams)} teams")
    
    # Load our team list from existing matches_with_features
    matches_path = DATA_PROCESSED / "matches_with_features.csv"
    matches = pd.read_csv(matches_path, low_memory=False)
    our_teams = sorted(set(matches["home_team"].unique()) | set(matches["away_team"].unique()))
    our_teams = [t for t in our_teams if isinstance(t, str) and t]
    log.info(f"  Our teams: {len(our_teams)}")
    
    # Build team name map (same logic as 29_ingest_beyondelo.py)
    mapping, unmatched = _build_beyondelo_team_map(be_teams, our_teams)
    log.info(f"  Mapped: {len(mapping)}/{len(our_teams)}, unmatched: {len(unmatched)}")
    if unmatched:
        log.warning(f"  Unmatched: {unmatched[:10]}")
    
    # Save mapping
    mapping_path = ELO_DIR / "beyondelo_team_map.json"
    with open(mapping_path, "w") as f:
        json.dump({k: {"id": v["id"], "name": v["name"]} for k, v in mapping.items()}, f, indent=2)
    
    # Pull matches for each team
    log.info(f"  Pulling matches for {len(mapping)} teams...")
    all_elo = []
    for i, (our_team, be_team) in enumerate(mapping.items()):
        team_id = be_team["id"]
        if (i + 1) % 25 == 0:
            log.info(f"    [{i+1}/{len(mapping)}] {be_team['name']}")
        for page in range(1, MAX_PAGES_PER_TEAM + 1):
            try:
                url = f"{BEYONDELO_API}/api/teams/{team_id}/matches?limit={MATCHES_PER_PAGE}&page={page}"
                r = requests.get(url, timeout=30)
                if r.status_code != 200:
                    break
                ms = r.json().get("matches", [])
                if not ms:
                    break
                for m in ms:
                    is_home = m["home_team_id"] == team_id
                    if is_home:
                        our_elo_before = m.get("home_elo_before")
                        our_elo_after = m.get("home_elo_after")
                        opp_id, opp_name = m["away_team_id"], m.get("away_team", {}).get("name", "")
                    else:
                        our_elo_before = m.get("away_elo_before")
                        our_elo_after = m.get("away_elo_after")
                        opp_id, opp_name = m["home_team_id"], m.get("home_team", {}).get("name", "")
                    if our_elo_before is None:
                        continue
                    all_elo.append({
                        "team": our_team, "team_id": team_id,
                        "team_name_be": be_team["name"], "date": m["match_date"],
                        "is_home": is_home, "elo_before": our_elo_before,
                        "elo_after": our_elo_after, "opp_id": opp_id, "opp_name": opp_name,
                        "competition": m.get("competition", {}).get("code", ""),
                        "season": m.get("season", ""),
                        "rd_before": m.get("home_rd" if is_home else "away_rd"),
                        "hfa": m.get("home_hfa" if is_home else "away_hfa"),
                    })
                if len(ms) < MATCHES_PER_PAGE:
                    break
                time.sleep(0.1)
            except Exception as e:
                log.error(f"    [error] {be_team['name']} page {page}: {e}")
                break
    
    if not all_elo:
        log.error("[FATAL] No beyondELO data collected")
        return None
    
    df = pd.DataFrame(all_elo)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["team", "date"]).reset_index(drop=True)
    
    out_path = ELO_DIR / "beyondelo_history.csv"
    df.to_csv(out_path, index=False)
    log.info(f"  Saved {len(df):,} Elo records to {out_path}")
    log.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    return df


# =============================================================================
# STEP 3: Refresh football-data.co.uk CSVs (download new seasons)
# =============================================================================
def fetch_footballdata_csvs():
    """Download latest football-data.co.uk CSVs for new seasons.
    
    URL pattern (discovered 2026-09):
      https://www.football-data.co.uk/mmz4281/{SEASON}/{LEAGUE_CODE}.csv
    
    Where SEASON is YY format (e.g. "2425" = 2024/25) and LEAGUE_CODE is:
      E0=EPL, SP1=La Liga, I1=Serie A, D1=Bundesliga, F1=Ligue 1
    
    New seasons (2024+) have different column schema than historical:
      - No PSC column (Pinnacle reliability issue since July 2025)
      - Has B365, BFD, BV, BW, PP, SKB, Max, Avg, BFE bookmaker odds
      - Has HS, AS, HC, AC for shots/corners (same field names as historical)
      - Has HxG, AxG (match-level xG, but we use Understat for this)
    
    For the live model, we only need: Date, HomeTeam, AwayTeam, FTHG, FTAG,
    HS, AS, HST, AST, HC, AC. Match odds are not needed (we use The Odds API live).
    """
    step("STEP 3: Refreshing football-data.co.uk CSVs")
    
    FD_BASE = "https://www.football-data.co.uk/mmz4281/"
    LEAGUE_CODES = {"EPL": "E0", "La_Liga": "SP1", "Serie_A": "I1",
                   "Bundesliga": "D1", "Ligue_1": "F1"}
    # Seasons to check — 2425, 2526, 2627 (we're a few weeks into 2627)
    SEASONS_TO_CHECK = ["2425", "2526", "2627"]
    
    raw_dir = PROJECT_ROOT / "data" / "raw" / "matches" / "fresh"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = []
    for league, code in LEAGUE_CODES.items():
        for season in SEASONS_TO_CHECK:
            filename = f"{season}_{code}.csv"
            url = FD_BASE + f"{season}/{code}.csv"
            out_path = raw_dir / filename
            # Skip if already downloaded today
            if out_path.exists() and (datetime.now().timestamp() - out_path.stat().st_mtime) < 86400:
                continue
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 500:
                    out_path.write_bytes(r.content)
                    downloaded.append((league, season, out_path, len(r.content)))
                    log.info(f"  {league} {season}: downloaded {len(r.content):,} bytes")
                else:
                    log.warning(f"  {league} {season}: not available (status {r.status_code})")
                time.sleep(0.3)
            except Exception as e:
                log.warning(f"    error: {e}")
    
    return downloaded


def parse_fresh_footballdata(downloaded):
    """Parse newly downloaded football-data CSVs into our matches schema.
    
    Extracts: date, league, season, home_team, away_team, ft_home_goals,
    ft_away_goals, home_shots, away_shots, home_corners, away_corners.
    Normalises team names to match our internal canonical names (e.g.
    "Man City" not "Man City" — they're already abbreviations in football-data).
    
    Returns DataFrame with same columns as matches_with_features.csv (subset).
    """
    if not downloaded:
        return pd.DataFrame()
    
    LEAGUE_FULL = {"EPL": "EPL", "La_Liga": "La_Liga", "Serie_A": "Serie_A",
                   "Bundesliga": "Bundesliga", "Ligue_1": "Ligue_1"}
    
    all_dfs = []
    for league, season, path, _ in downloaded:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(path, encoding="latin-1")
            except Exception as e:
                log.warning(f"  could not read {path}: {e}")
                continue
        
        if df.empty:
            continue
        
        # Normalise column names — football-data uses CapsCase
        col_map = {
            "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
            "FTHG": "ft_home_goals", "FTAG": "ft_away_goals",
            "HS": "home_shots", "AS": "away_shots",
            "HST": "home_shots_on_target", "AST": "away_shots_on_target",
            "HC": "home_corners", "AC": "away_corners",
        }
        rename = {k: v for k, v in col_map.items() if k in df.columns}
        df = df.rename(columns=rename)
        
        if "date" not in df.columns:
            continue
        
        # Parse date — football-data uses DD/MM/YYYY format
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df["league"] = LEAGUE_FULL.get(league, league)
        df["season"] = season
        
        cols = ["date", "league", "season", "home_team", "away_team",
                "ft_home_goals", "ft_away_goals",
                "home_shots", "away_shots", "home_shots_on_target",
                "away_shots_on_target", "home_corners", "away_corners"]
        available = [c for c in cols if c in df.columns]
        df = df[available].copy()
        df = df.dropna(subset=["date", "home_team", "away_team"])
        all_dfs.append(df)
        log.info(f"  parsed {league} {season}: {len(df)} matches")
    
    if not all_dfs:
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "league", "home_team", "away_team"], keep="last"
    )
    return combined


# =============================================================================
# STEP 4-5: Merge fresh data + recompute features + retrain model
# =============================================================================
def refresh_model_with_fresh_data(fd_downloaded=None):
    """Merge fresh Understat + football-data matches into historical, 
    recompute features, retrain."""
    step("STEP 4-5: Merge fresh data + recompute features + retrain LGBM")
    
    # Load historical matches_with_features
    hist_path = DATA_PROCESSED / "matches_with_features.csv"
    hist = pd.read_csv(hist_path, low_memory=False)
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
    log.info(f"  Historical matches: {len(hist):,}")
    log.info(f"    latest date: {hist['date'].max()}")
    
    # Load fresh Understat
    fresh_path = XG_DIR / "understat_fresh.csv"
    if not fresh_path.exists():
        log.error("  [FATAL] No fresh Understat — run step 1 first")
        return
    fresh = pd.read_csv(fresh_path)
    fresh["date"] = pd.to_datetime(fresh["date"], utc=True)
    log.info(f"  Fresh Understat records: {len(fresh):,}")
    
    # Parse fresh football-data CSVs (for shots/corners)
    fd_df = pd.DataFrame()
    if fd_downloaded:
        fd_df = parse_fresh_footballdata(fd_downloaded)
        if not fd_df.empty:
            fd_df["date"] = pd.to_datetime(fd_df["date"])
            log.info(f"  Fresh football-data matches: {len(fd_df):,}")
    
    # Convert per-team → match-level (one row per match, not per team)
    fresh_matches = []
    for (league, date), grp in fresh.groupby(["league", "date"]):
        home_rows = grp[grp["h_a"] == "h"]
        away_rows = grp[grp["h_a"] == "a"]
        if home_rows.empty or away_rows.empty:
            continue
        h = home_rows.iloc[0]
        a = away_rows.iloc[0]
        
        home_team = normalize_team(h["team"])
        away_team = normalize_team(a["team"])
        
        # Skip if already in historical
        d = pd.Timestamp(date).tz_localize(None).date() if pd.Timestamp(date).tzinfo else pd.Timestamp(date).date()
        existing = hist[
            (hist["home_team"] == home_team) &
            (hist["away_team"] == away_team) &
            (hist["date"].dt.date == d)
        ]
        if len(existing) > 0:
            continue
        
        # Match-level record from Understat
        record = {
            "league": league, "season": h["season"], 
            "date": pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo else pd.Timestamp(date),
            "home_team": home_team, "away_team": away_team,
            "ft_home_goals": int(h["scored"]) if pd.notna(h["scored"]) else None,
            "ft_away_goals": int(a["scored"]) if pd.notna(a["scored"]) else None,
            "home_npxg": float(h["npxg"]) if pd.notna(h["npxg"]) else None,
            "away_npxg": float(a["npxg"]) if pd.notna(a["npxg"]) else None,
            "home_npxgd": float(h["npxg"] - h["npxga"]) if pd.notna(h["npxg"]) and pd.notna(h["npxga"]) else None,
            "away_npxgd": float(a["npxg"] - a["npxga"]) if pd.notna(a["npxg"]) and pd.notna(a["npxga"]) else None,
            "home_xg": float(h["xg"]) if pd.notna(h["xg"]) else None,
            "away_xg": float(a["xg"]) if pd.notna(a["xg"]) else None,
            "home_deep": int(h["deep"]) if pd.notna(h["deep"]) else None,
            "away_deep": int(a["deep"]) if pd.notna(a["deep"]) else None,
            "home_ppda": float(h["ppda_att"]) / max(h["ppda_def"], 1) if pd.notna(h["ppda_att"]) and pd.notna(h["ppda_def"]) else None,
            "away_ppda": float(a["ppda_att"]) / max(a["ppda_def"], 1) if pd.notna(a["ppda_att"]) and pd.notna(a["ppda_def"]) else None,
        }
        
        # Try to merge shots/corners from football-data.co.uk
        if not fd_df.empty:
            match_date = record["date"].date() if hasattr(record["date"], "date") else record["date"]
            fd_match = fd_df[
                (fd_df["league"] == league) &
                (fd_df["date"].dt.date == match_date) &
                (fd_df["home_team"] == home_team) &
                (fd_df["away_team"] == away_team)
            ]
            if len(fd_match) > 0:
                fd_row = fd_match.iloc[0]
                record["home_shots"] = fd_row.get("home_shots")
                record["away_shots"] = fd_row.get("away_shots")
                record["home_shots_on_target"] = fd_row.get("home_shots_on_target")
                record["away_shots_on_target"] = fd_row.get("away_shots_on_target")
                record["home_corners"] = fd_row.get("home_corners")
                record["away_corners"] = fd_row.get("away_corners")
            else:
                record["home_shots"] = None
                record["away_shots"] = None
                record["home_shots_on_target"] = None
                record["away_shots_on_target"] = None
                record["home_corners"] = None
                record["away_corners"] = None
        else:
            record["home_shots"] = None
            record["away_shots"] = None
            record["home_shots_on_target"] = None
            record["away_shots_on_target"] = None
            record["home_corners"] = None
            record["away_corners"] = None
        
        fresh_matches.append(record)
    
    fresh_df = pd.DataFrame(fresh_matches)
    log.info(f"  Fresh matches (new, not in historical): {len(fresh_df):,}")
    if len(fresh_df) > 0:
        log.info(f"    per league: {fresh_df['league'].value_counts().to_dict()}")
        # Report shots/corners coverage
        if "home_shots" in fresh_df.columns:
            shots_cov = fresh_df["home_shots"].notna().sum()
            log.info(f"    shots/corners coverage: {shots_cov}/{len(fresh_df)} "
                     f"({shots_cov/len(fresh_df)*100:.0f}%)")
    
    if fresh_df.empty:
        log.info("  No new matches. Skipping retrain.")
        return
    
    # Add missing columns from hist to fresh
    for c in hist.columns:
        if c not in fresh_df.columns:
            fresh_df[c] = np.nan
    fresh_df = fresh_df[hist.columns]
    
    # Combine
    combined = pd.concat([hist, fresh_df], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined = combined.drop_duplicates(
        subset=["date", "league", "home_team", "away_team"], keep="last"
    )
    log.info(f"  Combined matches: {len(combined):,}")
    
    # Save combined data (before feature recompute)
    combined.to_csv(hist_path, index=False)
    
    # Recompute rolling features
    log.info("  Recomputing rolling features...")
    if "home_xpts" not in combined.columns:
        combined["home_xpts"] = np.nan
        combined["away_xpts"] = np.nan
    
    feature_cols = [
        "date", "league", "season", "home_team", "away_team",
        "ft_home_goals", "ft_away_goals",
        "home_npxg", "away_npxg", "home_npxgd", "away_npxgd",
        "home_xpts", "away_xpts", "home_deep", "away_deep",
        "home_ppda", "away_ppda", "home_shots", "away_shots",
        "home_corners", "away_corners",
    ]
    available_cols = [c for c in feature_cols if c in combined.columns]
    feat_input = combined[available_cols].copy()
    
    long_df = per_team_long_format(feat_input)
    long_with_features = compute_team_rolling_features(long_df)
    wide_df = wide_format_features(long_with_features, feat_input)
    
    rolling_cols = [c for c in wide_df.columns if c.startswith("rolling_") or c.endswith("_matches_history")]
    merge_keys = ["date", "league", "home_team", "away_team"]
    combined = combined.merge(
        wide_df[merge_keys + rolling_cols],
        on=merge_keys, how="left", suffixes=("", "_new")
    )
    for c in rolling_cols:
        if c + "_new" in combined.columns:
            combined[c] = combined[c + "_new"].combine_first(combined[c])
            combined = combined.drop(columns=[c + "_new"])
    
    # Save with updated features
    combined.to_csv(hist_path, index=False)
    log.info(f"  Saved with rolling features: {len(combined):,} matches, {len(combined.columns)} columns")
    
    # Retrain LGBM
    log.info("  Retraining LGBM + DC + calibrators on ALL data...")
    elo_path = ELO_DIR / "ALL_elo.csv"
    if not elo_path.exists():
        log.error(f"  [FATAL] {elo_path} missing — re-run 02_ingest_elo.py first")
        return
    elo = pd.read_csv(elo_path)
    elo["date"] = pd.to_datetime(elo["date"])
    
    teams = sorted(set(combined["home_team"]) | set(combined["away_team"]))
    state = init_dynamic_from_elo(teams, elo)
    
    training_buffer = []
    calibrator_history = defaultdict(list)
    
    for i, m in combined.iterrows():
        h, a = m["home_team"], m["away_team"]
        hg, ag = m["ft_home_goals"], m["ft_away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue
        
        _, dc_lam_h, dc_lam_a = predict_dynamic_dc(state, h, a, DC_MAX_GOALS)
        feat = {"dc_lam_home": dc_lam_h, "dc_lam_away": dc_lam_a, "is_local_derby": 0}
        for c in LGBM_FEATURES:
            if c in feat:
                continue
            feat[c] = m.get(c, np.nan)
        
        # Calibrator history — use canonical scoreline_matrix
        # (handles tail overflow correctly; old hand-rolled version missed it)
        P = scoreline_matrix(dc_lam_h, dc_lam_a, rho=state.rho, max_goals=DC_MAX_GOALS)
        n = P.shape[0]
        idx = np.arange(n)
        over_mask = idx[:, None] + idx[None, :] > 2.5
        btts_mask = (idx[:, None] >= 1) & (idx[None, :] >= 1)
        p_home_raw = float(np.tril(P, -1).sum())
        p_draw_raw = float(np.trace(P))
        p_away_raw = float(np.triu(P, 1).sum())
        p_over_raw = float(P[over_mask].sum())
        p_btts_raw = float(P[btts_mask].sum())
        
        calibrator_history["home_win"].append((float(p_home_raw), int(hg > ag)))
        calibrator_history["draw"].append((float(p_draw_raw), int(hg == ag)))
        calibrator_history["away_win"].append((float(p_away_raw), int(hg < ag)))
        calibrator_history["over_25"].append((float(p_over_raw), int(hg + ag > 2.5)))
        calibrator_history["btts_yes"].append((float(p_btts_raw), int(hg > 0 and ag > 0)))
        
        training_buffer.append((feat, int(hg), int(ag)))
        
        target_h = m.get("home_npxg")
        target_a = m.get("away_npxg")
        if pd.isna(target_h) or pd.isna(target_a):
            target_h, target_a = int(hg), int(ag)
        update_dynamic_dc(state, h, a, float(target_h), float(target_a))
        
        if (i + 1) % 2000 == 0:
            log.info(f"    ... {i + 1}/{len(combined)}")
    
    log.info(f"  Training buffer: {len(training_buffer)} matches")
    log.info(f"  DC state: {state.matches_seen} updates")
    
    X_train = pd.DataFrame([f for f, _, _ in training_buffer])[LGBM_FEATURES]
    y_h = pd.Series([hg for _, hg, _ in training_buffer])
    y_a = pd.Series([ag for _, _, ag in training_buffer])
    X_clean = X_train.fillna(X_train.median(numeric_only=True)).fillna(0)
    
    params = {"objective": "poisson", "metric": "poisson", "num_leaves": 31,
              "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8,
              "bagging_freq": 5, "min_data_in_leaf": 30, "verbose": -1}
    model_home = lgb.train(params, lgb.Dataset(X_clean, label=y_h, feature_name=LGBM_FEATURES), 100)
    model_away = lgb.train(params, lgb.Dataset(X_clean, label=y_a, feature_name=LGBM_FEATURES), 100)
    
    model_home.save_model(str(MODELS_DIR / "lgbm_home.txt"))
    model_away.save_model(str(MODELS_DIR / "lgbm_away.txt"))
    log.info("  Saved LGBM models")
    
    with open(MODELS_DIR / "dc_state.pkl", "wb") as f:
        pickle.dump({"attack": state.attack, "defence": state.defence,
                     "home_adv": state.home_adv, "rho": state.rho,
                     "matches_seen": state.matches_seen}, f)
    log.info(f"  Saved DC state ({state.matches_seen} matches)")
    
    calibrators = {}
    for market, history in calibrator_history.items():
        if len(history) < 200:
            continue
        X_cal = np.array([p for p, _ in history])
        y_cal = np.array([o for _, o in history])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X_cal, y_cal)
        calibrators[market] = iso
    with open(MODELS_DIR / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)
    log.info(f"  Saved calibrators ({len(calibrators)} markets)")
    
    # Save latest team features
    latest_features = {}
    for team, grp in long_with_features.groupby("team"):
        grp = grp.sort_values("date")
        latest = grp.iloc[-1]
        rolling_cols_team = [c for c in grp.columns if c.startswith("rolling_")]
        feat_dict = {c: float(latest[c]) if pd.notna(latest[c]) else 0.0 for c in rolling_cols_team}
        feat_dict["matches_history"] = int(latest.get("matches_history", 0))
        feat_dict["last_match_date"] = str(latest["date"].date())
        latest_features[team] = feat_dict
    with open(MODELS_DIR / "team_features.pkl", "wb") as f:
        pickle.dump(latest_features, f)
    log.info(f"  Saved team features for {len(latest_features)} teams")
    
    # Save feature names
    with open(MODELS_DIR / "lgbm_features.pkl", "wb") as f:
        pickle.dump(LGBM_FEATURES, f)
    
    log.info(f"  ✅ Model refreshed. Latest match: {combined['date'].max()}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("=" * 70)
    log.info("DAILY UPDATE STARTED")
    log.info(f"  Time: {datetime.now()}")
    log.info("=" * 70)
    
    # Import data freshness recorder
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from data_freshness import record_refresh
    except ImportError:
        record_refresh = None
    
    start = time.time()
    
    # Step 1: Understat
    try:
        records = fetch_fresh_understat()
        if record_refresh and records is not None:
            record_refresh("understat", status="ok",
                          message=f"Fresh Understat fetched",
                          records=len(records) if hasattr(records, '__len__') else 0)
    except Exception as e:
        log.error(f"Understat fetch failed: {e}", exc_info=True)
        if record_refresh:
            record_refresh("understat", status="error", message=str(e))
    
    # Step 2: beyondELO
    try:
        df = fetch_beyondelo()
        if record_refresh and df is not None:
            record_refresh("beyondelo", status="ok",
                          message=f"beyondELO re-pulled",
                          records=len(df) if hasattr(df, '__len__') else 0)
    except Exception as e:
        log.error(f"beyondELO fetch failed: {e}", exc_info=True)
        if record_refresh:
            record_refresh("beyondelo", status="error", message=str(e))
    
    # Step 3: football-data.co.uk CSVs (best effort — may be no new seasons)
    try:
        downloaded = fetch_footballdata_csvs()
        if record_refresh:
            record_refresh("footballdata", status="ok",
                          message=f"Downloaded {len(downloaded) if downloaded else 0} CSVs",
                          records=len(downloaded) if downloaded else 0)
    except Exception as e:
        log.error(f"football-data CSV fetch failed: {e}", exc_info=True)
        if record_refresh:
            record_refresh("footballdata", status="error", message=str(e))
    
    # Step 4-5: Merge + retrain
    try:
        refresh_model_with_fresh_data(downloaded)
        if record_refresh:
            record_refresh("matches_features", status="ok",
                          message="Rebuilt with fresh data + rolling features")
            record_refresh("lgbm_model", status="ok", message="Retrained on all data")
            record_refresh("dc_state", status="ok", message="Updated via Kalman filter")
            record_refresh("calibrators", status="ok", message="Refit on all data")
            record_refresh("team_features", status="ok", message="Latest features saved")
    except Exception as e:
        log.error(f"Model refresh failed: {e}", exc_info=True)
        if record_refresh:
            record_refresh("matches_features", status="error", message=str(e))
            record_refresh("lgbm_model", status="error", message=str(e))
    
    elapsed = time.time() - start
    log.info("=" * 70)
    log.info(f"DAILY UPDATE COMPLETE in {elapsed/60:.1f} min")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
