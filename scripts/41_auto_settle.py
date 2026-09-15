"""
41_auto_settle.py — Auto-settlement of pending bet signals using FREE APIs.

Uses TheSportsDB (free, no API key needed for v3, 1 req/s rate limit) to fetch
match results. Falls back to other free sources if needed.

TheSportsDB endpoints used:
- eventsday.php?d=YYYY-MM-DD&l=LEAGUE_ID — all events on a given day for a league
- eventslast.php?id=TEAM_ID — last 5 events for a team

League IDs:
  EPL:        4328
  Bundesliga: 4331
  Serie A:    4332
  La Liga:    4335
  Ligue 1:    4334

Team name matching: TheSportsDB uses full names ("Manchester United", "Coventry City")
We use fuzzy matching to map to our internal canonical names ("Man United", "Coventry").

Usage:
    python scripts/41_auto_settle.py
    (one-shot; called by scheduler every 30 min)
"""
import os
import sys
import time
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta

import requests

PROJECT_ROOT = Path("/home/z/my-project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DB_PATH = PROJECT_ROOT / "data" / "live_signals.db"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# football-data.org config (PRIMARY source — more accurate than TheSportsDB)
FD_API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "525d8ca1e2a449d48f8dbbe2a7c652ca")
FD_BASE_URL = os.environ.get("FOOTBALL_DATA_BASE_URL", "https://api.football-data.org/v4")
FD_LEAGUE_CODES = {
    "EPL": "PL",        # Premier League
    "La Liga": "PD",    # Primera Division
    "Serie A": "SA",    # Serie A
    "Bundesliga": "BL1", # Bundesliga
    "Ligue 1": "FL1",   # Ligue 1
}

# TheSportsDB config (FALLBACK — free, no API key, less reliable)
# We saw TheSportsDB return 0-1 instead of 0-5 for Coventry-Brighton.
# Use only if football-data.org fails.
TS_LEAGUE_IDS = {
    "EPL": "4328",        # English Premier League
    "Bundesliga": "4331", # German Bundesliga
    "Serie A": "4332",    # Italian Serie A
    "La Liga": "4335",    # Spanish La Liga
    "Ligue 1": "4334",    # French Ligue 1
}
TS_BASE = "https://www.thesportsdb.com/api/v1/json/3"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [settle] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("settle")


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# === TEAM NAME MATCHING ===
# TheSportsDB uses full names. We use abbreviations. Same map as
# 38_refresh_model.py / 39_daily_update.py UNDERSTAT_TO_INTERNAL.
INTERNAL_FROM_FULL = {
    "Manchester United": "Man United", "Manchester City": "Man City",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom", "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Luton Town": "Luton", "Leicester City": "Leicester",
    "Leeds United": "Leeds", "Huddersfield Town": "Huddersfield",
    "Cardiff City": "Cardiff", "Hull City": "Hull",
    "Norwich City": "Norwich", "Stoke City": "Stoke",
    "Sunderland AFC": "Sunderland", "Sunderland": "Sunderland",
    "Coventry City": "Coventry",
    "AFC Bournemouth": "Bournemouth", "Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford", "Brentford": "Brentford",
    "Burnley FC": "Burnley", "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace",
    "Everton FC": "Everton", "Everton": "Everton",
    "Fulham FC": "Fulham", "Fulham": "Fulham",
    "Liverpool FC": "Liverpool", "Liverpool": "Liverpool",
    "Arsenal FC": "Arsenal", "Arsenal": "Arsenal",
    "Chelsea FC": "Chelsea", "Chelsea": "Chelsea",
    "West Ham United": "West Ham", "West Ham": "West Ham",
    "Watford FC": "Watford", "Watford": "Watford",
    "Aston Villa FC": "Aston Villa", "Aston Villa": "Aston Villa",
    "Southampton FC": "Southampton", "Southampton": "Southampton",
    "Bristol City": "Bristol City",
    "Queens Park Rangers": "QPR",
    # Serie A
    "AC Milan": "Milan", "Parma Calcio 1913": "Parma", "SPAL 2013": "Spal",
    "Benevento Calcio": "Benevento", "Frosinone Calcio": "Frosinone",
    "FC Crotone": "Crotone", "US Lecce": "Lecce",
    "AC Monza": "Monza", "Monza": "Monza",
    "US Salernitana 1919": "Salernitana",
    "Spezia Calcio": "Spezia", "US Cremonese": "Cremonese",
    "Venezia FC": "Venezia", "Genoa FC": "Genoa", "Genoa": "Genoa",
    "Hellas Verona": "Verona", "Verona": "Verona",
    "Chievo Verona": "Chievo",
    "Brescia Calcio": "Brescia", "Brescia": "Brescia",
    "Palermo FC": "Palermo", "Palermo": "Palermo",
    "Pescara Calcio": "Pescara", "Pescara": "Pescara",
    "Empoli FC": "Empoli", "Empoli": "Empoli",
    "Bologna FC": "Bologna", "Bologna": "Bologna",
    "Sassuolo Calcio": "Sassuolo", "Sassuolo": "Sassuolo",
    "Cagliari Calcio": "Cagliari", "Cagliari": "Cagliari",
    "Udinese Calcio": "Udinese", "Udinese": "Udinese",
    "Torino FC": "Torino", "Torino": "Torino",
    "Sampdoria": "Sampdoria", "UC Sampdoria": "Sampdoria",
    "Fiorentina": "Fiorentina", "ACF Fiorentina": "Fiorentina",
    "Atalanta BC": "Atalanta", "Atalanta": "Atalanta",
    "SSC Napoli": "Napoli", "Napoli": "Napoli",
    "AS Roma": "Roma", "Roma": "Roma",
    "Inter Milan": "Inter", "FC Internazionale": "Inter",
    "Juventus FC": "Juventus", "Juventus": "Juventus",
    "SS Lazio": "Lazio", "Lazio": "Lazio",
    "Como 1907": "Como", "Como": "Como",
    # Bundesliga
    "Arminia Bielefeld": "Bielefeld",
    "Borussia Dortmund": "Dortmund",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Cologne": "FC Koln", "1. FC Köln": "FC Koln",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf",
    "SpVgg Greuther Fuerth": "Greuther Furth",
    "SpVgg Greuther Fürth": "Greuther Furth",
    "Hamburger SV": "Hamburg",
    "Hannover 96": "Hannover",
    "FC Heidenheim": "Heidenheim", "1. FC Heidenheim": "Heidenheim",
    "Hertha Berlin": "Hertha", "Hertha BSC": "Hertha",
    "Bayer Leverkusen": "Leverkusen",
    "Borussia M.Gladbach": "M'gladbach", "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "Mainz 05": "Mainz", "1. FSV Mainz 05": "Mainz",
    "Nuernberg": "Nurnberg", "1. FC Nuremberg": "Nurnberg",
    "RasenBallsport Leipzig": "RB Leipzig", "RB Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "Werder Bremen": "Werder Bremen", "SV Werder Bremen": "Werder Bremen",
    "Bayern Munich": "Bayern Munich", "FC Bayern Munich": "Bayern Munich",
    "VfL Wolfsburg": "Wolfsburg",
    "SC Freiburg": "Freiburg", "Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim", "1899 Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg", "Augsburg": "Augsburg",
    "FC Schalke 04": "Schalke 04", "Schalke 04": "Schalke 04",
    "VfL Bochum": "Bochum", "Bochum": "Bochum",
    "Union Berlin": "Union Berlin", "1. FC Union Berlin": "Union Berlin",
    "SV Darmstadt 98": "Darmstadt", "Darmstadt 98": "Darmstadt",
    "SC Paderborn 07": "Paderborn", "Paderborn": "Paderborn",
    "FC St. Pauli": "St Pauli", "St Pauli": "St Pauli",
    "Holstein Kiel": "Holstein Kiel",
    # La Liga
    "Athletic Club": "Ath Bilbao", "Athletic Bilbao": "Ath Bilbao",
    "Atletico Madrid": "Ath Madrid", "Atlético Madrid": "Ath Madrid",
    "Real Betis": "Betis", "Real Betis Balompie": "Betis",
    "Celta Vigo": "Celta", "RC Celta de Vigo": "Celta",
    "Espanyol": "Espanol", "RCD Espanyol": "Espanol",
    "SD Huesca": "Huesca", "Huesca": "Huesca",
    "Deportivo La Coruna": "La Coruna", "Deportivo de La Coruña": "La Coruna",
    "Real Sociedad": "Sociedad",
    "Sporting Gijon": "Sp Gijon", "Sporting de Gijón": "Sp Gijon",
    "Real Valladolid": "Valladolid",
    "Rayo Vallecano": "Vallecano",
    "Deportivo Alaves": "Alaves", "Alavés": "Alaves", "Alaves": "Alaves",
    "UD Almeria": "Almeria", "UD Almería": "Almeria",
    "Cadiz CF": "Cadiz", "Cádiz CF": "Cadiz",
    "SD Eibar": "Eibar", "Eibar": "Eibar",
    "Elche CF": "Elche", "Elche": "Elche",
    "CD Leganes": "Leganes", "CD Leganés": "Leganes", "Leganes": "Leganes",
    "Malaga CF": "Malaga", "Málaga CF": "Malaga", "Malaga": "Malaga",
    "Real Madrid": "Real Madrid",
    "FC Barcelona": "Barcelona", "Barcelona": "Barcelona",
    "Sevilla FC": "Sevilla", "Sevilla": "Sevilla",
    "Valencia CF": "Valencia", "Valencia": "Valencia",
    "Villarreal CF": "Villarreal", "Villarreal": "Villarreal",
    "Girona FC": "Girona", "Girona": "Girona",
    "UD Las Palmas": "Las Palmas", "Las Palmas": "Las Palmas",
    "Getafe CF": "Getafe", "Getafe": "Getafe",
    "RCD Mallorca": "Mallorca", "Mallorca": "Mallorca",
    "CA Osasuna": "Osasuna", "Osasuna": "Osasuna",
    "Granada CF": "Granada", "Granada": "Granada",
    "Levante UD": "Levante", "Levante": "Levante",
    "Cordoba CF": "Cordoba", "Cordoba": "Cordoba",
    # Ligue 1
    "SC Bastia": "Bastia",
    "Clermont Foot 63": "Clermont", "Clermont Foot": "Clermont",
    "Paris Saint-Germain": "Paris SG", "Paris Saint Germain": "Paris SG",
    "Paris Saint Germain FC": "Paris SG", "PSG": "Paris SG",
    "Saint-Etienne": "St Etienne", "AS Saint-Etienne": "St Etienne",
    "Nimes Olympique": "Nimes", "Nîmes Olympique": "Nimes",
    "Dijon FCO": "Dijon",
    "Amiens SC": "Amiens",
    "AC Ajaccio": "Ajaccio",
    "Le Havre AC": "Le Havre",
    "ESTAC Troyes": "Troyes", "Troyes": "Troyes",
    "FC Metz": "Metz", "Metz": "Metz",
    "FC Lorient": "Lorient", "Lorient": "Lorient",
    "AJ Auxerre": "Auxerre", "Auxerre": "Auxerre",
    "RC Lens": "Lens", "Lens": "Lens",
    "Toulouse FC": "Toulouse", "Toulouse": "Toulouse",
    "Stade Brestois 29": "Brest", "Stade Brest": "Brest", "Brest": "Brest",
    "Stade de Reims": "Reims", "Reims": "Reims",
    "RC Strasbourg Alsace": "Strasbourg", "Strasbourg": "Strasbourg",
    "Montpellier HSC": "Montpellier", "Montpellier": "Montpellier",
    "Stade Rennais FC": "Rennes", "Rennes": "Rennes",
    "OGC Nice": "Nice", "Nice": "Nice",
    "FC Girondins de Bordeaux": "Bordeaux", "Girondins de Bordeaux": "Bordeaux",
    "FC Nantes": "Nantes", "Nantes": "Nantes",
    "Olympique de Marseille": "Marseille", "Marseille": "Marseille",
    "AS Monaco": "Monaco", "Monaco": "Monaco",
    "LOSC Lille": "Lille", "Lille OSC": "Lille", "Lille": "Lille",
    "Olympique Lyonnais": "Lyon", "Lyon": "Lyon",
    "SM Caen": "Caen", "Caen": "Caen",
    "En Avant de Guingamp": "Guingamp", "Guingamp": "Guingamp",
    "Angers SCO": "Angers", "Angers": "Angers",
    "AS Nancy Lorraine": "Nancy", "Nancy": "Nancy",
    "Stade Lavallois": "Laval", "Laval": "Laval",
    "Evian Thonon Gaillard": "Evian", "Evian": "Evian",
    "Paris FC": "Paris FC",
}


def normalize_team_name(name):
    """Map full team name → our internal canonical name."""
    if not name:
        return name
    return INTERNAL_FROM_FULL.get(name, name)


def fuzzy_team_match(api_name, our_name):
    """Fuzzy match team names: try direct map, then lowercase + strip suffixes."""
    # Direct mapping
    mapped = normalize_team_name(api_name)
    if mapped == our_name:
        return True
    
    # Lowercase compare
    a = (api_name or "").lower().strip()
    b = (our_name or "").lower().strip()
    if a == b:
        return True
    
    # Strip common suffixes (using removesuffix — str.replace with "" is catastrophic)
    for affix in [" fc", " cf", " afc", " city", " united", " town",
                  " and hove albion", "hove albion", " club"]:
        a_s = a.removesuffix(affix)
        b_s = b.removesuffix(affix)
        if a_s == b_s and (a_s != a or b_s != b):
            return True
        if a_s == b or a == b_s:
            return True
    
    # Try abbreviation expansion via word-boundary regex (avoid `.replace("", "X")`)
    import re
    abbrevs = [
        (r"\bmanchester\b", "man"),
        (r"\bman\b", "manchester"),
        (r"\bunited\b", "utd"),
        (r"\butd\b", "united"),
        (r"\bhotspurs\b", "hotspur"),
        (r"\bhotspur\b", "hotspurs"),
    ]
    a_exp = a
    b_exp = b
    for pattern, repl in abbrevs:
        a_exp = re.sub(pattern, repl, a_exp)
        b_exp = re.sub(pattern, repl, b_exp)
    if a_exp == b_exp and a_exp != a:
        return True
    
    # NOTE: We intentionally DO NOT do first-word matching. It caused false
    # positives like "Manchester City" matching "Manchester United" (both
    # first word "manchester"), settling the wrong match's score.
    return False


# =============================================================================
# football-data.org API (PRIMARY — more accurate than TheSportsDB)
# =============================================================================
def fetch_fd_matches(league_code, days=7):
    """Fetch matches for last N days from football-data.org.
    
    Returns list of dicts: {home_team, away_team, home_goals, away_goals, status, utcDate}
    """
    headers = {"X-Auth-Token": FD_API_KEY}
    today = datetime.utcnow().date()
    date_from = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    date_to = today.strftime("%Y-%m-%d")
    
    try:
        url = f"{FD_BASE_URL}/competitions/{league_code}/matches"
        params = {"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"}
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            matches = data.get("matches", [])
            # Normalize to our format
            result = []
            for m in matches:
                ft = m.get("score", {}).get("fullTime", {})
                home_goals = ft.get("home")
                away_goals = ft.get("away")
                if home_goals is None or away_goals is None:
                    continue
                result.append({
                    "home_team": m.get("homeTeam", {}).get("name", ""),
                    "away_team": m.get("awayTeam", {}).get("name", ""),
                    "home_goals": int(home_goals),
                    "away_goals": int(away_goals),
                    "status": m.get("status", ""),
                    "utcDate": m.get("utcDate", ""),
                })
            return result
        elif r.status_code == 429:
            log.warning(f"  football-data.org 429 (rate limited, 10 req/min)")
            time.sleep(8)
            return []
        else:
            log.warning(f"  football-data.org HTTP {r.status_code} for {league_code}")
            return []
    except Exception as e:
        log.error(f"  football-data.org error: {e}")
        return []


# =============================================================================
# TheSportsDB API (FALLBACK)
# =============================================================================
def fetch_day_events(date_str, league_id):
    """Fetch all events for a date + league from TheSportsDB (free).
    
    date_str: "YYYY-MM-DD" format
    league_id: TheSportsDB league ID (e.g. "4328" for EPL)
    Returns list of event dicts.
    """
    try:
        url = f"{TS_BASE}/eventsday.php?d={date_str}&l={league_id}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            events = data.get("events", [])
            if isinstance(events, list):
                return events
        return []
    except Exception as e:
        log.error(f"  TheSportsDB error on {date_str}/{league_id}: {e}")
        return []


def fetch_league_recent_ts(league_id, days=7):
    """Fetch all FINISHED events in last N days for a league from TheSportsDB (fallback).
    
    Only returns events with strStatus in {"FT", "AET", "PEN"} (Full Time / After Extra Time / Penalties).
    This prevents settling on in-play scores (e.g., a 0-1 live score that becomes 0-5 at full time).
    
    Returns flat list of event dicts with home/away/scores.
    """
    all_events = []
    # FINISHED statuses: "FT" = Full Time, "AET" = After Extra Time, "PEN" = After Penalties
    FINISHED_STATUSES = {"FT", "AET", "PEN", "Match Finished", "Finished", "FT_PEN"}
    today = datetime.utcnow().date()
    for d_offset in range(days):
        d = today - timedelta(days=d_offset)
        date_str = d.strftime("%Y-%m-%d")
        events = fetch_day_events(date_str, league_id)
        for e in events:
            str_status = (e.get("strStatus") or "").strip()
            int_home = e.get("intHomeScore")
            int_away = e.get("intAwayScore")
            # Defensive: TheSportsDB can return scores as strings ("2"), empty
            # strings (""), or None. Validate before including.
            try:
                h_int = int(int_home) if int_home is not None and int_home != "" else None
                a_int = int(int_away) if int_away is not None and int_away != "" else None
            except (ValueError, TypeError):
                continue
            # Only include if status is FT/AET/PEN AND we have a valid integer score
            if str_status in FINISHED_STATUSES and h_int is not None and a_int is not None:
                # Replace the string score with the int we validated
                e["intHomeScore"] = h_int
                e["intAwayScore"] = a_int
                all_events.append(e)
        time.sleep(1.1)  # rate limit 1 req/s
    return all_events


# =============================================================================
# SETTLEMENT LOGIC
# =============================================================================
def _determine_won(signal, home_goals, away_goals):
    """Determine if a bet won, lost, or pushed. Returns True/False/None (push)."""
    market = signal["market"]
    side = signal["side"]
    line = signal["line"]
    
    if market == "1X2":
        if side == "home": return home_goals > away_goals
        if side == "away": return home_goals < away_goals
        if side == "draw": return home_goals == away_goals
    elif market == "O/U 2.5":
        total = home_goals + away_goals
        if side == "over": return total > 2.5
        if side == "under": return total < 2.5
    elif market.startswith("AH"):
        try:
            handicap = float(line) if line is not None else float(market.split(" ")[1])
        except (TypeError, ValueError, IndexError):
            return None
        if side == "home":
            diff = home_goals + handicap - away_goals
        else:  # away
            diff = away_goals - handicap - home_goals
        if diff > 0.25: return True
        if diff < -0.25: return False
        return None  # push
    return None


def settle_match_signals(conn, home_team_internal, away_team_internal, home_goals, away_goals):
    """Settle all pending signals for a given match. Returns count of settled."""
    signals = conn.execute(
        "SELECT id, market, side, line, best_odds, kelly_stake FROM signals "
        "WHERE status='pending' AND home_team=? AND away_team=?",
        (home_team_internal, away_team_internal)
    ).fetchall()
    
    settled = 0
    for s in signals:
        won = _determine_won(s, int(home_goals), int(away_goals))
        if won is None:
            continue  # AH push or unresolved market
        profit = s["kelly_stake"] * (s["best_odds"] - 1) if won else -s["kelly_stake"]
        conn.execute(
            "UPDATE signals SET status='settled', home_goals=?, away_goals=?, won=?, profit=? WHERE id=?",
            (int(home_goals), int(away_goals), int(won), float(profit), s["id"]))
        settled += 1
        log.info(f"    SETTLED: {home_team_internal} vs {away_team_internal} "
                 f"{s['market']} {s['side']} ({home_goals}-{away_goals}) "
                 f"{'WON' if won else 'LOST'} profit={profit:+.0f}")
    return settled


def run_auto_settlement():
    """Main entry: settle all pending signals whose kickoff was > 90 min ago.
    
    PRIMARY: football-data.org (more accurate, has API key, 10 req/min)
    FALLBACK: TheSportsDB (free, less accurate — saw wrong score 0-1 vs 0-5)
    
    Matches are 90 min + stoppage (~100 min). If kickoff was > 90 min ago,
    the match should be finished.
    """
    log.info("=" * 60)
    log.info("AUTO-SETTLEMENT (primary: football-data.org, fallback: TheSportsDB)")
    log.info("=" * 60)
    
    conn = get_db()
    # Pending signals whose kickoff was > 90 minutes ago.
    pending = conn.execute("""SELECT DISTINCT home_team, away_team, league, kickoff
                              FROM signals
                              WHERE status='pending'
                              AND strftime('%s', replace(kickoff, 'T', ' ')) < strftime('%s', 'now', '-90 minutes')""").fetchall()
    conn.close()
    
    if not pending:
        log.info("  No matches to settle")
        return 0
    
    log.info(f"  {len(pending)} matches to settle (pending, kickoff > 90 min ago)")
    for r in pending[:5]:
        log.info(f"    {r['league']} | {r['home_team']} vs {r['away_team']} @ {r['kickoff']}")
    
    # Group by league
    by_league = {}
    for r in pending:
        by_league.setdefault(r["league"], []).append(r)
    
    total_settled = 0
    primary_failed_leagues = []
    
    # PRIMARY: football-data.org
    for league, code in FD_LEAGUE_CODES.items():
        if league not in by_league:
            continue
        league_pending = by_league[league]
        log.info(f"  [{league}] fetching from football-data.org (code={code})...")
        fd_matches = fetch_fd_matches(code, days=7)
        log.info(f"    got {len(fd_matches)} finished matches")
        
        if not fd_matches:
            log.warning(f"    football-data.org returned no matches for {league}, will try TheSportsDB")
            primary_failed_leagues.append(league)
            continue
        
        for r in league_pending:
            home_internal = r["home_team"]
            away_internal = r["away_team"]
            # Find this match in FD results
            match = None
            for m in fd_matches:
                if fuzzy_team_match(m["home_team"], home_internal) and \
                   fuzzy_team_match(m["away_team"], away_internal):
                    match = m
                    break
            if not match:
                continue
            # Settle all signals for this match
            conn = get_db()
            settled = settle_match_signals(conn, home_internal, away_internal,
                                            match["home_goals"], match["away_goals"])
            conn.commit()
            conn.close()
            total_settled += settled
        
        # Rate limit: 10 req/min = 6s between calls
        time.sleep(7)
    
    # FALLBACK: TheSportsDB for leagues where football-data.org failed
    for league in primary_failed_leagues:
        ts_league_id = TS_LEAGUE_IDS.get(league)
        if not ts_league_id:
            continue
        log.info(f"  [{league}] FALLBACK: fetching from TheSportsDB (id={ts_league_id})...")
        events = fetch_league_recent_ts(ts_league_id, days=7)
        log.info(f"    got {len(events)} finished events")
        
        for r in by_league[league]:
            home_internal = r["home_team"]
            away_internal = r["away_team"]
            match = None
            for e in events:
                if fuzzy_team_match(e.get("strHomeTeam", ""), home_internal) and \
                   fuzzy_team_match(e.get("strAwayTeam", ""), away_internal):
                    match = e
                    break
            if not match:
                continue
            home_goals = match.get("intHomeScore")
            away_goals = match.get("intAwayScore")
            if home_goals is None or away_goals is None:
                continue
            conn = get_db()
            settled = settle_match_signals(conn, home_internal, away_internal, home_goals, away_goals)
            conn.commit()
            conn.close()
            total_settled += settled
    
    log.info(f"  Total signals settled: {total_settled}")
    
    # Update fetch log
    conn = get_db()
    conn.execute("INSERT INTO fetch_log (fetch_type, credits_used, status, message) VALUES (?,?,?,?)",
                 ("settle", 0, "ok", f"{total_settled} settled (FD+TS)"))
    conn.commit()
    conn.close()
    
    return total_settled


if __name__ == "__main__":
    settled = run_auto_settlement()
    # Record freshness
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from data_freshness import record_refresh
        record_refresh("settle", status="ok",
                       message=f"Settled {settled} signals via football-data.org + TheSportsDB fallback",
                       records=settled)
    except ImportError:
        pass
