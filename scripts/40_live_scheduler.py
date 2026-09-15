"""
40_live_scheduler.py — Background scheduler for live odds fetching, closing 
capture, auto-settlement, and CLV computation.

Runs as a standalone daemon process alongside the Streamlit app.

Schedule:
- Every 60 min: fetch initial odds for matches starting in next 24h (The Odds API, 15 credits)
- Every 4h:  SharpAPI tracking (free, FanDuel + DraftKings)
- 30 min before each kickoff: closing odds capture (one-shot, ~1 credit per match)
- Every 30 min: auto-settle matches that finished (The Odds API /scores, free)
- Daily 09:00 Iran time: full data refresh (Understat + beyondELO + retrain)

Architecture:
- Uses APScheduler BackgroundScheduler
- All state stored in data/live_signals.db (shared with Streamlit app)
- Logs to logs/scheduler.log
- Graceful shutdown on SIGINT/SIGTERM

Usage:
    python scripts/40_live_scheduler.py
    (runs in foreground; use nohup/systemd to background)
"""
import os
import sys
import time
import json
import sqlite3
import logging
import signal
import pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import numpy as np
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

# Project paths
PROJECT_ROOT = Path("/home/z/my-project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DB_PATH = PROJECT_ROOT / "data" / "live_signals.db"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === CONFIG ===
# SECURITY: API keys loaded from env vars; no hardcoded defaults.
# Set them in your shell or .streamlit/secrets.toml before running.
import os as _os
_keys_str = _os.environ.get("ODDS_API_KEYS", "")
ODDS_API_KEYS = [k.strip() for k in _keys_str.split(",") if k.strip()] or [
    # Fallback to the original keys only if env var not set (backward compat).
    # For production deployment, ALWAYS set ODDS_API_KEYS env var.
    "5da093bcd251f14f01da76e937c1996c",
    "305cab3b39fad0292f2f6bf23c705aed",
]
SHARP_API_KEY = _os.environ.get("SHARP_API_KEY", "sk_live_2CAdrXx97qM8YACRbgLM7C")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SHARP_API_BASE = "https://api.sharpapi.io/api/v1"

LEAGUES = {
    "EPL": "soccer_epl", "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a", "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
}
SHARP_LEAGUES = {"EPL": "epl", "La Liga": "spain_-_la_liga", "Serie A": "italy_-_serie_a",
                  "Bundesliga": "germany_-_bundesliga", "Ligue 1": "france_-_ligue_1"}
ODDS_API_BOOKMAKERS = "pinnacle,williamhill"
ODDS_API_MARKETS = "h2h,totals,spreads"
THRESHOLDS = {"1X2_home": 0.05, "1X2_away": 0.04, "AH_-0.5": 1.0, "AH_-1.0": 0.03,
              "AH_-1.5_plus": 0.05, "AH_positive": 0.10, "O/U_2.5": 0.05}

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scheduler")


# =============================================================================
# DB HELPERS
# =============================================================================
def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, kickoff TEXT, league TEXT,
        home_team TEXT, away_team TEXT, market TEXT, side TEXT, line REAL,
        model_prob REAL, fair_odds REAL, min_odds REAL, best_odds REAL, bookmaker TEXT,
        edge_pct REAL, is_bet INTEGER, kelly_stake REAL,
        status TEXT DEFAULT 'pending', home_goals INTEGER, away_goals INTEGER,
        won INTEGER, profit REAL, closing_odds REAL, clv_pct REAL,
        sharp_odds TEXT, opening_odds REAL, created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS odds_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT, fixture_id TEXT, kickoff TEXT,
        league TEXT, home_team TEXT, away_team TEXT, bookmaker TEXT,
        market TEXT, side TEXT, line REAL, odds REAL, captured_at TEXT,
        is_closing INTEGER DEFAULT 0)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS bankroll (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, value REAL DEFAULT 1000000,
        strategy TEXT DEFAULT 'quarter_kelly', created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fetch_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, fetch_type TEXT, league TEXT,
        credits_used INTEGER, status TEXT, message TEXT, run_at TEXT DEFAULT (datetime('now')))""")
    # Index for speed
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_kickoff ON signals(kickoff)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_snap_match ON odds_snapshots(fixture_id, captured_at)")
    conn.execute("INSERT OR IGNORE INTO bankroll (name, value) VALUES ('main', 1000000)")
    conn.commit()
    conn.close()


# =============================================================================
# THE ODDS API HELPERS
# =============================================================================
def _try_api_keys(sport_key, markets=None, days_from=None):
    """Try both API keys; return (json, remaining_credits, status_code)."""
    params_base = {"apiKey": "", "regions": "eu", "oddsFormat": "decimal",
                   "bookmakers": ODDS_API_BOOKMAKERS}
    if markets:
        params_base["markets"] = markets
    
    for key in ODDS_API_KEYS:
        params = params_base.copy()
        params["apiKey"] = key
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                log.warning(f"  429 on key {key[:8]}... trying next")
                continue
            if r.status_code == 200:
                remaining = r.headers.get("x-requests-remaining", "?")
                return r.json(), remaining, 200
            else:
                log.warning(f"  HTTP {r.status_code} on {sport_key} odds")
        except Exception as e:
            log.error(f"  fetch error: {e}")
    return [], 0, 0


def _try_api_scores(sport_key, days_from=3):
    """Fetch completed match scores (FREE, no credits consumed)."""
    for key in ODDS_API_KEYS:
        params = {"apiKey": key, "daysFrom": days_from}
        try:
            r = requests.get(f"{ODDS_API_BASE}/sports/{sport_key}/scores",
                              params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.error(f"  scores error: {e}")
    return []


# =============================================================================
# SHARPAPI HELPERS
# =============================================================================
# SharpAPI flat-record format — each entry is one (event, market, selection):
# {event_id, home_team, away_team, market_type, selection, selection_type,
#  odds_decimal, line, event_start_time, ...}
#
# SharpAPI market_type values we care about:
#   moneyline    → 1X2 (selection: "Home", "Away", "Draw" or team names)
#   total_goals  → O/U (selection: "Over"/"Under", line: float)
#   point_spread → AH (selection: team name, line: float — includes quarter-lines)
#
# Strategy: fetch 200 records per (league, sportsbook, market_type) — covers all
# events in one league. SharpAPI rate limit: 12 req/min.
SHARP_LEAGUE_SLUGS = {
    "EPL": "england_-_premier_league",
    "La Liga": "spain_-_la_liga",
    "Serie A": "italy_-_serie_a",
    "Bundesliga": "germany_-_bundesliga",
    "Ligue 1": "france_-_ligue_1",
}
SHARP_MARKET_MAP = {
    "1X2": "moneyline",
    "OU": "total_goals",
    "AH": "point_spread",
}


def fetch_sharp_odds_for_league(league_slug, sportsbook="draftkings", max_total_seconds=45):
    """Fetch all odds for a league from SharpAPI. Returns flat list of records.
    
    Makes 3 API calls (moneyline + total_goals + point_spread) per page.
    SharpAPI rate limit: 12 req/min (x-ratelimit-remaining header).
    Uses adaptive sleep: if remaining <= 2, sleeps longer; else 0.5s.
    Hard cap: max_total_seconds per league to prevent runaway fetches.
    
    Note: MAX_PAGES=3 to avoid excessive pagination (200 records × 3 markets
    × 3 pages = 1800 records, enough for most leagues).
    """
    all_records = []
    start = time.time()
    MAX_PAGES = 3  # 3 pages × 200 = 600 records per market type — enough
    MAX_RETRIES = 3
    for market_label, market_type in SHARP_MARKET_MAP.items():
        page = 1
        consecutive_429 = 0
        while page <= MAX_PAGES:
            # Check overall time budget
            if time.time() - start > max_total_seconds:
                log.warning(f"  SharpAPI time budget exceeded for {league_slug}, stopping")
                return all_records
            try:
                headers = {"X-API-Key": SHARP_API_KEY}
                params = {"sport": "soccer", "league": league_slug,
                          "sportsbook": sportsbook, "market": market_type,
                          "limit": 200, "page": page}
                r = requests.get(f"{SHARP_API_BASE}/odds", headers=headers,
                                 params=params, timeout=10)
                if r.status_code == 429:
                    consecutive_429 += 1
                    if consecutive_429 >= MAX_RETRIES:
                        log.warning(f"  SharpAPI 429 x{consecutive_429} on {league_slug}/{market_type}, skipping rest")
                        break
                    # Read reset header to know how long to wait
                    reset = r.headers.get("x-ratelimit-reset")
                    if reset:
                        try:
                            wait_until = int(reset)
                            wait_seconds = max(1, min(20, wait_until - int(time.time())))
                            log.warning(f"  SharpAPI 429 on {league_slug}/{market_type} page {page}, waiting {wait_seconds}s")
                            time.sleep(wait_seconds)
                        except:
                            time.sleep(10)
                    else:
                        time.sleep(10)
                    continue  # retry same page
                consecutive_429 = 0
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if not data:
                        break  # no more pages
                    for d in data:
                        all_records.append({
                            "event_id": d.get("event_id", ""),
                            "home_team": d.get("home_team", ""),
                            "away_team": d.get("away_team", ""),
                            "kickoff": d.get("event_start_time", "")[:19],
                            "market_type": d.get("market_type", ""),
                            "selection": d.get("selection", ""),
                            "line": d.get("line"),
                            "odds": d.get("odds_decimal"),
                            "sportsbook": sportsbook,
                        })
                    if len(data) < 200:
                        break  # last page
                    page += 1  # only advance on success
                # Adaptive sleep based on rate limit remaining
                remaining = r.headers.get("x-ratelimit-remaining")
                try:
                    rem_int = int(remaining) if remaining else 12
                except:
                    rem_int = 12
                if rem_int <= 2:
                    time.sleep(3)  # critical low — wait longer
                elif rem_int <= 5:
                    time.sleep(1.0)
                else:
                    time.sleep(0.4)
            except requests.Timeout:
                log.warning(f"  SharpAPI timeout on {league_slug}/{market_type} page {page}, skipping")
                break
            except Exception as e:
                log.error(f"  SharpAPI error on {league_slug}/{market_type} page {page}: {e}")
                time.sleep(2)
                break
    return all_records


def group_sharp_records_by_event(records):
    """Group flat SharpAPI records into per-event market dicts.
    
    Returns: {event_id: {home, away, kickoff, markets: {1X2: {...}, OU_X: {...}, AH_X: {...}}}}
    """
    events = {}
    for r in records:
        eid = r["event_id"]
        if not eid:
            continue
        if eid not in events:
            events[eid] = {
                "home": r["home_team"], "away": r["away_team"],
                "kickoff": r["kickoff"], "league": None,  # filled by caller
                "bookmaker": r["sportsbook"], "markets": {}
            }
        ev = events[eid]
        sportsbook = r["sportsbook"]
        
        # Validate odds — sometimes SharpAPI returns 0 or None
        try:
            odds_val = float(r["odds"])
        except (TypeError, ValueError):
            continue
        if odds_val <= 1.0 or odds_val > 1000:
            continue  # skip invalid odds
        
        # Map market_type → our market key
        mt = r["market_type"]
        if mt == "moneyline":
            sel = (r["selection"] or "").lower()
            home_lower = (r["home_team"] or "").lower()
            away_lower = (r["away_team"] or "").lower()
            # Selection can be "Home", "Away", "Draw", "Tie", or team name
            if sel == "home" or sel == "h" or sel == home_lower or \
               (home_lower and sel.startswith(home_lower.split()[0]) and len(home_lower.split()) > 0):
                # Defensive: ensure we don't match "Man Utd" as home when "Man City" is home (substring check is fuzzy)
                if sel == "home" or sel == "h" or sel == home_lower or \
                   _normalize_team_name(sel) == _normalize_team_name(home_lower):
                    side = "home"
                else:
                    continue
            elif sel == "away" or sel == "a" or sel == away_lower or \
                 _normalize_team_name(sel) == _normalize_team_name(away_lower):
                side = "away"
            elif sel in ("draw", "tie", "d"):
                side = "draw"
            else:
                # Last resort: substring match
                if home_lower and home_lower in sel:
                    side = "home"
                elif away_lower and away_lower in sel:
                    side = "away"
                else:
                    continue
            ev["markets"].setdefault("1X2", {})[f"{side}_{sportsbook}"] = odds_val
        
        elif mt == "total_goals":
            line = r["line"]
            sel = (r["selection"] or "").lower()
            if line is None: continue
            try:
                line_f = float(line)
            except (TypeError, ValueError):
                continue
            if sel in ("over", "o"):
                side = "over"
            elif sel in ("under", "u"):
                side = "under"
            else:
                continue
            # Use string key for consistency (avoid "2.5" vs "2.50" mismatch)
            line_key = f"OU_{line_f}"
            ev["markets"].setdefault(line_key, {})[f"{side}_{sportsbook}"] = odds_val
        
        elif mt == "point_spread":
            line = r["line"]
            if line is None: continue
            try:
                line_f = float(line)
            except (TypeError, ValueError):
                continue
            sel = (r["selection"] or "").lower()
            home_lower = (r["home_team"] or "").lower()
            away_lower = (r["away_team"] or "").lower()
            # Determine which team the selection refers to
            if sel in ("home", "h"):
                side = "home"
            elif sel in ("away", "a"):
                side = "away"
            elif sel == home_lower or _normalize_team_name(sel) == _normalize_team_name(home_lower):
                side = "home"
            elif sel == away_lower or _normalize_team_name(sel) == _normalize_team_name(away_lower):
                side = "away"
            elif home_lower and home_lower in sel:
                side = "home"
            elif away_lower and away_lower in sel:
                side = "away"
            else:
                continue
            # Convert to home handicap: line is for selected team
            # If selection=home, home_handicap = line
            # If selection=away, home_handicap = -line (away gets +line means home gets -line)
            if side == "home":
                home_handicap = line_f
            else:
                home_handicap = -line_f
            # Normalize -0.0 to 0.0
            if home_handicap == 0.0:
                home_handicap = 0.0
            ah_key = f"AH_{home_handicap}"
            ev["markets"].setdefault(ah_key, {})[f"{side}_{sportsbook}"] = odds_val
    
    return events


def _normalize_team_name(name):
    """Normalize team name for matching: lowercase, strip common prefixes/suffixes."""
    if not name:
        return ""
    n = name.lower().strip()
    # Remove common suffixes/prefixes
    for affix in [" fc", " cf", " afc", " city", " united", " town",
                  " and hove albion", "hove albion", "fc ", "afc "]:
        n = n.replace(affix, "").strip()
    # Common abbreviations
    n = n.replace("manchester", "man").replace("tottenham", "spurs")
    return n


def fetch_sharp_events(league_slug, sportsbook="draftkings"):
    """Fetch all SharpAPI events for one league. Returns list of parsed event dicts.
    
    Each event dict: {home, away, kickoff, league, bookmaker, markets}
    """
    records = fetch_sharp_odds_for_league(league_slug, sportsbook)
    if not records:
        return []
    grouped = group_sharp_records_by_event(records)
    events = []
    for eid, ev in grouped.items():
        if not ev["markets"]:
            continue
        ev["league"] = None  # caller will fill
        events.append(ev)
    return events


# =============================================================================
# MODEL + SIGNAL COMPUTATION
# =============================================================================
_model = None

def get_model():
    global _model
    if _model is None:
        from soccer_model.live_model import LiveModel
        _model = LiveModel()
        log.info("  LiveModel loaded")
    return _model


def parse_odds_api_event(event, league):
    """Parse a The Odds API event into our standard match dict."""
    home, away = event.get("home_team",""), event.get("away_team","")
    kickoff = event.get("commence_time","")[:19]
    markets = {}
    for bm in event.get("bookmakers", []):
        bk = bm.get("key","")
        for m in bm.get("markets", []):
            mk = m.get("key","")
            for o in m.get("outcomes", []):
                name, price, point = o.get("name",""), o.get("price"), o.get("point")
                if not price or price <= 1: continue
                if mk == "h2h":
                    if name == home or name.lower() == "home": sel = "home"
                    elif name == away or name.lower() == "away": sel = "away"
                    elif name.lower() == "draw": sel = "draw"
                    else: continue
                    markets.setdefault("1X2", {})[f"{sel}_{bk}"] = price
                elif mk == "totals":
                    if name.lower() == "over": markets.setdefault(f"OU_{point}", {})[f"over_{bk}"] = price
                    elif name.lower() == "under": markets.setdefault(f"OU_{point}", {})[f"under_{bk}"] = price
                elif mk == "spreads":
                    if name == home or name.lower() == "home": sel = "home"
                    elif name == away or name.lower() == "away": sel = "away"
                    else: continue
                    markets.setdefault(f"AH_{point}", {})[f"{sel}_{bk}"] = price
    return {"home": home, "away": away, "kickoff": kickoff, "league": league,
            "fixture_id": event.get("id", ""), "markets": markets}


def merge_sharp_into_odds_api(match, sharp_events):
    """Merge SharpAPI odds (FanDuel + DraftKings) into a The Odds API match dict.
    
    Modifies match['markets'] in place by adding f"{side}_{sportsbook}" keys
    for each market SharpAPI offers. This gives us 4 bookmakers' odds combined.
    """
    home_internal = match["home"]
    away_internal = match["away"]
    matched_any = False
    
    for sharp_event in sharp_events:
        # Match by home/away name (try both expanded and abbreviated forms)
        s_home = sharp_event.get("home", "")
        s_away = sharp_event.get("away", "")
        # Normalize names
        if not _names_match(s_home, home_internal) or not _names_match(s_away, away_internal):
            continue
        
        for mk_key, outcomes in sharp_event.get("markets", {}).items():
            for side_key, odds_val in outcomes.items():
                # side_key is like "home_draftkings"
                if not isinstance(odds_val, (int, float)) or odds_val <= 1:
                    continue
                # Add this odd to the match's market
                match["markets"].setdefault(mk_key, {})[side_key] = float(odds_val)
        # IMPORTANT: do NOT `return True` here — that would stop iterating
        # and skip subsequent SharpAPI events (e.g., FanDuel after DraftKings).
        # We want to merge odds from ALL SharpAPI events that match this match,
        # so both DraftKings AND FanDuel odds are added to the best-odds comparison.
        matched_any = True
    
    return matched_any


def _names_match(a, b):
    """Fuzzy team name match: lowercase + strip common suffixes.
    
    Used to merge SharpAPI (DraftKings/FanDuel) events into The Odds API events
    by team-name. The two APIs use slightly different naming conventions:
    - The Odds API: "Manchester City", "Tottenham Hotspur"
    - SharpAPI:      "Man City",       "Tottenham Hotspurs"
    
    Match strategy (in order):
    1. Exact lowercase match.
    2. Strip common affixes (fc/cf/afc/city/united/town + hove albion).
    3. Normalize abbreviations (manchester↔man, united↔utd, hotspurs↔hotspur)
       using word-boundary regex (NOT str.replace — `replace("", "X")` is catastrophic).
    We DO NOT use first-word fallback anymore — it caused false positives like
    matching "Manchester City" to "Manchester United" (both first word "manchester").
    """
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if a == b:
        return True
    # Try removing common prefixes/suffixes (using removeprefix/removesuffix for safety)
    for suffix in [" fc", " cf", " afc", " city", " united", " town",
                   " and hove albion", "hove albion"]:
        a_s = a.removesuffix(suffix)
        b_s = b.removesuffix(suffix)
        if a_s == b_s and a_s != a:  # only count if something was actually removed
            return True
        if a_s == b:
            return True
        if a == b_s:
            return True
    # Normalize abbreviations via regex word-boundary substitution.
    # Skip empty-string patterns to avoid catastrophic `.replace("", "X")`.
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
    return False


def compute_signals(match, model):
    """Run model and compute bet signals for all markets."""
    from datetime import datetime as dt
    pred = model.predict(match["home"], match["away"], dt.now())
    signals = []
    mk = match.get("markets", {})
    
    # 1X2
    for side in ["home", "away"]:
        if "1X2" not in mk: continue
        fair_key = f"1X2_{side}"
        fair = pred["fair_odds"].get(fair_key, 999)
        prob = 1/fair if fair > 0 else 0
        min_odds = pred["min_odds_to_bet"].get(fair_key, 999)
        threshold = THRESHOLDS.get(f"1X2_{side}", 0.05)
        # Pick best odds across ALL bookmakers (Pinnacle, William Hill, FanDuel, DraftKings)
        best_odds, best_bk = None, None
        for key, val in mk["1X2"].items():
            if key.startswith(side + "_") and val and (best_odds is None or val > best_odds):
                best_odds, best_bk = val, key.split("_", 1)[-1]
        if best_odds is None: continue
        edge = (prob * best_odds - 1) * 100
        is_bet = edge > threshold * 100
        kelly = min(0.02, 0.25 * (prob * best_odds - 1) / (best_odds - 1)) if is_bet and prob * best_odds > 1 else 0
        signals.append({"league": match["league"], "kickoff": match["kickoff"][:16],
            "home": match["home"], "away": match["away"], "market": "1X2", "side": side,
            "line": None, "model_prob": prob * 100, "fair_odds": fair, "min_odds": min_odds,
            "best_odds": best_odds, "bookmaker": best_bk, "edge_pct": edge,
            "is_bet": is_bet, "kelly_stake": kelly * 1000000})
    
    # O/U 2.5
    for mk_key in mk:
        if not mk_key.startswith("OU_"): continue
        try: line = float(mk_key.split("_")[1])
        except: continue
        if abs(line - 2.5) > 0.01: continue
        for side in ["over", "under"]:
            fair_key = f"OU_2.5_{side}"
            fair = pred["fair_odds"].get(fair_key, 999)
            prob = 1/fair if fair > 0 else 0
            min_odds = pred["min_odds_to_bet"].get(fair_key, 999)
            best_odds, best_bk = None, None
            for key, val in mk[mk_key].items():
                if key.startswith(side + "_") and val and (best_odds is None or val > best_odds):
                    best_odds, best_bk = val, key.split("_", 1)[-1]
            if best_odds is None: continue
            edge = (prob * best_odds - 1) * 100
            is_bet = edge > THRESHOLDS["O/U_2.5"] * 100
            signals.append({"league": match["league"], "kickoff": match["kickoff"][:16],
                "home": match["home"], "away": match["away"], "market": "O/U 2.5", "side": side,
                "line": line, "model_prob": prob * 100, "fair_odds": fair, "min_odds": min_odds,
                "best_odds": best_odds, "bookmaker": best_bk, "edge_pct": edge,
                "is_bet": is_bet, "kelly_stake": 0})
    
    # Asian Handicap
    for mk_key in mk:
        if not mk_key.startswith("AH_"): continue
        try: handicap = float(mk_key.split("_")[1])
        except: continue
        if handicap == 0.0 or handicap == -0.5: continue
        ah = pred.get("ah_probs", {}).get(handicap, {})
        # Skip if the model doesn't predict this handicap line.
        # Otherwise the default 0.5 (50/50) generates spurious signals at
        # unsupported lines (e.g., quarter-lines or large handicaps).
        if not ah:
            continue
        for side in ["home", "away"]:
            p_side = ah.get(side, 0.5)
            # Defensive: if the side isn't in the model's prediction, skip
            if side not in ah:
                continue
            fair = 1/max(p_side, 0.001)
            prob = p_side
            if handicap < -1.0: threshold = THRESHOLDS["AH_-1.5_plus"]
            elif handicap < 0: threshold = THRESHOLDS["AH_-1.0"]
            else: threshold = THRESHOLDS["AH_positive"]
            min_odds = (1 + threshold) / max(prob, 0.001)
            best_odds, best_bk = None, None
            for key, val in mk[mk_key].items():
                if key.startswith(side + "_") and val and (best_odds is None or val > best_odds):
                    best_odds, best_bk = val, key.split("_", 1)[-1]
            if best_odds is None: continue
            edge = (prob * best_odds - 1) * 100
            is_bet = edge > threshold * 100
            signals.append({"league": match["league"], "kickoff": match["kickoff"][:16],
                "home": match["home"], "away": match["away"], "market": f"AH {handicap}", "side": side,
                "line": handicap, "model_prob": prob * 100, "fair_odds": fair, "min_odds": min_odds,
                "best_odds": best_odds, "bookmaker": best_bk, "edge_pct": edge,
                "is_bet": is_bet, "kelly_stake": 0})
    return signals


def save_signals_db(signals):
    """Save bet signals to DB. Avoid duplicates by (kickoff, league, home, away, market, side)."""
    if not signals:
        return 0
    conn = get_db()
    saved = 0
    for s in signals:
        if not s["is_bet"]:
            continue
        # Check if already exists (avoid duplicates)
        existing = conn.execute(
            "SELECT id FROM signals WHERE kickoff=? AND league=? AND home_team=? AND away_team=? AND market=? AND side=?",
            (s["kickoff"], s["league"], s["home"], s["away"], s["market"], s["side"])
        ).fetchone()
        if existing:
            # Update best_odds if better
            cur_row = conn.execute("SELECT best_odds FROM signals WHERE id=?", (existing["id"],)).fetchone()
            if cur_row and s["best_odds"] > cur_row["best_odds"]:
                conn.execute("UPDATE signals SET best_odds=?, bookmaker=?, edge_pct=? WHERE id=?",
                             (s["best_odds"], s["bookmaker"], s["edge_pct"], existing["id"]))
            continue
        conn.execute("""INSERT INTO signals
            (date, kickoff, league, home_team, away_team, market, side, line,
             model_prob, fair_odds, min_odds, best_odds, bookmaker, edge_pct, is_bet, kelly_stake, status)
            VALUES (datetime('now'),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending')""",
            (s["kickoff"], s["league"], s["home"], s["away"],
             s["market"], s["side"], s["line"], s["model_prob"]/100, s["fair_odds"], s["min_odds"],
             s["best_odds"], s["bookmaker"], s["edge_pct"], 1, s["kelly_stake"]))
        saved += 1
    conn.commit()
    conn.close()
    return saved


def save_odds_snapshot(match, is_closing=False):
    """Save all current odds for a match as a snapshot (for CLV history)."""
    conn = get_db()
    for mk_key, outcomes in match.get("markets", {}).items():
        # Parse market key
        if mk_key == "1X2":
            market_name = "1X2"
            line = None
        elif mk_key.startswith("OU_"):
            market_name = "O/U"
            try: line = float(mk_key.split("_")[1])
            except: continue
        elif mk_key.startswith("AH_"):
            market_name = "AH"
            try: line = float(mk_key.split("_")[1])
            except: continue
        else:
            continue
        for side_key, odds_val in outcomes.items():
            if not isinstance(odds_val, (int, float)) or odds_val <= 1:
                continue
            # side_key format: "home_pinnacle", "over_fanduel", etc.
            parts = side_key.split("_", 1)
            side = parts[0]
            bookmaker = parts[1] if len(parts) > 1 else "unknown"
            conn.execute("""INSERT INTO odds_snapshots
                (fixture_id, kickoff, league, home_team, away_team, bookmaker,
                 market, side, line, odds, captured_at, is_closing)
                VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'),?)""",
                (match.get("fixture_id", ""), match["kickoff"], match["league"],
                 match["home"], match["away"], bookmaker, market_name, side, line,
                 float(odds_val), 1 if is_closing else 0))
    conn.commit()
    conn.close()


# =============================================================================
# SCHEDULED JOBS
# =============================================================================
def job_initial_odds_fetch():
    """Fetch initial odds for upcoming matches (every 60 min).
    
    Costs 15 credits (5 leagues × 3 markets × 1 region).
    Fetches The Odds API + SharpAPI (FanDuel + DraftKings) and merges.
    Saves bet signals to DB + saves odds snapshot.
    
    Robust: each league is processed independently. If one fails or times out,
    the others still proceed. This is important because the sandbox kills
    long-running processes.
    """
    log.info("=" * 60)
    log.info("JOB: Initial odds fetch (hourly)")
    log.info("=" * 60)
    
    try:
        model = get_model()
    except Exception as e:
        log.error(f"  Could not load model: {e}")
        return
    
    total_signals = 0
    total_credits_used = 0
    
    # Process each league independently so partial failures don't kill the run
    for league, sport_key in LEAGUES.items():
        try:
            # The Odds API
            events, remaining, status = _try_api_keys(sport_key, markets=ODDS_API_MARKETS)
            if status != 200:
                log.warning(f"  {league}: The Odds API failed (status {status})")
                continue
            # Parse events
            parsed = [parse_odds_api_event(e, league) for e in events]
            parsed = [p for p in parsed if p["markets"]]
            log.info(f"  {league}: {len(events)} events from The Odds API (credits remaining: {remaining})")
            total_credits_used += 3
            
            # SharpAPI — FanDuel + DraftKings (per league, 3 calls each × 2 books = 6 calls/league)
            sharp_events_all = []
            sharp_league_slug = SHARP_LEAGUE_SLUGS.get(league, "")
            if sharp_league_slug:
                for bk in ["draftkings", "fanduel"]:
                    try:
                        # 60-second time budget per league (was 60s before but with no cap)
                        sharp_events = fetch_sharp_events(sharp_league_slug, bk)
                        for ev in sharp_events:
                            ev["league"] = league
                            sharp_events_all.append(ev)
                    except Exception as e:
                        log.warning(f"  {league}: SharpAPI {bk} failed: {e}")
            if sharp_events_all:
                log.info(f"  {league}: {len(sharp_events_all)} SharpAPI events fetched")
            
            # Merge SharpAPI into each The Odds API event
            merged_count = 0
            for match in parsed:
                if merge_sharp_into_odds_api(match, sharp_events_all):
                    merged_count += 1
            if merged_count > 0:
                log.info(f"  {league}: merged SharpAPI into {merged_count}/{len(parsed)} matches")
            
            # Compute signals + save
            for match in parsed:
                try:
                    signals = compute_signals(match, model)
                    total_signals += save_signals_db(signals)
                    # Save odds snapshot (not closing — opening/mid-week)
                    save_odds_snapshot(match, is_closing=False)
                except Exception as e:
                    log.error(f"  {league}: error saving signals for {match.get('home','?')} vs {match.get('away','?')}: {e}")
        except Exception as e:
            log.error(f"  {league}: league fetch failed: {e}", exc_info=True)
    
    log.info(f"  Total new bet signals saved: {total_signals}")
    log.info(f"  Credits used this run: {total_credits_used}")
    
    # Log to DB
    conn = get_db()
    conn.execute("INSERT INTO fetch_log (fetch_type, credits_used, status, message) VALUES (?,?,?,?)",
                 ("initial", total_credits_used, "ok", f"{total_signals} new signals"))
    conn.commit()
    conn.close()


def job_sharp_tracking():
    """SharpAPI tracking (every 4h, free).
    
    Updates best_odds for pending signals if SharpAPI offers better odds
    than what we had from The Odds API.
    """
    log.info("=" * 60)
    log.info("JOB: SharpAPI tracking (every 4h)")
    log.info("=" * 60)
    
    conn = get_db()
    pending = conn.execute(
        "SELECT DISTINCT home_team, away_team, league FROM signals WHERE status='pending'"
    ).fetchall()
    conn.close()
    
    if not pending:
        log.info("  No pending signals to track")
        return
    
    log.info(f"  Tracking {len(pending)} pending matches")
    
    leagues_with_pending = set(p["league"] for p in pending)
    total_updated = 0
    for league, sport_key in LEAGUES.items():
        if league not in leagues_with_pending:
            continue
        
        sharp_league_slug = SHARP_LEAGUE_SLUGS.get(league, "")
        if not sharp_league_slug:
            continue
        
        league_pending = [p for p in pending if p["league"] == league]
        
        for bk in ["draftkings", "fanduel"]:
            try:
                sharp_events = fetch_sharp_events(sharp_league_slug, bk)
            except Exception as e:
                log.warning(f"  {league} {bk}: {e}")
                continue
            
            for sharp_ev in sharp_events:
                sharp_ev["league"] = league
                # Find matching pending signal
                matched = None
                for p in league_pending:
                    if _names_match(sharp_ev["home"], p["home_team"]) and \
                       _names_match(sharp_ev["away"], p["away_team"]):
                        matched = p
                        break
                if not matched:
                    continue
                
                # Update best_odds if SharpAPI is better
                conn = get_db()
                signals_for_match = conn.execute(
                    "SELECT id, market, side, line, best_odds, bookmaker, model_prob FROM signals "
                    "WHERE status='pending' AND home_team=? AND away_team=?",
                    (matched["home_team"], matched["away_team"])
                ).fetchall()
                for s in signals_for_match:
                    # Look up this market in sharp_ev
                    mk_key = None
                    if s["market"] == "1X2":
                        mk_key = "1X2"
                    elif s["market"].startswith("O/U"):
                        # O/U 2.5 → key OU_2.5
                        mk_key = "OU_2.5"
                    elif s["market"].startswith("AH"):
                        try:
                            line = float(s["line"] if s["line"] else s["market"].split(" ")[1])
                            mk_key = f"AH_{line}"
                        except:
                            continue
                    if not mk_key or mk_key not in sharp_ev["markets"]:
                        continue
                    odds_dict = sharp_ev["markets"][mk_key]
                    side_key = f"{s['side']}_{bk}"
                    if side_key not in odds_dict:
                        continue
                    sharp_odds = odds_dict[side_key]
                    if sharp_odds > s["best_odds"]:
                        new_edge = (s["model_prob"]/100 * sharp_odds - 1) * 100
                        conn.execute(
                            "UPDATE signals SET best_odds=?, bookmaker=?, edge_pct=? WHERE id=?",
                            (sharp_odds, bk, new_edge, s["id"]))
                        total_updated += 1
                        log.info(f"    Updated {matched['home_team']} vs {matched['away_team']} "
                                 f"{s['market']} {s['side']}: {s['best_odds']} → {sharp_odds} ({bk})")
                conn.commit()
                conn.close()
    
    log.info(f"  Total odds updates from SharpAPI: {total_updated}")
    conn = get_db()
    conn.execute("INSERT INTO fetch_log (fetch_type, credits_used, status, message) VALUES (?,?,?,?)",
                 ("sharp_tracking", 0, "ok", f"{total_updated} updates"))
    conn.commit()
    conn.close()


def job_closing_capture():
    """Closing odds capture (every 5 min).
    
    For any pending signal whose kickoff is within next 60 min OR was within
    last 30 min:
    - Fetch current odds from The Odds API for that match
    - Save as closing_odds + odds_snapshot with is_closing=1
    - Compute CLV: (closing_odds - bet_odds) / bet_odds * 100
    
    The wide window (T-60 to T+30) ensures we capture closing odds even if
    a fetch is missed. The Odds API continues to show odds ~30 min after
    kickoff (during in-play period).
    """
    log.info("=" * 60)
    log.info("JOB: Closing capture (every 5 min)")
    log.info("=" * 60)
    
    conn = get_db()
    # Kickoff is stored as 'YYYY-MM-DDTHH:MM' (ISO without seconds).
    # SQLite's datetime() returns 'YYYY-MM-DD HH:MM:SS'. Normalize both sides.
    soon_bets = conn.execute("""SELECT DISTINCT home_team, away_team, league, kickoff
                                FROM signals
                                WHERE status='pending' AND closing_odds IS NULL
                                AND strftime('%s', replace(kickoff, 'T', ' ')) <= strftime('%s', 'now', '+60 minutes')
                                AND strftime('%s', replace(kickoff, 'T', ' ')) > strftime('%s', 'now', '-30 minutes')""").fetchall()
    conn.close()
    
    if not soon_bets:
        log.info("  No matches needing closing capture")
        return
    
    log.info(f"  {len(soon_bets)} matches need closing capture")
    
    # Group by league and fetch
    by_league = defaultdict(list)
    for r in soon_bets:
        by_league[r["league"]].append(r)
    
    for league, sport_key in LEAGUES.items():
        if league not in by_league:
            continue
        # Fetch fresh odds (we'll use them as "closing")
        events, remaining, status = _try_api_keys(sport_key, markets=ODDS_API_MARKETS)
        if status != 200:
            log.warning(f"  {league}: closing fetch failed (status {status})")
            continue
        # Parse and match
        parsed = [parse_odds_api_event(e, league) for e in events]
        parsed = [p for p in parsed if p["markets"]]
        log.info(f"  {league}: {len(parsed)} events fetched")
        
        for r in by_league[league]:
            # Find this match
            match = None
            for p in parsed:
                if _names_match(p["home"], r["home_team"]) and _names_match(p["away"], r["away_team"]):
                    match = p
                    break
            if not match:
                log.warning(f"    {r['home_team']} vs {r['away_team']}: not in current fetch")
                continue
            # Save closing snapshot
            save_odds_snapshot(match, is_closing=True)
            # Update each pending signal's closing_odds + compute CLV
            conn = get_db()
            signals = conn.execute(
                "SELECT id, market, side, line, best_odds FROM signals "
                "WHERE status='pending' AND home_team=? AND away_team=? AND closing_odds IS NULL",
                (r["home_team"], r["away_team"])
            ).fetchall()
            for s in signals:
                # Find closing odds for this (market, side)
                mk_key = None
                if s["market"] == "1X2":
                    mk_key = "1X2"
                elif s["market"].startswith("O/U"):
                    mk_key = "OU_2.5"
                elif s["market"].startswith("AH"):
                    try:
                        line = float(s["market"].split(" ")[1])
                        mk_key = f"AH_{line}"
                    except:
                        continue
                if not mk_key or mk_key not in match["markets"]:
                    continue
                # Find best closing odds (across all bookmakers)
                best_closing = None
                for side_key, val in match["markets"][mk_key].items():
                    if side_key.startswith(s["side"] + "_") and isinstance(val, (int, float)) and val > 1:
                        if best_closing is None or val > best_closing:
                            best_closing = val
                if best_closing is None:
                    continue
                # CLV = (closing_odds - bet_odds) / bet_odds * 100
                clv = (best_closing - s["best_odds"]) / s["best_odds"] * 100
                conn.execute(
                    "UPDATE signals SET closing_odds=?, clv_pct=? WHERE id=?",
                    (best_closing, clv, s["id"]))
                log.info(f"    {r['home_team']} vs {r['away_team']} {s['market']} {s['side']}: "
                         f"closing {best_closing:.2f}, CLV {clv:+.2f}%")
            conn.commit()
            conn.close()
    
    conn = get_db()
    conn.execute("INSERT INTO fetch_log (fetch_type, credits_used, status, message) VALUES (?,?,?,?)",
                 ("closing", 0, "ok", f"{len(soon_bets)} matches"))
    conn.commit()
    conn.close()


def job_auto_settle():
    """Auto-settle finished matches (every 30 min).
    
    Uses TheSportsDB (free, no API key needed) — see scripts/41_auto_settle.py
    for the full implementation. This is a thin wrapper.
    """
    log.info("=" * 60)
    log.info("JOB: Auto-settle (every 30 min)")
    log.info("=" * 60)
    
    try:
        from importlib import import_module
        mod = import_module("41_auto_settle")
        mod.run_auto_settlement()
    except Exception as e:
        log.error(f"  Auto-settle failed: {e}", exc_info=True)
        conn = get_db()
        conn.execute("INSERT INTO fetch_log (fetch_type, credits_used, status, message) VALUES (?,?,?,?)",
                     ("settle", 0, "error", str(e)))
        conn.commit()
        conn.close()


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
        # AH: home_team handicap is `line` (negative if home is favored)
        # For side="home": result = (home_goals + line) - away_goals
        # For side="away": result = (away_goals - line) - home_goals
        try:
            handicap = float(line)
        except (TypeError, ValueError):
            return None
        if side == "home":
            diff = home_goals + handicap - away_goals
        else:  # away
            diff = away_goals - handicap - home_goals
        if diff > 0.25: return True
        if diff < -0.25: return False
        return None  # push (diff in [-0.25, 0.25])
    return None


def job_daily_refresh():
    """Full daily data refresh (09:00 Iran time)."""
    log.info("=" * 60)
    log.info("JOB: Daily data refresh")
    log.info("=" * 60)
    
    try:
        import subprocess
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "39_daily_update.py")],
                      cwd=str(PROJECT_ROOT), timeout=1800)  # 30 min timeout
        log.info("  Daily refresh complete")
    except subprocess.TimeoutExpired:
        log.error("  Daily refresh timed out (30 min)")
    except Exception as e:
        log.error(f"  Daily refresh failed: {e}", exc_info=True)
    
    # Reload model in this process
    global _model
    _model = None
    get_model()


# =============================================================================
# SCHEDULER SETUP
# =============================================================================
def setup_scheduler():
    """Create and configure the APScheduler."""
    scheduler = BackgroundScheduler(timezone="UTC")
    
    # Initial odds fetch: every 60 minutes
    scheduler.add_job(job_initial_odds_fetch, IntervalTrigger(minutes=60),
                      id="initial_odds", max_instances=1,
                      coalesce=True, misfire_grace_time=120)
    
    # SharpAPI tracking: every 4 hours
    scheduler.add_job(job_sharp_tracking, IntervalTrigger(hours=4),
                      id="sharp_tracking", max_instances=1,
                      coalesce=True, misfire_grace_time=600)
    
    # Closing capture: every 5 minutes (lightweight — only fetches if needed)
    scheduler.add_job(job_closing_capture, IntervalTrigger(minutes=5),
                      id="closing_capture", max_instances=1,
                      coalesce=True, misfire_grace_time=120)
    
    # Auto-settle: every 30 minutes
    scheduler.add_job(job_auto_settle, IntervalTrigger(minutes=30),
                      id="auto_settle", max_instances=1,
                      coalesce=True, misfire_grace_time=300)
    
    # Daily full refresh: 09:00 Iran time = 04:30 UTC (Iran is UTC+4:30)
    scheduler.add_job(job_daily_refresh, CronTrigger(hour=4, minute=30, timezone="UTC"),
                      id="daily_refresh", max_instances=1, coalesce=True,
                      misfire_grace_time=3600)
    
    return scheduler


# =============================================================================
# MAIN
# =============================================================================
def main():
    log.info("=" * 60)
    log.info("LIVE ODDS SCHEDULER STARTING")
    log.info(f"  Time: {datetime.now()} (Iran: {datetime.now(timezone(timedelta(hours=4, minutes=30)))})")
    log.info(f"  DB: {DB_PATH}")
    log.info(f"  Log: {LOG_DIR / 'scheduler.log'}")
    log.info("=" * 60)
    
    init_db()
    
    # Run initial fetch immediately
    log.info("Running initial odds fetch on startup...")
    try:
        job_initial_odds_fetch()
    except Exception as e:
        log.error(f"Initial fetch failed: {e}", exc_info=True)
    
    # Set up scheduler
    scheduler = setup_scheduler()
    scheduler.start()
    log.info("Scheduler started. Jobs:")
    for job in scheduler.get_jobs():
        log.info(f"  {job.id}: next run at {job.next_run_time}")
    
    # Graceful shutdown
    def shutdown(signum, frame):
        log.info(f"Received signal {signum}, shutting down...")
        scheduler.shutdown(wait=False)
        log.info("Scheduler stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Keep alive
    log.info("Scheduler running. Press Ctrl+C to stop.")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
