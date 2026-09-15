"""
Live Soccer Betting Signal App — Streamlit Dashboard

Features:
1. Reads bet signals from the shared SQLite DB (populated by 40_live_scheduler.py)
2. Shows merged odds from Pinnacle + William Hill + FanDuel + DraftKings
3. Displays CLV (closing line value) for each bet
4. Auto-settlement status (pending / settled)
5. Performance dashboard with yield, hit rate, CLV breakdown
6. Bankroll strategy comparison
7. Scheduler status (last fetch, next scheduled)
8. Manual refresh button (triggers scheduler fetch)
9. Data freshness indicators for each data source
10. Manual "refresh all" button to update all data
11. Password-protected (private)

Run with:
    streamlit run streamlit_app.py
"""
import os, sys, sqlite3, json, time, pickle, subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import streamlit as st
import pandas as pd
import numpy as np
import requests
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from soccer_model.live_model import LiveModel
from soccer_model.config import DC_MAX_GOALS, BANKROLL
from data_freshness import get_all_freshness, record_refresh

# === CONFIG ===
PROJECT_ROOT = Path(__file__).parent
DB_PATH = PROJECT_ROOT / "data" / "live_signals.db"
SCHEDULER_LOG = PROJECT_ROOT / "logs" / "scheduler.log"
DAILY_LOG = PROJECT_ROOT / "logs" / "daily_update.log"

# SECURITY: API keys from env vars (with fallback for backward compat).
# For production, set ODDS_API_KEYS env var as comma-separated string.
import os as _os
_keys_str = _os.environ.get("ODDS_API_KEYS", "")
ODDS_API_KEYS = [k.strip() for k in _keys_str.split(",") if k.strip()] or [
    "5da093bcd251f14f01da76e937c1996c",
    "305cab3b39fad0292f2f6bf23c705aed",
]
# App password from env var (with fallback for backward compat).
APP_PASSWORD = _os.environ.get("APP_PASSWORD", "soccer2026")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
LEAGUES = {
    "EPL": "soccer_epl", "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a", "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
}

# === DB HELPERS ===
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
    conn.execute("INSERT OR IGNORE INTO bankroll (name, value) VALUES ('main', 1000000)")
    conn.commit()
    conn.close()

def check_api_credits():
    """Check The Odds API remaining credits without consuming any."""
    # The /sports endpoint is free (returns list of sports)
    # We use one key to ping status
    try:
        r = requests.get(f"{ODDS_API_BASE}/sports",
                          params={"apiKey": ODDS_API_KEYS[0]}, timeout=10)
        remaining = r.headers.get("x-requests-remaining", "?")
        used = r.headers.get("x-requests-used", "?")
        return remaining, used
    except:
        return "?", "?"

# === SHARPAPI odds snapshot reader ===
def get_all_bookmaker_odds_for_signal(home, away, market, side, line=None):
    """Get all bookmaker odds snapshot for a signal — to show the merged odds table.
    
    Returns dict like {"pinnacle": 3.15, "williamhill": 3.10, "draftkings": 3.20, "fanduel": 3.05}
    """
    conn = get_db()
    # Parse market + line
    if market == "1X2":
        market_name = "1X2"
    elif market.startswith("O/U"):
        market_name = "O/U"
    elif market.startswith("AH"):
        market_name = "AH"
    else:
        conn.close()
        return {}
    
    # Get latest non-closing snapshot
    rows = conn.execute("""SELECT bookmaker, odds FROM odds_snapshots
                           WHERE home_team=? AND away_team=? AND market=? AND side=?
                           AND (line IS ? OR line = ?)
                           AND is_closing = 0
                           ORDER BY captured_at DESC LIMIT 50""",
                        (home, away, market_name, side, line, line)).fetchall()
    conn.close()
    
    # Keep first (most recent) per bookmaker
    bookmaker_odds = {}
    for r in rows:
        bk = r["bookmaker"]
        if bk not in bookmaker_odds:
            bookmaker_odds[bk] = r["odds"]
    return bookmaker_odds


@st.cache_resource
def load_model():
    return LiveModel()


# === STREAMLIT APP ===
st.set_page_config(page_title="⚽ Soccer Betting Signals", page_icon="⚽", layout="wide")

if "auth" not in st.session_state: st.session_state.auth = False
if not st.session_state.auth:
    st.title("⚽ Soccer Betting Signals")
    pw = st.text_input("Password", type="password")
    # Single button click — the old code created two buttons with the same label,
    # which Streamlit treats as the same widget (fragile).
    if st.button("Login"):
        if pw == APP_PASSWORD:
            st.session_state.auth = True; st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

init_db()
model = load_model()
st.title("⚽ Soccer Betting Signals")
st.caption("v12-gated-sweep · Pinnacle + William Hill + FanDuel + DraftKings · LGBM + DC + Elo + Rest/Congestion · Auto-settlement via football-data.org")

# Page navigation: Signals (default) or Team Manager
if "page" not in st.session_state:
    st.session_state.page = "signals"

# Top navigation bar
nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 6])
with nav_c1:
    if st.button("📊 Signals", help="View bet signals and dashboard"):
        st.session_state.page = "signals"
        st.rerun()
with nav_c2:
    if st.button("🏒 Team Manager", help="Match team names across sources and add new teams"):
        st.session_state.page = "team_manager"
        st.rerun()

st.divider()

# Route to the selected page
if st.session_state.page == "team_manager":
    from team_manager_page import render_team_manager_page
    render_team_manager_page()
    st.stop()

# Top bar: API credits + scheduler status
c1, c2, c3, c4 = st.columns(4)
try:
    remaining, used = check_api_credits()
    c1.metric("API Credits", f"{remaining}")
except:
    c1.metric("API Credits", "?")

conn = get_db()
last_fetch = conn.execute("""SELECT run_at FROM fetch_log ORDER BY id DESC LIMIT 1""").fetchone()
last_settle = conn.execute("""SELECT run_at FROM fetch_log WHERE fetch_type='settle' ORDER BY id DESC LIMIT 1""").fetchone()
scheduler_running = SCHEDULER_LOG.exists() and (time.time() - SCHEDULER_LOG.stat().st_mtime) < 600
c2.metric("Last Fetch", last_fetch["run_at"][-8:] if last_fetch else "—")
c3.metric("Last Settle", last_settle["run_at"][-8:] if last_settle else "—")
c4.metric("Scheduler", "🟢 running" if scheduler_running else "🔴 stopped")
conn.close()

# === DATA FRESHNESS PANEL ===
st.divider()
with st.expander("📊 Data Freshness — last update for each source", expanded=False):
    freshness_data = get_all_freshness()
    if freshness_data:
        # Display as a table
        fresh_df = pd.DataFrame(freshness_data)
        # Color code by age
        def color_age(age_str):
            if age_str == "never":
                return "background-color: #ffcdd2"  # red
            if "s" in age_str and "d" not in age_str and "h" not in age_str:
                # just seconds
                return "background-color: #c8e6c9"
            if "m" in age_str and "h" not in age_str and "d" not in age_str:
                # minutes — fresh
                return "background-color: #c8e6c9"
            if "h" in age_str and "d" not in age_str:
                # hours — OK
                # Parse number
                try:
                    hours = int(age_str.split("h")[0])
                    if hours < 6:
                        return "background-color: #c8e6c9"
                    elif hours < 24:
                        return "background-color: #fff9c4"
                    else:
                        return "background-color: #ffcdd2"
                except:
                    return ""
            # days
            return "background-color: #ffcdd2"
        
        # Display key columns
        display_cols = ["label", "status", "age_human", "records", "description"]
        st.dataframe(
            fresh_df[display_cols].rename(columns={
                "label": "Data Source",
                "status": "Status",
                "age_human": "Last Updated",
                "records": "Records",
                "description": "Description",
            }).style.map(color_age, subset=["Last Updated"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No freshness data yet. Run the daily refresh or fetch.")
    
    # Manual refresh buttons
    st.write("**Manual Refresh:**")
    col1, col2, col3, col4 = st.columns(4)
    
    if col1.button("🔄 Understat", help="Fetch fresh Understat NPxG data via soccerdata (~5 min). Only fetches Understat, does not retrain."):
        with st.spinner("Fetching Understat..."):
            try:
                # Run ONLY the Understat fetch step (not the full 39_daily_update.py).
                # Old code ran the full pipeline, which was misleading.
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import sys; sys.path.insert(0, 'scripts'); "
                     "from importlib import import_module; "
                     "mod = import_module('39_daily_update'); "
                     "records = mod.fetch_fresh_understat(); "
                     "from data_freshness import record_refresh; "
                     "if records is not None: "
                     "  record_refresh('understat', status='ok', records=len(records) if hasattr(records, '__len__') else 0)"],
                    cwd=str(PROJECT_ROOT), timeout=600, capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("✅ Understat fetched (run Retrain to update model)")
                else:
                    st.error(f"❌ Failed: {proc.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                st.warning("⏰ Timeout (10 min)")
            except Exception as e:
                st.error(f"❌ {e}")
        st.rerun()
    
    if col2.button("⚽ beyondELO", help="Re-pull all beyondELO data (~10 min)"):
        with st.spinner("Fetching beyondELO..."):
            try:
                # Run just step 2 of daily update via Python
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import sys; sys.path.insert(0, 'scripts'); "
                     "from importlib import import_module; "
                     "mod = import_module('39_daily_update'); "
                     "df = mod.fetch_beyondelo(); "
                     "from data_freshness import record_refresh; "
                     "record_refresh('beyondelo', status='ok', records=len(df) if df is not None else 0)"],
                    cwd=str(PROJECT_ROOT), timeout=900, capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("✅ beyondELO refreshed")
                else:
                    st.error(f"❌ {proc.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                st.warning("⏰ Timeout (15 min)")
            except Exception as e:
                st.error(f"❌ {e}")
        st.rerun()
    
    if col3.button("📋 football-data CSVs", help="Download latest football-data.co.uk CSVs (~30 sec)"):
        with st.spinner("Downloading football-data CSVs..."):
            try:
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import sys; sys.path.insert(0, 'scripts'); "
                     "from importlib import import_module; "
                     "mod = import_module('39_daily_update'); "
                     "downloaded = mod.fetch_footballdata_csvs(); "
                     "from data_freshness import record_refresh; "
                     "record_refresh('footballdata', status='ok', "
                     "message=f'Downloaded {len(downloaded) if downloaded else 0} CSVs', "
                     "records=len(downloaded) if downloaded else 0)"],
                    cwd=str(PROJECT_ROOT), timeout=120, capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("✅ football-data CSVs downloaded")
                else:
                    st.error(f"❌ {proc.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                st.warning("⏰ Timeout (2 min)")
            except Exception as e:
                st.error(f"❌ {e}")
        st.rerun()
    
    if col4.button("🎯 Retrain Model", help="Recompute features + retrain LGBM on all data (~3 min)"):
        with st.spinner("Retraining model..."):
            try:
                proc = subprocess.run(
                    [sys.executable, "-c",
                     "import sys; sys.path.insert(0, 'scripts'); "
                     "from importlib import import_module; "
                     "mod = import_module('39_daily_update'); "
                     "mod.refresh_model_with_fresh_data(mod.fetch_footballdata_csvs()); "
                     "from data_freshness import record_refresh; "
                     "for src in ['matches_features', 'lgbm_model', 'dc_state', 'calibrators', 'team_features']: "
                     "  record_refresh(src, status='ok', message='Manual retrain')"],
                    cwd=str(PROJECT_ROOT), timeout=600, capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("✅ Model retrained on all data")
                else:
                    st.error(f"❌ {proc.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                st.warning("⏰ Timeout (10 min)")
            except Exception as e:
                st.error(f"❌ {e}")
        # Force model reload
        try:
            st.cache_resource.clear()
        except:
            pass
        st.rerun()
    
    st.write("**Full Pipeline:**")
    if st.button("🚀 Run Full Daily Refresh (Understat + beyondELO + CSVs + Retrain, ~20 min)", 
                 help="Equivalent to scripts/39_daily_update.py — refreshes ALL data sources + retrains model"):
        with st.spinner("Running full daily refresh. This takes 15-20 minutes..."):
            try:
                proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "39_daily_update.py")],
                                    cwd=str(PROJECT_ROOT), timeout=1800,
                                    capture_output=True, text=True)
                if proc.returncode == 0:
                    st.success("✅ Full daily refresh complete")
                    st.cache_resource.clear()
                else:
                    st.error(f"❌ Failed: {proc.stderr[-500:]}")
            except subprocess.TimeoutExpired:
                st.warning("⏰ Timeout (30 min). Check logs/daily_update.log")
            except Exception as e:
                st.error(f"❌ {e}")
        st.rerun()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Refresh from DB", type="primary", help="Reload signals from scheduler DB"):
        st.session_state.signals_df = None
        st.rerun()
    if st.button("⚡ Trigger Scheduler Fetch", help="Run initial odds fetch (one-shot, ~2 min). Uses subprocess.run with timeout to avoid spawning a long-running daemon."):
        try:
            # Use subprocess.run (not Popen) with a one-shot command that ONLY calls
            # job_initial_odds_fetch() — does NOT start the scheduler daemon.
            # This avoids the bug where multiple clicks spawn multiple daemons.
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import sys; sys.path.insert(0, 'scripts'); "
                 "from importlib import import_module; "
                 "mod = import_module('40_live_scheduler'); "
                 "mod.init_db(); mod.job_initial_odds_fetch()"],
                cwd=str(PROJECT_ROOT), timeout=300,
                capture_output=True, text=True)
            if proc.returncode == 0:
                st.success("✅ Initial fetch complete")
                st.session_state.signals_df = None  # force reload
            else:
                st.error(f"❌ Failed: {proc.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            st.warning("⏰ Timeout (5 min). The fetch may have hit SharpAPI rate limits.")
        except Exception as e:
            st.error(f"Could not trigger: {e}")
    if st.button("📅 Run Daily Refresh Now", help="Run the full daily data refresh (Understat + beyondELO + retrain). Takes 15-20 min."):
        try:
            # Use subprocess.run with 30-min timeout (not Popen) to prevent
            # concurrent runs that would corrupt model files.
            proc = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "39_daily_update.py")],
                cwd=str(PROJECT_ROOT), timeout=1800,
                capture_output=True, text=True)
            if proc.returncode == 0:
                st.success("✅ Daily refresh complete")
                st.cache_resource.clear()  # force model reload
                st.session_state.signals_df = None
            else:
                st.error(f"❌ Failed: {proc.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            st.warning("⏰ Timeout (30 min). Check logs/daily_update.log")
        except Exception as e:
            st.error(f"Could not trigger: {e}")
    if st.button("💰 Settle Pending Now", help="Run auto-settlement immediately"):
        try:
            proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "41_auto_settle.py")],
                          cwd=str(PROJECT_ROOT), timeout=120,
                          capture_output=True, text=True)
            if proc.returncode == 0:
                st.success("✅ Settlement complete")
                st.session_state.signals_df = None
            else:
                st.error(f"❌ Failed: {proc.stderr[-500:]}")
        except Exception as e:
            st.error(f"Could not trigger: {e}")
    
    min_odds_f = st.slider("Min odds", 1.0, 10.0, 1.50, 0.05)
    min_edge_f = st.slider("Min edge %", 0.0, 20.0, 0.0, 0.5)
    only_bets = st.checkbox("Only show bet signals", value=True)
    show_settled = st.checkbox("Show settled too", value=False)
    sel_leagues = st.multiselect("Leagues", list(LEAGUES.keys()), default=list(LEAGUES.keys()))
    sel_markets = st.multiselect("Markets", ["1X2", "O/U 2.5", "AH"], default=["1X2", "O/U 2.5", "AH"])
    st.divider()

    # Bankroll management
    st.subheader("💰 Bankroll")
    conn = get_db()
    br = conn.execute("SELECT value FROM bankroll WHERE name='main'").fetchone()
    bankroll = br["value"] if br else 1000000
    new_br = st.number_input("Bankroll (Toman)", value=float(bankroll), step=100000.0)
    if st.button("Update Bankroll"):
        conn.execute("UPDATE bankroll SET value=? WHERE name='main'", (new_br,))
        conn.commit()
        st.success(f"Bankroll updated to {new_br:,.0f}")
    # Separate button to reset signals (with confirmation checkbox)
    st.markdown("**⚠️ Danger zone:**")
    confirm_reset = st.checkbox("I confirm I want to reset ALL settled signals back to pending",
                                 value=False, key="confirm_reset")
    if st.button("🔄 Reset All Signals", disabled=not confirm_reset, help="Irreversibly clears won/profit on all settled signals"):
        conn.execute("UPDATE signals SET status='pending', won=NULL, profit=NULL WHERE status='settled'")
        conn.commit()
        st.success("All signals reset to pending")
        st.session_state.signals_df = None
    conn.close()

    staking = st.selectbox("Staking strategy", ["¼ Kelly", "½ Kelly", "Flat 1%", "Flat 2%"])
    staking_mult = {"¼ Kelly": 0.25, "½ Kelly": 0.50, "Flat 1%": None, "Flat 2%": None}
    flat_pct = {"Flat 1%": 0.01, "Flat 2%": 0.02}

    st.divider()
    st.subheader("📊 Stats")
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM signals WHERE status='pending'").fetchone()[0]
    settled = conn.execute("SELECT COUNT(*) FROM signals WHERE status='settled'").fetchone()[0]
    wins = conn.execute("SELECT COUNT(*) FROM signals WHERE won=1").fetchone()[0]
    losses = conn.execute("SELECT COUNT(*) FROM signals WHERE won=0").fetchone()[0]
    profit = conn.execute("SELECT COALESCE(SUM(profit),0) FROM signals WHERE status='settled'").fetchone()[0]
    staked = conn.execute("SELECT COALESCE(SUM(kelly_stake),0) FROM signals WHERE status='settled'").fetchone()[0]
    clv_avg = conn.execute("SELECT AVG(clv_pct) FROM signals WHERE clv_pct IS NOT NULL").fetchone()[0]
    clv_count = conn.execute("SELECT COUNT(*) FROM signals WHERE clv_pct IS NOT NULL").fetchone()[0]
    conn.close()
    st.metric("Signals", total); st.metric("Pending", pending); st.metric("Settled", settled)
    if settled > 0:
        st.metric("Yield", f"{profit/staked*100:+.2f}%" if staked > 0 else "—")
        st.metric("Hit Rate", f"{wins/settled*100:.1f}%")
        st.metric("Avg CLV", f"{clv_avg:+.2f}%" if clv_avg else "—")
        st.caption(f"({clv_count} with CLV data)")

# Load signals from DB
if "signals_df" not in st.session_state or st.session_state.signals_df is None:
    conn = get_db()
    query = "SELECT * FROM signals"
    where_clauses = []
    if not show_settled:
        where_clauses.append("status='pending'")
    if sel_leagues:
        leagues_sql = ",".join([f"'{l}'" for l in sel_leagues])
        where_clauses.append(f"league IN ({leagues_sql})")
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    query += " ORDER BY edge_pct DESC"
    df = pd.read_sql(query, conn)
    conn.close()
    st.session_state.signals_df = df

df = st.session_state.signals_df.copy()

if len(df) == 0:
    st.warning("No signals found. Click '⚡ Trigger Scheduler Fetch' in the sidebar.")
    st.stop()

# Apply filters
if sel_markets:
    prefixes = {"1X2": "1X2", "O/U 2.5": "O/U 2.5", "AH": "AH"}
    df = df[df["market"].str.startswith(tuple([prefixes[m] for m in sel_markets]))]
if min_odds_f > 1.0:
    df = df[df["best_odds"] >= min_odds_f]
if min_edge_f > 0:
    df = df[df["edge_pct"] >= min_edge_f]
if only_bets:
    df = df[df["is_bet"] == 1]

df["match"] = df["home_team"] + " vs " + df["away_team"]

# Main table — show all signals with merged bookmaker odds
st.subheader(f"📊 Bet Signals ({len(df)} signals)")

# For each row, fetch all bookmaker odds to show the merged table
@st.cache_data(ttl=60)
def get_merged_odds_df(signal_ids, homes, aways, markets, sides, lines):
    """Get merged bookmaker odds for a list of signals. Cached 60s."""
    rows = []
    for i, (sid, h, a, m, s, l) in enumerate(zip(signal_ids, homes, aways, markets, sides, lines)):
        bookmaker_odds = get_all_bookmaker_odds_for_signal(h, a, m, s, l)
        row = {"id": sid}
        for bk in ["pinnacle", "williamhill", "draftkings", "fanduel"]:
            row[bk] = bookmaker_odds.get(bk)
        rows.append(row)
    return pd.DataFrame(rows)

# Fetch merged odds (only for displayed rows to keep it fast)
display_df = df.head(100).copy()  # cap at 100 rows for performance
merged = get_merged_odds_df(
    display_df["id"].tolist(),
    display_df["home_team"].tolist(),
    display_df["away_team"].tolist(),
    display_df["market"].tolist(),
    display_df["side"].tolist(),
    display_df["line"].tolist(),
)
display_df = display_df.merge(merged, on="id", how="left")

def color_edge(v):
    if v is None: return ""
    if v > 5: return "background-color: #c8e6c9"
    elif v > 0: return "background-color: #fff9c4"
    return "background-color: #ffcdd2"

def color_clv(v):
    if v is None or pd.isna(v): return ""
    if v > 0: return "background-color: #c8e6c9"  # beat the closing = green
    elif v < -2: return "background-color: #ffcdd2"
    return "background-color: #fff9c4"

# Select columns to display
display_cols = ["league", "kickoff", "match", "market", "side", "model_prob", "fair_odds",
                "min_odds", "pinnacle", "williamhill", "draftkings", "fanduel",
                "best_odds", "bookmaker", "edge_pct", "closing_odds", "clv_pct",
                "kelly_stake", "status"]
display_cols = [c for c in display_cols if c in display_df.columns]

st.dataframe(
    display_df[display_cols].style.map(color_edge, subset=["edge_pct"])
                                .map(color_clv, subset=["clv_pct"]),
    use_container_width=True,
    column_config={
        "model_prob": st.column_config.NumberColumn("Model %", format="%.1f%%"),
        "fair_odds": st.column_config.NumberColumn("Fair Odds", format="%.2f"),
        "min_odds": st.column_config.NumberColumn("Min Odds to Bet", format="%.2f",
            help="Your bookie must offer ABOVE this"),
        "pinnacle": st.column_config.NumberColumn("Pinnacle", format="%.2f"),
        "williamhill": st.column_config.NumberColumn("William Hill", format="%.2f"),
        "draftkings": st.column_config.NumberColumn("DraftKings", format="%.2f"),
        "fanduel": st.column_config.NumberColumn("FanDuel", format="%.2f"),
        "best_odds": st.column_config.NumberColumn("Best Odds", format="%.2f"),
        "edge_pct": st.column_config.NumberColumn("Edge %", format="%+.2f%%"),
        "closing_odds": st.column_config.NumberColumn("Closing", format="%.2f"),
        "clv_pct": st.column_config.NumberColumn("CLV %", format="%+.2f%%",
            help="Closing Line Value — positive means you beat the closing line"),
        "kelly_stake": st.column_config.NumberColumn("Stake (Toman)", format="%,.0f"),
    },
    hide_index=True,
)

# Bet signal cards
bet_df = display_df[display_df["is_bet"] == 1]
if len(bet_df) > 0:
    st.subheader(f"✅ Active Signals ({len(bet_df)} bets)")
    for _, row in bet_df.head(15).iterrows():
        c1, c2, c3, c4 = st.columns(4)
        c1.write(f"**{row['match']}**")
        c1.caption(f"{row['league']} · {row['kickoff']}")
        c2.write(f"**{row['market']} {row['side']}**")
        c2.caption(f"@ {row['best_odds']:.2f} ({row['bookmaker']})")
        c3.write(f"**Edge: {row['edge_pct']:+.2f}%**")
        c3.caption(f"Model: {row['model_prob']:.1f}% | Fair: {row['fair_odds']:.2f}")
        c4.write(f"**Min odds: {row['min_odds']:.2f}**")
        clv_str = f"CLV: {row['clv_pct']:+.2f}%" if pd.notna(row.get('clv_pct')) else "CLV: —"
        c4.caption(f"{clv_str} | closing: {row['closing_odds']:.2f}" if pd.notna(row.get('closing_odds')) else "Closing: not yet captured")

# Performance dashboard
st.divider()
st.subheader("📈 Performance Dashboard")
conn = get_db()
try:
    settled_df = pd.read_sql("SELECT * FROM signals WHERE status='settled' AND won IS NOT NULL", conn)
    if len(settled_df) > 0:
        c1, c2, c3, c4 = st.columns(4)
        total_profit = settled_df['profit'].sum()
        total_staked = settled_df['kelly_stake'].sum()
        c1.metric("Total Profit", f"{total_profit:+,.0f}")
        yld = total_profit / total_staked * 100 if total_staked > 0 else 0
        c2.metric("Yield", f"{yld:+.2f}%")
        c3.metric("Hit Rate", f"{settled_df['won'].mean()*100:.1f}%")
        clv_avg = settled_df['clv_pct'].dropna().mean() if 'clv_pct' in settled_df.columns else None
        c4.metric("Avg CLV", f"{clv_avg:+.2f}%" if clv_avg is not None else "—")

        # Per-market breakdown
        st.write("**Per-market:**")
        ms = settled_df.groupby("market").agg(bets=("id","count"), profit=("profit","sum"),
            stake=("kelly_stake","sum"), wins=("won","sum"),
            avg_clv=("clv_pct", "mean"))
        ms["yield"] = ms["profit"] / ms["stake"] * 100
        ms["hit_rate"] = ms["wins"] / ms["bets"] * 100
        st.dataframe(ms)

        # Per-bookmaker breakdown (which book gives us best odds)
        st.write("**Per-bookmaker (best odds source):**")
        bk_df = settled_df.groupby("bookmaker").agg(bets=("id","count"),
            profit=("profit","sum"), avg_edge=("edge_pct", "mean"),
            avg_clv=("clv_pct", "mean"))
        st.dataframe(bk_df)

        # Cumulative profit chart
        settled_df = settled_df.sort_values("date")
        settled_df["cum_profit"] = settled_df["profit"].cumsum()
        st.line_chart(settled_df[["cum_profit"]], use_container_width=True)
    else:
        st.info("No settled bets yet. The scheduler auto-settles ~90 min after kickoff.")
except Exception as e:
    st.info(f"Performance tracking appears after matches are settled. ({e})")
finally: conn.close()

# Bankroll strategy comparison
st.divider()
st.subheader("💰 Bankroll Strategy Comparison")
st.caption("Compare staking strategies on settled bets (if any)")
strategies = {"¼ Kelly": 0.25, "½ Kelly": 0.50, "Flat 1%": 0.01, "Flat 2%": 0.02}
conn = get_db()
try:
    settled_df = pd.read_sql("SELECT * FROM signals WHERE status='settled' AND won IS NOT NULL", conn)
    if len(settled_df) > 0:
        br = 1000000
        results = {}
        for name, param in strategies.items():
            bankroll = br
            for _, b in settled_df.iterrows():
                odds = b["best_odds"]
                prob = b["model_prob"] / 100
                if "Kelly" in name:
                    kelly = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
                    stake = max(0, min(0.02, kelly * param)) * bankroll
                else:
                    stake = param * bankroll
                if b["won"] == 1: bankroll += stake * (odds - 1)
                elif b["won"] == 0: bankroll -= stake
            results[name] = {"Final Bankroll": f"{bankroll:,.0f}", "ROI": f"{(bankroll-br)/br*100:+.1f}%",
                "Growth": f"{bankroll/br:.2f}x"}
        st.dataframe(pd.DataFrame(results).T)
    else:
        st.info("Need settled bets to compare strategies.")
except: st.info("Strategy comparison available after settlement.")
finally: conn.close()

# Confidence on profitability
st.divider()
st.subheader("🎯 Confidence on Profitability")
conn = get_db()
try:
    settled_df = pd.read_sql("SELECT * FROM signals WHERE status='settled' AND won IS NOT NULL", conn)
    if len(settled_df) >= 30:
        n = len(settled_df)
        avg_odds = settled_df["best_odds"].mean()
        yield_val = settled_df["profit"].sum() / settled_df["kelly_stake"].sum() * 100
        sigma = 0.5 * (avg_odds - 1) / (n ** 0.5) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Bets (n)", n)
        c2.metric("Yield", f"{yield_val:+.2f}%")
        c3.metric("95% CI lower", f"{yield_val - 2*sigma:+.2f}%")
        if yield_val - 2*sigma > 0:
            st.success(f"✅ Edge is statistically significant at 95% confidence (2σ lower = {yield_val-2*sigma:+.2f}%)")
        else:
            st.warning(f"⚠️ Edge not yet significant at 95% (need more bets or higher yield)")
    elif len(settled_df) > 0:
        st.info(f"Only {len(settled_df)} settled bets — need 30+ for confidence analysis")
    else:
        st.info("No settled bets yet. The scheduler auto-settles after matches finish.")
except: st.info("Confidence analysis available after settlement.")
finally: conn.close()

# CLV Analysis
st.divider()
st.subheader("📊 CLV Analysis")
st.caption("Closing Line Value: positive = you beat the closing line. CLV > 1% is the gold standard — wins will follow.")
conn = get_db()
try:
    clv_df = pd.read_sql("SELECT * FROM signals WHERE clv_pct IS NOT NULL", conn)
    if len(clv_df) > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("Bets with CLV", len(clv_df))
        c2.metric("Avg CLV", f"{clv_df['clv_pct'].mean():+.2f}%")
        positive_clv = (clv_df['clv_pct'] > 0).sum()
        c3.metric("% Beat Closing", f"{positive_clv/len(clv_df)*100:.1f}%")
        
        # CLV distribution
        st.write("**CLV distribution:**")
        clv_buckets = pd.cut(clv_df["clv_pct"], 
                              bins=[-100, -5, -2, 0, 2, 5, 100],
                              labels=["< -5%", "-5% to -2%", "-2% to 0%", "0% to +2%", "+2% to +5%", "> +5%"])
        st.bar_chart(clv_buckets.value_counts().sort_index())
        
        # CLV by market
        st.write("**CLV by market:**")
        clv_by_market = clv_df.groupby("market").agg(bets=("id", "count"),
            avg_clv=("clv_pct", "mean"),
            pct_positive=("clv_pct", lambda x: (x > 0).mean() * 100))
        st.dataframe(clv_by_market)
    else:
        st.info("No CLV data yet. The scheduler captures closing odds 30 min before kickoff.")
except: st.info("CLV analysis available after closing odds are captured.")
finally: conn.close()

# Scheduler log tail
st.divider()
st.subheader("📋 Recent Scheduler Activity")
try:
    if SCHEDULER_LOG.exists():
        with open(SCHEDULER_LOG) as f:
            lines = f.readlines()[-30:]
        st.code("".join(lines), language="text")
    else:
        st.info("No scheduler log yet. Start the scheduler with: python scripts/40_live_scheduler.py")
except Exception as e:
    st.info(f"Could not read scheduler log: {e}")

st.divider()
st.caption("⚠️ Paper trading only · Auto-fetches every hour · Auto-settles ~90 min after kickoff · Auto-refreshes daily at 09:00 Iran time")
