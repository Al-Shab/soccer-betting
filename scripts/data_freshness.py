"""
data_freshness.py — Track last update time for each data source.

Provides functions to:
- Record when each data source was last refreshed (sqlite table: data_freshness)
- Query freshness for UI display
- Manual trigger to refresh individual sources

Data sources tracked:
- understat — NPxG / deep / PPDA via soccerdata
- beyondelo — Elo ratings + match history
- footballdata — football-data.co.uk CSVs (match results + shots/corners)
- odds_api — The Odds API credits remaining
- model — LGBM model + DC state + calibrators retrain time
- settle — last auto-settlement run
- closing — last closing-odds capture run
"""
import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path("/home/z/my-project")
DB_PATH = PROJECT_ROOT / "data" / "live_signals.db"

DATA_SOURCES = {
    "understat": {
        "label": "Understat (NPxG)",
        "file": "data/raw/xg/understat_fresh.csv",
        "description": "Per-match NPxG, deep completions, PPDA — used for rolling features",
    },
    "beyondelo": {
        "label": "beyondELO",
        "file": "data/raw/elo/beyondelo_history.csv",
        "description": "Cross-league Elo ratings + match history (UCL/UEL/UECL)",
    },
    "footballdata": {
        "label": "football-data.co.uk",
        "file": "data/raw/matches/fresh",  # directory — newest file mtime
        "description": "Match results + shots/corners (CSVs from football-data.co.uk)",
    },
    "matches_features": {
        "label": "Matches + Features",
        "file": "data/processed/matches_with_features.csv",
        "description": "Combined historical + fresh matches with rolling features",
    },
    "lgbm_model": {
        "label": "LGBM Model",
        "file": "models/lgbm_home.txt",
        "description": "LightGBM home/away goals model + DC state + calibrators",
    },
    "dc_state": {
        "label": "DC State",
        "file": "models/dc_state.pkl",
        "description": "Dynamic Dixon-Coles attack/defence ratings",
    },
    "calibrators": {
        "label": "Calibrators",
        "file": "models/calibrators.pkl",
        "description": "Per-market IsotonicRegression calibrators",
    },
    "team_features": {
        "label": "Team Features",
        "file": "models/team_features.pkl",
        "description": "Latest rolling features per team (used by live model)",
    },
}


def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_freshness_table():
    """Create the data_freshness table if it doesn't exist."""
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS data_freshness (
        source TEXT PRIMARY KEY,
        last_updated TEXT,
        last_status TEXT DEFAULT 'ok',
        last_message TEXT,
        records INTEGER,
        updated_at TEXT DEFAULT (datetime('now')))""")
    conn.commit()
    conn.close()


def record_refresh(source: str, status: str = "ok", message: str = "", records: int = 0):
    """Record that a data source was just refreshed."""
    init_freshness_table()
    conn = get_db()
    conn.execute("""INSERT INTO data_freshness (source, last_updated, last_status, last_message, records, updated_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                    ON CONFLICT(source) DO UPDATE SET
                      last_updated = excluded.last_updated,
                      last_status = excluded.last_status,
                      last_message = excluded.last_message,
                      records = excluded.records,
                      updated_at = excluded.updated_at""",
                 (source, datetime.utcnow().isoformat(), status, message, records))
    conn.commit()
    conn.close()


def get_freshness(source: str = None):
    """Get freshness info for one or all sources.
    
    Returns:
        - If source is None: dict of all sources {source: {last_updated, last_status, last_message, records, age_hours}}
        - If source is specified: dict for that source only
    """
    init_freshness_table()
    conn = get_db()
    
    if source:
        row = conn.execute("SELECT * FROM data_freshness WHERE source=?", (source,)).fetchone()
        conn.close()
        if not row:
            return None
        # Compute age
        last = datetime.fromisoformat(row["last_updated"])
        age = datetime.utcnow() - last
        return {
            "source": source,
            "last_updated": row["last_updated"],
            "last_status": row["last_status"],
            "last_message": row["last_message"],
            "records": row["records"],
            "age_hours": age.total_seconds() / 3600,
            "age_human": _humanize_age(age),
        }
    
    rows = conn.execute("SELECT * FROM data_freshness").fetchall()
    conn.close()
    result = {}
    for row in rows:
        try:
            last = datetime.fromisoformat(row["last_updated"])
            age = datetime.utcnow() - last
            result[row["source"]] = {
                "source": row["source"],
                "last_updated": row["last_updated"],
                "last_status": row["last_status"],
                "last_message": row["last_message"],
                "records": row["records"],
                "age_hours": age.total_seconds() / 3600,
                "age_human": _humanize_age(age),
            }
        except:
            continue
    return result


def get_file_freshness(source_key: str):
    """Get file mtime-based freshness for sources that have a file path.
    
    This complements the DB-based freshness — useful for sources that haven't
    been refreshed via the scheduler yet (e.g., historical data).
    """
    if source_key not in DATA_SOURCES:
        return None
    file_path = PROJECT_ROOT / DATA_SOURCES[source_key]["file"]
    if file_path.is_dir():
        # Find newest file in directory
        try:
            files = list(file_path.glob("*.csv"))
            if not files:
                return None
            newest = max(files, key=lambda p: p.stat().st_mtime)
            # Use tz-aware UTC to avoid timezone bugs on non-UTC systems.
            # datetime.fromtimestamp() returns NAIVE local time; subtracting
            # from datetime.utcnow() (naive UTC) gives a wrong delta by the
            # system's UTC offset.
            from datetime import timezone
            mtime = datetime.fromtimestamp(newest.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None
    elif file_path.exists():
        from datetime import timezone
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=timezone.utc)
    else:
        return None
    # Use tz-aware UTC for both sides to avoid negative ages on non-UTC systems.
    from datetime import timezone
    age = datetime.now(timezone.utc) - mtime
    return {
        "source": source_key,
        "file_mtime": mtime.replace(tzinfo=None).isoformat(),  # strip tz for display
        "age_hours": max(0, age.total_seconds() / 3600),  # clamp at 0 to avoid negatives
        "age_human": _humanize_age(age),
        "file_path": str(file_path),
    }


def _humanize_age(age):
    """Convert age timedelta to human-readable string.
    
    Clamps at 0 to handle negative ages (e.g., from clock skew or timezone bugs).
    """
    seconds = max(0, int(age.total_seconds()))
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def get_all_freshness():
    """Get combined freshness from both DB (manual refreshes) and file mtime.
    
    Returns list of dicts with: source, label, description, last_updated (DB or file),
    age_human, status, records, file_path
    """
    init_freshness_table()
    db_freshness = get_freshness()
    result = []
    for source_key, info in DATA_SOURCES.items():
        db_data = db_freshness.get(source_key, {})
        file_data = get_file_freshness(source_key)
        # Prefer DB last_updated if more recent than file mtime, else use file mtime
        if db_data and (not file_data or db_data["age_hours"] < file_data["age_hours"]):
            last_updated = db_data["last_updated"]
            age_human = db_data["age_human"]
            status = db_data["last_status"]
            records = db_data["records"]
            message = db_data.get("last_message", "")
        elif file_data:
            last_updated = file_data["file_mtime"]
            age_human = file_data["age_human"]
            status = "ok"  # file exists
            records = None
            message = ""
        else:
            last_updated = None
            age_human = "never"
            status = "missing"
            records = None
            message = ""
        result.append({
            "source": source_key,
            "label": info["label"],
            "description": info["description"],
            "last_updated": last_updated,
            "age_human": age_human,
            "status": status,
            "records": records,
            "message": message,
            "file_path": info["file"],
        })
    return result


if __name__ == "__main__":
    # Print freshness for all sources
    print("=" * 80)
    print("DATA FRESHNESS REPORT")
    print("=" * 80)
    for r in get_all_freshness():
        print(f"\n{r['label']} ({r['source']})")
        print(f"  Status: {r['status']}")
        print(f"  Last updated: {r['last_updated'] or 'never'} ({r['age_human']})")
        if r['records']:
            print(f"  Records: {r['records']:,}")
        if r['message']:
            print(f"  Message: {r['message']}")
        print(f"  Description: {r['description']}")
        print(f"  File: {r['file_path']}")
