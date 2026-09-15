"""
01_ingest_matches.py  (v0.2 — post expert review)
------------------------------------------------
Download historical match results + closing odds from football-data.co.uk
for the 5 major European leagues, 8 seasons (2016/17 to 2023/24).

Key v0.2 changes:
- PSC (Pinnacle) columns are loaded but flagged — used as reference through
  2023/24 only; excluded from 2024/25 onward (handled in market_state.py)
- All odds columns preserved (PSC, BbMx, BbAv, Betfair) so Layer 4 can compute
  multi-book consensus + dispersion
- Point-in-time discipline: matches are sorted by date within each season;
  the raw CSV is the immutable source of truth (never edited in place)
- Schema standardization handles column-name drift across seasons
"""
import os
import sys
import io
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from soccer_model.config import LEAGUES, SEASONS, MATCHES_DIR

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk"

# ---------------------------------------------------------------------------
# Standardized column schema
# ---------------------------------------------------------------------------
# Football-data.co.uk column names changed in 2017/18 (added Betfair, Asian
# handicap column renames) and again in 2019 (added more books).
# We map all known variants to a consistent internal schema.

# Match identity + scoreline
IDENT_COLS = {
    "Div": "div",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "ft_home_goals",
    "FTAG": "ft_away_goals",
    "FTR": "ft_result",
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
}

# Match stats (useful as Layer 3 features when present)
STAT_COLS = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
    "Referee": "referee",
}

# 1X2 closing odds — keep ALL books for Layer 4 multi-book consensus
ODDS_1X2_COLS = {
    # Pinnacle (PSC) — sharp reference through 2023/24, exclude post-2024/25
    "PSCH": "psc_home",
    "PSCD": "psc_draw",
    "PSCA": "psc_away",
    # Bet365
    "B365H": "b365_home",
    "B365D": "b365_draw",
    "B365A": "b365_away",
    # Pinnacle (legacy PIN/PN)
    "PSH": "ps_home_legacy",
    "PSD": "ps_draw_legacy",
    "PSA": "ps_away_legacy",
    "PINH": "pin_home_legacy",
    "PIND": "pin_draw_legacy",
    "PINA": "pin_away_legacy",
    # Market max/avg (BetBrain)
    "BbMxH": "bbmx_home",
    "BbMxD": "bbmx_draw",
    "BbMxA": "bbmx_away",
    "BbAvH": "bbav_home",
    "BbAvD": "bbav_draw",
    "BbAvA": "bbav_away",
    # Betfair exchange (closing + SP)
    "Bb1X2": "bb1x2_ref_count",
    "BbOU": "bbou_ref_count",
    "BbAH": "bbah_ref_count",
    "BFH": "bf_home",
    "BFD": "bf_draw",  # Betfair exchange closing draw
    "BFA": "bf_away",
    "BFHSp": "bf_home_sp",
    "BFDSP": "bf_draw_sp",  # NOTE: actual CSV column is "BFDSP" not "BFD"
                            # — the duplicate "BFD" key silently overwrote bf_draw.
                            # Verify the actual CSV column name before relying on this.
    "BFASp": "bf_away_sp",
    # William Hill, Ladbrokes, VC, etc. (varies by season)
    "WHH": "wh_home",
    "WHD": "wh_draw",
    "WHA": "wh_away",
    "VCH": "vc_home",
    "VCD": "vc_draw",
    "VCA": "vc_away",
    "LBH": "lb_home",
    "LBD": "lb_draw",
    "LBA": "lb_away",
    "SBOH": "sbo_home",
    "SBOD": "sbo_draw",
    "SBOA": "sbo_away",
    "GBH": "gb_home",
    "GBD": "gb_draw",
    "GBA": "gb_away",
    "BSH": "bs_home",
    "BSD": "bs_draw",
    "BSA": "bs_away",
    "SJH": "sj_home",
    "SJD": "sj_draw",
    "SJA": "sj_away",
    "IWH": "iw_home",
    "IWD": "iw_draw",
    "IWA": "iw_away",
}

# Over/Under 2.5 goals — keep all books
ODDS_OU_COLS = {
    "BbMx>2.5": "bbmx_over_25",
    "BbAv>2.5": "bbav_over_25",
    "BbMxD>2.5": "bbmx_under_25",
    "BbAvD>2.5": "bbav_under_25",
    "BbMx<2.5": "bbmx_under_25_alt",
    "BbAv<2.5": "bbav_under_25_alt",
    "B365>2.5": "b365_over_25",
    "B365<2.5": "b365_under_25",
    "P>2.5": "ps_over_25_legacy",
    "P<2.5": "ps_under_25_legacy",
    "Max>2.5": "max_over_25",
    "Max<2.5": "max_under_25",
    "Avg>2.5": "avg_over_25",
    "Avg<2.5": "avg_under_25",
}

# Asian handicap — keep all books
ODDS_AH_COLS = {
    "BbAHh": "bbah_handicap",
    "BbMxAHH": "bbmx_ah_home",
    "BbMxAHA": "bbmx_ah_away",
    "BbAvAHH": "bbav_ah_home",
    "BbAvAHA": "bbav_ah_away",
    "B365AHH": "b365_ah_home",
    "B365AHA": "b365_ah_away",
    "B365AHh": "b365_ah_handicap",
    "PAHh": "ps_ah_handicap_legacy",
    "AHH": "ps_ah_home_legacy",
    "AHA": "ps_ah_away_legacy",
    "MaxAHH": "max_ah_home",
    "MaxAHA": "max_ah_away",
    "AvgAHH": "avg_ah_home",
    "AvgAHA": "avg_ah_away",
}

# BTTS — keep all books (when present; varies by season)
ODDS_BTTS_COLS = {
    "BbMx>2.5G": "bbmx_btts_yes",   # ambiguous in some seasons
    "BbAv>2.5G": "bbav_btts_yes",
    "BbMx<2.5G": "bbmx_btts_no",
    "BbAv<2.5G": "bbav_btts_no",
    "BbMxBbts": "bbmx_btts_yes_alt",
    "BbAvBbts": "bbav_btts_yes_alt",
    "BbMxBbts<": "bbmx_btts_no_alt",
    "BbAvBbts<": "bbav_btts_no_alt",
    "BbMxBTTS": "bbmx_btts_yes",
    "BbAvBTTS": "bbav_btts_yes",
    "BbMx<BTTS": "bbmx_btts_no",
    "BbAv<BTTS": "bbav_btts_no",
}

ALL_STANDARD = {**IDENT_COLS, **STAT_COLS, **ODDS_1X2_COLS,
                **ODDS_OU_COLS, **ODDS_AH_COLS, **ODDS_BTTS_COLS}


def download_csv(league_code: str, season: str, dest_dir: Path) -> Path | None:
    """Download one CSV from football-data.co.uk."""
    url = f"{FOOTBALL_DATA_BASE}/mmz4281/{season}/{league_code}.csv"
    dest = dest_dir / f"{league_code}_{season}.csv"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  [skip] {dest.name} (already downloaded)")
        return dest
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            dest.write_bytes(r.content)
            print(f"  [ok] {dest.name} — {len(r.content):,} bytes")
            return dest
        print(f"  [fail] {league_code} {season}: HTTP {r.status_code}, {len(r.content)} bytes")
        return None
    except Exception as e:
        print(f"  [fail] {league_code} {season}: {e}")
        return None


def normalize_csv(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Rename columns to standard schema, add league/season cols, parse date."""
    # Strip BOM (byte-order mark) from first column name and strip whitespace
    df.columns = [c.strip().lstrip("\ufeff").lstrip("ï»¿") for c in df.columns]

    rename_map = {}
    for col in df.columns:
        if col in ALL_STANDARD:
            rename_map[col] = ALL_STANDARD[col]
        elif col.upper() in ALL_STANDARD:
            rename_map[col] = ALL_STANDARD[col.upper()]

    df = df.rename(columns=rename_map)

    # Handle duplicate columns from case-folding (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    keep = [c for c in ALL_STANDARD.values() if c in df.columns]
    df = df[keep].copy()

    df["league"] = league
    df["season"] = season

    # Parse date — football-data.co.uk uses DD/MM/YYYY except earliest seasons (DD/MM/YY)
    # Try the 4-digit year format first, then fall back to 2-digit year
    parsed = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
    else:
        # Some seasons have mixed formats (e.g. a few rows in DD/MM/YY); fill those
        remaining_mask = parsed.isna() & df["date"].notna()
        if remaining_mask.any():
            fallback = pd.to_datetime(df.loc[remaining_mask, "date"],
                                      format="%d/%m/%y", errors="coerce")
            parsed.loc[remaining_mask] = fallback
    df["date"] = parsed
    df = df[df["date"].notna()].copy()

    # Drop rows with missing teams (CSV artifacts)
    df = df[df["home_team"].notna() & df["away_team"].notna()].copy()
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()

    # Sort by date within season — preserves point-in-time ordering
    df = df.sort_values(["date", "div"]).reset_index(drop=True)

    # Convert numeric odds columns (stored as strings in some seasons)
    numeric_cols = [c for c in df.columns
                    if any(c.startswith(prefix) for prefix in
                           ["psc_", "b365_", "ps_", "pin_",
                            "bbmx_", "bbav_", "bf_", "max_", "avg_",
                            "ft_", "ht_", "home_", "away_"])
                    and c not in ("home_team", "away_team",
                                  "home_shots_on_target", "away_shots_on_target")]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def main():
    print("=" * 70)
    print("Phase 0 — Step 1: Ingesting match data from football-data.co.uk")
    print("(v0.2 — post expert review: keep all bookmaker columns)")
    print("=" * 70)

    all_files = []
    for league_key, info in LEAGUES.items():
        code = info["fd_code"]
        print(f"\n--- {info['name']} ({code}) ---")
        for season in SEASONS:
            f = download_csv(code, season, MATCHES_DIR)
            if f:
                all_files.append((league_key, season, f))
            time.sleep(0.3)

    # Concat per league, write consolidated CSV + Parquet
    print("\n" + "=" * 70)
    print("Normalizing and concatenating per league...")

    all_combined = []
    for league_key, info in LEAGUES.items():
        league_dfs = []
        for lk, season, f in all_files:
            if lk != league_key:
                continue
            try:
                # utf-8-sig strips BOM if present; falls back to latin-1 for some old files
                try:
                    df = pd.read_csv(f, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(f, encoding="latin-1")
                df = normalize_csv(df, league_key, season)
                print(f"  {league_key} {season}: {len(df)} matches")
                league_dfs.append(df)
            except Exception as e:
                print(f"  [error] {f.name}: {e}")

        if not league_dfs:
            continue
        full = pd.concat(league_dfs, ignore_index=True)
        all_combined.append(full)

        out_csv = MATCHES_DIR / f"{league_key}_all.csv"
        full.to_csv(out_csv, index=False)
        try:
            out_pq = MATCHES_DIR / f"{league_key}_all.parquet"
            full.to_parquet(out_pq, index=False)
        except Exception as e:
            print(f"  [warn] parquet write failed: {e}")

        print(f"  → {out_csv.name}: {len(full)} matches, "
              f"{full['season'].nunique()} seasons, "
              f"{full['date'].min().date()} → {full['date'].max().date()}")

        # PSC availability check (will be excluded in Layer 4 post-2024/25)
        psc_cols = [c for c in full.columns if c.startswith("psc_")]
        psc_nonnull = full[psc_cols[0]].notna().sum() if psc_cols else 0
        print(f"    PSC 1X2 odds: {psc_nonnull}/{len(full)} non-null")

    # Combined file across all leagues
    if all_combined:
        all_df = pd.concat(all_combined, ignore_index=True)
        all_csv = MATCHES_DIR / "ALL_matches.csv"
        all_df.to_csv(all_csv, index=False)
        print(f"\n  → {all_csv.name}: {len(all_df)} matches across all leagues")

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE.")
    print(f"Raw CSVs in: {MATCHES_DIR}")
    print(f"Consolidated CSVs/Parquet per league in: {MATCHES_DIR}")


if __name__ == "__main__":
    main()
