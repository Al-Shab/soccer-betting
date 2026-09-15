"""
10_ingest_understat_soccerdata.py — Pull NPxG via soccerdata's Understat loader.

The Mexwell archive gave us xG but not NPxG. soccerdata's Understat loader
(which uses the tls_requests library, not Selenium) gives us both:
  - home_xg, away_xg (matches Mexwell)
  - home_np_xg, away_np_xg (NEW — non-penalty xG, the expert's preferred metric)

This is gold: NPxG excludes penalty xG, which is too episodic to be a stable
predictor of underlying chance generation.

We pull 8 seasons × 5 leagues = 40 (league, season) calls. Each call returns
one full season's match stats. Soccerdata caches results, so subsequent runs
are fast.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
import soccerdata

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from soccer_model.config import LEAGUES, SEASONS, XG_DIR

# Map our league keys to soccerdata's Understat league names
LEAGUE_TO_SD = {
    "EPL":        "ENG-Premier League",
    "La_Liga":    "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie_A":    "ITA-Serie A",
    "Ligue_1":    "FRA-Ligue 1",
}

# Pass season codes in multi-year format ("2122", "2223", etc.) — soccerdata's
# Understat loader interprets these unambiguously. Single-year codes like "2021"
# are ambiguous (could mean 2020/21 or 2021/22) and default to the wrong one.
def season_to_sd(season_key: str) -> str:
    """Return season key as-is (already in '2122' multi-year format)."""
    return str(season_key)


def main():
    print("=" * 70)
    print("Phase 0 — Step 10: Pull NPxG via soccerdata Understat loader")
    print("=" * 70)

    all_data = []
    for league_key, info in LEAGUES.items():
        sd_league = LEAGUE_TO_SD[league_key]
        for season in SEASONS:
            sd_season = season_to_sd(season)
            print(f"\n--- {info['name']} {season} (soccerdata: {sd_league} {sd_season}) ---")
            try:
                us = soccerdata.Understat(leagues=[sd_league], seasons=[sd_season])
                df = us.read_team_match_stats()
                if df.empty:
                    print(f"  [empty] {league_key} {season}")
                    continue
                df = df.reset_index()
                # Add our league key for joining
                df["league"] = league_key
                # Derive the canonical season key from the match date
                # (month >= 8 → start of new season in YY/YY format)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], utc=True)
                    df["season"] = df["date"].apply(
                        lambda d: f"{(d.year if d.month >= 8 else d.year - 1) % 100:02d}"
                                  f"{(d.year + 1 if d.month >= 8 else d.year) % 100:02d}"
                    )
                else:
                    df["season"] = season
                # Extract the columns we want — including xpts and NPxGD
                cols_to_keep = [
                    "league", "season", "date",
                    "home_team", "away_team",
                    "home_goals", "away_goals",
                    "home_xg", "away_xg",
                    "home_np_xg", "away_np_xg",
                    "home_np_xg_difference", "away_np_xg_difference",  # NPxGD
                    "home_expected_points", "away_expected_points",     # xpts
                    "home_ppda", "away_ppda",
                    "home_deep_completions", "away_deep_completions",
                ]
                cols_present = [c for c in cols_to_keep if c in df.columns]
                df = df[cols_present].copy()
                print(f"  [ok] {league_key} {season}: {len(df)} matches, "
                      f"NPxG coverage: {df['home_np_xg'].notna().sum()}/{len(df)}, "
                      f"date range {df['date'].min().date()} → {df['date'].max().date()}")
                all_data.append(df)
                time.sleep(1.0)  # be polite
            except Exception as e:
                print(f"  [error] {league_key} {season}: {type(e).__name__}: {e}")
                continue

    if not all_data:
        print("[fatal] No data pulled")
        return

    combined = pd.concat(all_data, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], utc=True)

    # Deduplicate: soccerdata may return the same match for different season label requests.
    # We keep first occurrence by (date, league, home_team, away_team).
    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        subset=["date", "league", "home_team", "away_team"],
        keep="first"
    ).reset_index(drop=True)
    after_dedup = len(combined)
    if before_dedup != after_dedup:
        print(f"\n  Deduplication: {before_dedup:,} → {after_dedup:,} (removed {before_dedup - after_dedup:,} duplicates)")

    combined = combined.sort_values(["league", "season", "date"]).reset_index(drop=True)

    # Save
    out_csv = XG_DIR / "understat_npxg.csv"
    combined.to_csv(out_csv, index=False)
    try:
        combined.to_parquet(XG_DIR / "understat_npxg.parquet", index=False)
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("UNDERSTAT NPxG INGESTION COMPLETE")
    print(f"  Total matches: {len(combined):,}")
    print(f"  Leagues: {sorted(combined['league'].unique().tolist())}")
    print(f"  Seasons: {sorted(combined['season'].unique().tolist())}")
    print(f"  Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")
    print(f"  Output: {out_csv}")

    # Coverage report
    print(f"\nField coverage:")
    for c in ["home_xg", "away_xg", "home_np_xg", "away_np_xg",
              "home_np_xg_difference", "away_np_xg_difference",
              "home_expected_points", "away_expected_points",
              "home_ppda", "away_ppda",
              "home_deep_completions", "away_deep_completions"]:
        if c in combined.columns:
            nn = combined[c].notna().sum()
            print(f"  {c:30s}: {nn}/{len(combined)} ({nn/len(combined)*100:.1f}%)")

    print(f"\nPer (league, season) coverage:")
    print(combined.groupby(["league", "season"]).size().to_string())


if __name__ == "__main__":
    main()
