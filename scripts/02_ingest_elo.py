"""
02_ingest_elo.py  (revised)
--------------------------
Compute Elo ratings from the match data we already have, rather than
downloading from clubelo.com (which now requires login).

This is actually cleaner:
- Full point-in-time integrity (we control the update rule)
- No external dependency
- Reproducible — same matches always give same Elo ratings
- Can be cross-checked against clubelo.com publicly-visible ratings later

Implementation: classic Elo with goal-difference adjustment (Lasek et al. 2013).
- Initial rating: 1500 for all teams
- Expected score: E_A = 1 / (1 + 10^((R_B - R_A - HFA) / 400))
  (HFA = home field advantage in Elo points, ~65)
- K factor: 18 base, scaled by goal difference: K * (1 + 0.7 * (|gd| - 1) if |gd| > 1)
- Update: R_new = R + K * (S - E)
- New teams enter at 1500 (the league mean)

For multi-league play, we compute Elo per league (cross-league play like UCL
would need a unified Elo, but Phase 0 has domestic only).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from soccer_model.config import LEAGUES, SEASONS, MATCHES_DIR, ELO_DIR
# Import canonical Elo constants from elo_constants.py so this script's
# output is consistent with what dixon_coles.init_dynamic_from_elo expects.
from soccer_model.elo_constants import INITIAL_RATING, HOME_ADVANTAGE_ELO

# ---------------------------------------------------------------------------
# Elo parameters (Lasek, Szlavik, Bhulai 2013 — well-cited setup)
# INITIAL_RATING and HOME_ADVANTAGE_ELO are imported from elo_constants.py
# to ensure consistency across the codebase. Other K-factor parameters are
# specific to the self-computed Elo and don't need to be shared.
# ---------------------------------------------------------------------------
K_BASE = 18.0               # base K factor
K_GOAL_DIFF_FACTOR = 0.7    # extra weight per additional goal in margin
MAX_GOAL_DIFF_WEIGHT = 4    # cap on goal-difference multiplier
LEAGUE_SWITCH_PENALTY = 0  # placeholder for cross-league adjustment (Phase 1)


def expected_score(rating_a: float, rating_b: float, is_home_a: bool) -> float:
    """E_A = 1 / (1 + 10^((R_B - R_A - HFA*a) / 400))"""
    hfa = HOME_ADVANTAGE_ELO if is_home_a else 0.0
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a - hfa) / 400.0))


def k_factor(goal_diff: int) -> float:
    """K factor with goal-difference adjustment."""
    gd = abs(goal_diff)
    multiplier = 1.0 + K_GOAL_DIFF_FACTOR * max(0, gd - 1)
    multiplier = min(multiplier, MAX_GOAL_DIFF_WEIGHT)
    return K_BASE * multiplier


def actual_score(home_goals: int, away_goals: int) -> tuple[float, float]:
    """(S_home, S_away) — 1 for win, 0.5 for draw, 0 for loss."""
    if home_goals > away_goals:
        return 1.0, 0.0
    elif home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def compute_elo_for_league(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through matches chronologically, compute Elo ratings per team per day.
    Returns long-format DataFrame: (date, team, elo).
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    ratings = {}  # team -> current rating
    history = []  # list of (date, team, elo_after)

    # Initialize at start of season — teams not yet seen get 1500
    for _, m in matches.iterrows():
        home = m["home_team"]
        away = m["away_team"]
        hg = m["ft_home_goals"]
        ag = m["ft_away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue

        r_home = ratings.get(home, INITIAL_RATING)
        r_away = ratings.get(away, INITIAL_RATING)

        e_home = expected_score(r_home, r_away, is_home_a=True)
        e_away = 1.0 - e_home
        s_home, s_away = actual_score(int(hg), int(ag))

        k = k_factor(int(hg) - int(ag))
        new_r_home = r_home + k * (s_home - e_home)
        new_r_away = r_away + k * (s_away - e_away)

        ratings[home] = new_r_home
        ratings[away] = new_r_away

        # Record the rating AFTER this match (as-of for the next match's prediction)
        match_date = m["date"]
        history.append({"date": match_date, "team": home, "elo": new_r_home,
                        "opp": away, "opp_elo_pre": r_away, "is_home": True,
                        "goals_for": int(hg), "goals_against": int(ag),
                        "season": m["season"], "league": m["league"]})
        history.append({"date": match_date, "team": away, "elo": new_r_away,
                        "opp": home, "opp_elo_pre": r_home, "is_home": False,
                        "goals_for": int(ag), "goals_against": int(hg),
                        "season": m["season"], "league": m["league"]})

    return pd.DataFrame(history)


def main():
    print("=" * 70)
    print("Phase 0 — Step 2: Computing Elo ratings from match data")
    print("=" * 70)

    matches_path = MATCHES_DIR / "ALL_matches.csv"
    if not matches_path.exists():
        print("[fatal] No matches found — run 01_ingest_matches.py first")
        return

    matches = pd.read_csv(matches_path, low_memory=False)
    matches["date"] = pd.to_datetime(matches["date"])
    print(f"\nLoaded {len(matches):,} matches")

    # Compute Elo per league (each league is its own Elo pool)
    all_elo = []
    for league_key in LEAGUES.keys():
        league_matches = matches[matches["league"] == league_key].copy()
        if league_matches.empty:
            continue
        elo_df = compute_elo_for_league(league_matches)
        all_elo.append(elo_df)
        print(f"  {league_key}: {len(elo_df):,} Elo updates, "
              f"{elo_df['team'].nunique()} teams")

    elo_all = pd.concat(all_elo, ignore_index=True)
    elo_all = elo_all.sort_values(["team", "date"]).reset_index(drop=True)

    # Save
    out_csv = ELO_DIR / "ALL_elo.csv"
    elo_all.to_csv(out_csv, index=False)
    try:
        elo_all.to_parquet(ELO_DIR / "ALL_elo.parquet", index=False)
    except Exception:
        pass

    print(f"\n  → {out_csv.name}: {len(elo_all):,} Elo update records")
    print(f"    Teams: {elo_all['team'].nunique()}")
    print(f"    Date range: {elo_all['date'].min().date()} → {elo_all['date'].max().date()}")
    print(f"    Elo range: {elo_all['elo'].min():.0f} – {elo_all['elo'].max():.0f}")

    # Sanity check: spot-check a few teams' final Elo (should roughly match public perception)
    print("\n  Final Elo ratings (most recent match per team):")
    latest = elo_all.sort_values("date").groupby("team").tail(1)
    # Compare against expected ranking — top clubs should be 1700-1900, mid 1450-1600, lower 1300-1450
    sample_teams = ["Man City", "Bayern Munich", "Real Madrid", "Liverpool",
                    "Arsenal", "Inter", "PSG", "Barcelona"]
    for t in sample_teams:
        sub = latest[latest["team"] == t]
        if not sub.empty:
            print(f"    {t:20s}: Elo = {sub.iloc[0]['elo']:.0f}  ({sub.iloc[0]['date'].date()})")

    print("\n" + "=" * 70)
    print("ELO COMPUTATION COMPLETE.")
    print(f"Output: {out_csv}")


if __name__ == "__main__":
    main()
