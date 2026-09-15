"""
features.py — Phase 0 v3: Per-team rolling features (exponential decay).

Compute point-in-time rolling averages of the user-requested metrics:
- NPxG (for, against)
- NPxGD (diff = for - against)
- Deep / touches in box (for, against)
- xpts (expected points)
- PPDA, OPPDA
- Shots (for, against)
- Corners (for, against)

Key design decisions:
1. Per-team long format — each team's match contributes one row, with
   'for' = team's metric, 'against' = opponent's metric (same match).
2. Exponential decay with half-life = 10 matches. Recent matches weight ~2x
   of matches 10 games back. Replaces the v0.1 design's arbitrary 5/10/20
   rolling windows — the expert review's recommendation.
3. Strictly point-in-time: rolling features for match m only include matches
   BEFORE m (in date order). The features for match m's prediction are
   computed using matches m-1, m-2, ... with exponential decay.
4. Output: per-match feature table that joins to matches_enriched by
   (date, league, home_team, away_team). Features are suffixed _home / _away
   so the model can use them directly.

The features table is then fed to the LightGBM head on top of the dynamic
Dixon-Coles latent strengths.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable

HALF_LIFE = 10  # 10 matches


def per_team_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (match × 2 teams) wide format to (team per match) long format.

    Each match contributes two rows:
    - one for the home team with is_home=True
    - one for the away team with is_home=False

    For each row:
    - 'for' metric = team's value (e.g., home_npxg for home team)
    - 'against' metric = opponent's value (e.g., away_npxg for home team)

    We retain home_team and away_team columns for both rows (used when joining
    back to the matches table).

    v5: now includes game-state-adjusted xG columns (xg_when_level, etc.) if
    they're in the matches table.
    """
    base_cols = [
        "date", "league", "season", "home_team", "away_team",
        "ft_home_goals", "ft_away_goals",
        "home_npxg", "away_npxg",
        "home_npxgd", "away_npxgd",
        "home_xpts", "away_xpts",
        "home_deep", "away_deep",
        "home_ppda", "away_ppda",
        "home_shots", "away_shots",
        "home_corners", "away_corners",
    ]
    # Game-state-adjusted columns (v5) — only include if present
    game_state_cols = [
        "home_xg_when_level", "away_xg_when_level",
        "home_xg_when_leading_1", "away_xg_when_leading_1",
        "home_xg_when_trailing_1", "away_xg_when_trailing_1",
        "home_xg_when_level_open_play", "away_xg_when_level_open_play",
    ]
    base_cols.extend([c for c in game_state_cols if c in matches.columns])
    base = matches[base_cols].copy()

    # Compute BTTS outcome per match (yes = both teams scored)
    base["match_btts_yes"] = (
        (base["ft_home_goals"] > 0) & (base["ft_away_goals"] > 0)
    ).astype(int)

    # Build rename dicts for home and away perspectives
    home_rename = {
        "home_team": "team", "away_team": "opp",
        "ft_home_goals": "goals_for", "ft_away_goals": "goals_against",
        "home_npxg": "npxg_for", "away_npxg": "npxg_against",
        "home_npxgd": "npxgd_for", "away_npxgd": "npxgd_against",
        "home_xpts": "xpts_for", "away_xpts": "xpts_against",
        "home_deep": "deep_for", "away_deep": "deep_against",
        "home_ppda": "ppda_for", "away_ppda": "oppda_against",
        "home_shots": "shots_for", "away_shots": "shots_against",
        "home_corners": "corners_for", "away_corners": "corners_against",
        "match_btts_yes": "btts_for",
        # Game-state xG — swap for/against for away perspective
        "home_xg_when_level": "xg_when_level_for", "away_xg_when_level": "xg_when_level_against",
        "home_xg_when_leading_1": "xg_when_leading_1_for", "away_xg_when_leading_1": "xg_when_leading_1_against",
        "home_xg_when_trailing_1": "xg_when_trailing_1_for", "away_xg_when_trailing_1": "xg_when_trailing_1_against",
        "home_xg_when_level_open_play": "xg_when_level_op_for", "away_xg_when_level_open_play": "xg_when_level_op_against",
    }
    away_rename = {
        "home_team": "opp", "away_team": "team",
        "ft_home_goals": "goals_against", "ft_away_goals": "goals_for",
        "home_npxg": "npxg_against", "away_npxg": "npxg_for",
        "home_npxgd": "npxgd_against", "away_npxgd": "npxgd_for",
        "home_xpts": "xpts_against", "away_xpts": "xpts_for",
        "home_deep": "deep_against", "away_deep": "deep_for",
        "home_ppda": "oppda_against", "away_ppda": "ppda_for",
        "home_shots": "shots_against", "away_shots": "shots_for",
        "home_corners": "corners_against", "away_corners": "corners_for",
        "match_btts_yes": "btts_for",
        # Game-state xG — for away perspective, away team's level xG is "for", home's is "against"
        "home_xg_when_level": "xg_when_level_against", "away_xg_when_level": "xg_when_level_for",
        "home_xg_when_leading_1": "xg_when_trailing_1_against", "away_xg_when_leading_1": "xg_when_trailing_1_for",
        "home_xg_when_trailing_1": "xg_when_leading_1_against", "away_xg_when_trailing_1": "xg_when_leading_1_for",
        "home_xg_when_level_open_play": "xg_when_level_op_against", "away_xg_when_level_open_play": "xg_when_level_op_for",
    }

    home_df_h = base.rename(columns=home_rename)
    home_df_h["is_home"] = True
    home_df_h["match_home_team"] = base["home_team"]
    home_df_h["match_away_team"] = base["away_team"]

    home_df_a = base.rename(columns=away_rename)
    home_df_a["is_home"] = False
    home_df_a["match_home_team"] = base["home_team"]
    home_df_a["match_away_team"] = base["away_team"]

    long_df = pd.concat([home_df_h, home_df_a], ignore_index=True)
    long_df["date"] = pd.to_datetime(long_df["date"])
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)
    return long_df


def exponential_decay_rolling(df: pd.DataFrame, value_col: str,
                                half_life: int = HALF_LIFE) -> pd.Series:
    """
    Compute exponential decay rolling average for a column.
    The result at row i is the weighted mean of rows 0..i-1 (strictly before),
    with weights w_j = 0.5^(i-j-1)/half_life.
    """
    values = df[value_col].values
    n = len(values)
    result = np.full(n, np.nan)

    # Cumulative weighted sums (excluding current row)
    decay = 0.5 ** (1.0 / half_life)
    weight_sum = 0.0
    weighted_sum = 0.0
    for i in range(n):
        if i > 0:
            # Set result[i] using previous values
            if weight_sum > 0:
                result[i] = weighted_sum / weight_sum
            else:
                result[i] = np.nan
        # Now include current row for future iterations
        v = values[i]
        if not np.isnan(v):
            # Apply decay to existing weights, then add current
            weight_sum *= decay
            weighted_sum *= decay
            weight_sum += 1.0
            weighted_sum += v
    return pd.Series(result, index=df.index)


def compute_team_rolling_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute rolling exponential-decay averages of all
    metrics. Point-in-time: row i uses only matches BEFORE i.
    """
    feature_cols = [
        "npxg_for", "npxg_against",
        "npxgd_for", "npxgd_against",
        "xpts_for", "xpts_against",
        "deep_for", "deep_against",
        "ppda_for", "oppda_against",
        "shots_for", "shots_against",
        "corners_for", "corners_against",
        "goals_for", "goals_against",
        "btts_for",
        # Game-state-adjusted xG (v5) — only included if present
        "xg_when_level_for", "xg_when_level_against",
        "xg_when_leading_1_for", "xg_when_leading_1_against",
        "xg_when_trailing_1_for", "xg_when_trailing_1_against",
        "xg_when_level_op_for", "xg_when_level_op_against",
    ]
    # Group by team, compute rolling for each
    result_frames = []
    for team, grp in long_df.groupby("team", sort=False):
        grp = grp.sort_values("date").copy()
        for col in feature_cols:
            if col in grp.columns:
                grp[f"rolling_{col}"] = exponential_decay_rolling(grp, col, HALF_LIFE)
        # Count of past matches (for cold-start awareness)
        grp["matches_history"] = range(len(grp))
        result_frames.append(grp)
    return pd.concat(result_frames, ignore_index=True)


def wide_format_features(long_df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """
    Convert back to wide format: one row per match with home_* and away_*
    rolling features.
    """
    # Split home/away using the is_home flag (we kept match_home_team/match_away_team
    # so we can join back to the original matches)
    home_features = long_df[long_df["is_home"]].copy()
    away_features = long_df[~long_df["is_home"]].copy()

    # Select rolling columns
    rolling_cols = [c for c in long_df.columns if c.startswith("rolling_")]

    # For home features, the rolling_* columns are home team's metrics.
    # For away features, the rolling_* columns are away team's metrics.
    # We rename to make them distinct in the merged frame.

    # Home features: keep join keys + rolling cols + history count
    keep_home = ["date", "league", "match_home_team", "match_away_team"] + rolling_cols + ["matches_history"]
    home_features = home_features[keep_home].rename(
        columns={c: c.replace("rolling_", "rolling_home_") for c in rolling_cols}
        | {"matches_history": "home_matches_history",
           "match_home_team": "home_team",
           "match_away_team": "away_team"}
    )

    # Away features
    keep_away = ["date", "league", "match_home_team", "match_away_team"] + rolling_cols + ["matches_history"]
    away_features = away_features[keep_away].rename(
        columns={c: c.replace("rolling_", "rolling_away_") for c in rolling_cols}
        | {"matches_history": "away_matches_history",
           "match_home_team": "home_team",
           "match_away_team": "away_team"}
    )

    # Merge — both frames have (date, league, home_team, away_team) as keys
    merged = home_features.merge(
        away_features,
        on=["date", "league", "home_team", "away_team"],
        how="outer",
    )
    return merged


def main():
    print("=" * 70)
    print("Phase 0 — Step 12: Compute rolling features")
    print("=" * 70)

    from soccer_model.config import DATA_PROCESSED, DATA_RAW
    matches_path = DATA_PROCESSED / "matches_enriched.csv"
    if not matches_path.exists():
        print("[fatal] No enriched matches — run scripts 09 + 11 first")
        return

    matches = pd.read_csv(matches_path, low_memory=False)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    print(f"\nLoaded {len(matches):,} enriched matches")

    # Convert to long format (one row per team per match)
    print("\nConverting to per-team long format...")
    long_df = per_team_long_format(matches)
    print(f"  Long format: {len(long_df):,} team-match records, {long_df['team'].nunique()} unique teams")

    # Compute rolling exponential-decay averages per team
    print("\nComputing rolling exponential-decay features (half-life = 10 matches)...")
    long_df_with_features = compute_team_rolling_features(long_df)
    print(f"  Done. Feature columns: {[c for c in long_df_with_features.columns if c.startswith('rolling_')]}")

    # Convert back to wide format (one row per match with home_* and away_* features)
    print("\nConverting back to wide format...")
    wide_df = wide_format_features(long_df_with_features, matches)
    print(f"  Wide format: {len(wide_df):,} matches, {len(wide_df.columns)} columns")

    # Merge features back to matches_enriched by (date, league, home_team, away_team)
    print("\nJoining features back to matches_enriched...")
    feature_cols = [c for c in wide_df.columns
                     if c.startswith("rolling_") or c.endswith("_matches_history")]
    matches_with_features = matches.merge(
        wide_df[["date", "league", "home_team", "away_team"] + feature_cols],
        on=["date", "league", "home_team", "away_team"],
        how="left",
    )

    # Save
    out_csv = DATA_PROCESSED / "matches_with_features.csv"
    matches_with_features.to_csv(out_csv, index=False)
    print(f"\n  Output: {out_csv}")
    print(f"  Total rows: {len(matches_with_features):,}")
    print(f"  Total columns: {len(matches_with_features.columns)}")

    # Coverage
    print(f"\nRolling feature coverage:")
    for c in feature_cols[:20]:  # first 20 for brevity
        if c in matches_with_features.columns:
            nn = matches_with_features[c].notna().sum()
            print(f"  {c:30s}: {nn:,}/{len(matches_with_features):,} ({nn/len(matches_with_features)*100:.1f}%)")

    # Sanity check: top team should have sensible rolling NPxG
    print(f"\nSanity check — Man City's rolling NPxG (last 5 matches):")
    team_check = matches_with_features[
        (matches_with_features["home_team"] == "Man City") &
        (matches_with_features["league"] == "EPL")
    ].sort_values("date").tail(5)
    for _, r in team_check.iterrows():
        print(f"  {r['date'].date()} vs {r['away_team']:20s}  "
              f"rolling_home_npxg_for={r['rolling_home_npxg_for']:.2f}  "
              f"rolling_home_npxg_against={r['rolling_home_npxg_against']:.2f}")


if __name__ == "__main__":
    main()
