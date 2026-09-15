"""
lgbm_head.py — Phase 0 v3: LightGBM head on top of dynamic Dixon-Coles.

Architecture:
- Layer 1 (Dynamic Dixon-Coles) gives latent strength per team → λ_home, λ_away
- Layer 1.5 (features.py) gives rolling exponential-decay averages per team
- Layer 1+ (LightGBM head) takes (λ_home, λ_away, rolling features) and
  predicts home_goals and away_goals as Poisson targets

The LightGBM head's predicted λ values are then fed into the standard
scoreline_matrix() function to derive all markets.

Walk-forward protocol:
- Train on all matches in seasons 0..N-1
- Predict on season N
- Re-train on seasons 0..N, predict on N+1, etc.

Two separate LightGBM models:
- lgbm_home: predicts home_goals given features
- lgbm_away: predicts away_goals given features
Both use Poisson loss (objective='poisson').

Features used:
- DC latent strengths (λ_home, λ_away from Layer 1)
- Rolling features (16 features × 2 teams = 32 features)
- Match context (league encoded, home advantage, matches_history)
- Total: ~40 features per match
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from soccer_model.config import (
    DATA_PROCESSED, MATCHES_DIR, ELO_DIR, MODELS_DIR,
    DC_MAX_GOALS, KALMAN_GAIN_ATTACK, KALMAN_GAIN_DEFENCE,
)
from soccer_model.dixon_coles import (
    DynamicDCState, init_dynamic_from_elo, predict_dynamic_dc, update_dynamic_dc,
    scoreline_matrix,
)
from soccer_model.elo_constants import INITIAL_RATING


# ---------------------------------------------------------------------------
# Feature engineering for the LGBM head
# ---------------------------------------------------------------------------

LGBM_FEATURES = [
    # DC latent strengths (continuous)
    "dc_lam_home", "dc_lam_away",
    # Rolling features (16 cols × 2 teams)
    "rolling_home_npxg_for", "rolling_home_npxg_against",
    "rolling_home_npxgd_for", "rolling_home_npxgd_against",
    "rolling_home_xpts_for", "rolling_home_xpts_against",
    "rolling_home_deep_for", "rolling_home_deep_against",
    "rolling_home_ppda_for", "rolling_home_oppda_against",
    "rolling_home_shots_for", "rolling_home_shots_against",
    "rolling_home_corners_for", "rolling_home_corners_against",
    "rolling_home_goals_for", "rolling_home_goals_against",
    "rolling_away_npxg_for", "rolling_away_npxg_against",
    "rolling_away_npxgd_for", "rolling_away_npxgd_against",
    "rolling_away_xpts_for", "rolling_away_xpts_against",
    "rolling_away_deep_for", "rolling_away_deep_against",
    "rolling_away_ppda_for", "rolling_away_oppda_against",
    "rolling_away_shots_for", "rolling_away_shots_against",
    "rolling_away_corners_for", "rolling_away_corners_against",
    "rolling_away_goals_for", "rolling_away_goals_against",
    # History depth
    "home_matches_history", "away_matches_history",
    # Match context
    "is_local_derby",  # placeholder for future; for now always 0
    # Rest and congestion — GATED: only show rest days when congested (3+ games in 7d)
    # The LGBM sees rest_days only when it matters (FIFPRO threshold);
    # for normal 7-day rest the feature is 0, avoiding dilution
    "rest_days_home_gated", "rest_days_away_gated", "rest_diff_gated",
    "matches_7d_home", "matches_7d_away", "congestion_diff_7d",
    "matches_14d_home", "matches_14d_away", "congestion_diff_14d",
    "is_congested_home", "is_congested_away",  # binary flag: 3+ games in 7d
]

# v5: extended with game-state xG and rolling BTTS rate
LGBM_FEATURES_V5 = LGBM_FEATURES + [
    "rolling_home_btts_for", "rolling_away_btts_for",
    # Game-state-adjusted xG
    "rolling_home_xg_when_level_for", "rolling_home_xg_when_level_against",
    "rolling_away_xg_when_level_for", "rolling_away_xg_when_level_against",
    "rolling_home_xg_when_leading_1_for", "rolling_home_xg_when_leading_1_against",
    "rolling_away_xg_when_leading_1_for", "rolling_away_xg_when_leading_1_against",
    "rolling_home_xg_when_trailing_1_for", "rolling_home_xg_when_trailing_1_against",
    "rolling_away_xg_when_trailing_1_for", "rolling_away_xg_when_trailing_1_against",
    "rolling_home_xg_when_level_op_for", "rolling_home_xg_when_level_op_against",
    "rolling_away_xg_when_level_op_for", "rolling_away_xg_when_level_op_against",
]


def prepare_lgbm_features(matches: pd.DataFrame,
                            dc_lambdas: dict[tuple, tuple]) -> pd.DataFrame:
    """
    Build the feature matrix for LGBM training/prediction.

    dc_lambdas: dict keyed by (date, league, home_team, away_team) → (lam_home, lam_away)
    """
    feature_rows = []
    for _, m in matches.iterrows():
        key = (m["date"], m["league"], m["home_team"], m["away_team"])
        lam_h, lam_a = dc_lambdas.get(key, (1.4, 1.1))  # fallback to league average

        row = {
            "dc_lam_home": lam_h,
            "dc_lam_away": lam_a,
            "is_local_derby": 0,  # placeholder
        }
        for c in LGBM_FEATURES:
            if c in row:
                continue
            if c in matches.columns:
                row[c] = m[c]
            else:
                row[c] = np.nan
        feature_rows.append(row)
    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Walk-forward LGBM head training and prediction
# ---------------------------------------------------------------------------

class LGBMHead:
    """Two LightGBM models (home and away) trained on the same features."""

    def __init__(self, params: Optional[dict] = None, feature_names: Optional[list] = None):
        default_params = {
            "objective": "poisson",
            "metric": "poisson",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 30,
            "verbose": -1,
        }
        self.params = params or default_params
        self.feature_names = feature_names or LGBM_FEATURES
        self.model_home = None
        self.model_away = None
        self.num_boost_round = 100

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series) -> None:
        """Train both models."""
        mask = y_home.notna() & y_away.notna()
        X_clean = X[mask].copy()
        y_h = y_home[mask].astype(int)
        y_a = y_away[mask].astype(int)

        # Fill NaN features with 0 (or mean)
        X_clean = X_clean.fillna(X_clean.median(numeric_only=True))
        X_clean = X_clean.fillna(0)

        # Use the feature names that match X's columns
        actual_feature_names = list(X_clean.columns)

        # Train home model
        dtrain_h = lgb.Dataset(X_clean, label=y_h, feature_name=actual_feature_names)
        self.model_home = lgb.train(
            self.params, dtrain_h, num_boost_round=self.num_boost_round
        )

        # Train away model
        dtrain_a = lgb.Dataset(X_clean, label=y_a, feature_name=actual_feature_names)
        self.model_away = lgb.train(
            self.params, dtrain_a, num_boost_round=self.num_boost_round
        )

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lam_home, lam_away) arrays."""
        X_filled = X.fillna(X.median(numeric_only=True)).fillna(0)
        lam_h = self.model_home.predict(X_filled)
        lam_a = self.model_away.predict(X_filled)
        # Clamp to sensible range
        lam_h = np.clip(lam_h, 0.05, 8.0)
        lam_a = np.clip(lam_a, 0.05, 8.0)
        return lam_h, lam_a


# ---------------------------------------------------------------------------
# Walk-forward orchestration combining Layer 1 (DC) + LGBM head
# ---------------------------------------------------------------------------

def walk_forward_predict_with_lgbm(
    matches: pd.DataFrame,
    elo_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Walk-forward prediction:
    1. For each match in chronological order:
       a. Compute DC lambdas using current latent strengths
       b. Build features (DC lambdas + rolling features)
       c. If LGBM head is available, predict final lambdas using it
       d. Otherwise use DC lambdas directly
       e. Derive scoreline matrix and market probabilities
       f. Update dynamic DC state based on observed NPxG (cleaner target)
       g. Add match to training set for LGBM head retraining

    2. Periodically (every season) retrain the LGBM head on all matches so far

    Returns a DataFrame with one row per match, columns:
        date, league, season, home_team, away_team,
        ft_home_goals, ft_away_goals, home_npxg, away_npxg,
        pred_lam_home, pred_lam_away,
        pred_p_home, pred_p_draw, pred_p_away,
        pred_p_over_25, pred_p_btts_yes,
        lgbm_used (bool)
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    # Initialise dynamic state from Elo
    state = init_dynamic_from_elo(teams, elo_df)

    # LGBM head — start with None, train when we have enough data
    lgbm_head = LGBMHead()
    lgbm_trained = False
    training_buffer = []  # list of (features_dict, home_goals, away_goals)
    last_train_season = None

    predictions = []
    dc_lambdas_per_match = {}  # for building features

    for i, m in matches.iterrows():
        h = m["home_team"]; a = m["away_team"]
        hg = m["ft_home_goals"]; ag = m["ft_away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue
        season = m["season"]
        league = m["league"]
        date = m["date"]

        # Step 1: compute DC lambdas
        _, dc_lam_h, dc_lam_a = predict_dynamic_dc(state, h, a, DC_MAX_GOALS)
        key = (date, league, h, a)
        dc_lambdas_per_match[key] = (dc_lam_h, dc_lam_a)

        # Step 2: build features for this match
        feat = {"dc_lam_home": dc_lam_h, "dc_lam_away": dc_lam_a, "is_local_derby": 0}
        for c in LGBM_FEATURES:
            if c in feat:
                continue
            if c in matches.columns:
                feat[c] = m[c]
            else:
                feat[c] = np.nan

        # Step 3: predict final lambdas
        if lgbm_trained:
            X = pd.DataFrame([feat])[LGBM_FEATURES]
            lam_h, lam_a = lgbm_head.predict(X)
            lam_h, lam_a = float(lam_h[0]), float(lam_a[0])
        else:
            # Use DC lambdas directly (warmup)
            lam_h, lam_a = dc_lam_h, dc_lam_a

        # Step 4: derive scoreline matrix and market probabilities
        # Use state.rho (default -0.05) for the DC low-score correction.
        # Hardcoding rho=0.0 silently disables DC correction in backtests.
        P = scoreline_matrix(lam_h, lam_a, rho=state.rho, max_goals=DC_MAX_GOALS)
        n = P.shape[0]
        # Vectorized sums (much faster than nested Python loops)
        idx = np.arange(n)
        over_mask = idx[:, None] + idx[None, :] > 2.5
        btts_mask = (idx[:, None] >= 1) & (idx[None, :] >= 1)
        p_home = float(np.tril(P, -1).sum())
        p_draw = float(np.trace(P))
        p_away = float(np.triu(P, 1).sum())
        p_over_25 = float(P[over_mask].sum())
        p_btts_yes = float(P[btts_mask].sum())

        predictions.append({
            "date": date, "league": league, "season": season,
            "home_team": h, "away_team": a,
            "ft_home_goals": int(hg), "ft_away_goals": int(ag),
            "home_npxg": m.get("home_npxg"), "away_npxg": m.get("away_npxg"),
            "pred_lam_home": lam_h, "pred_lam_away": lam_a,
            "dc_lam_home": dc_lam_h, "dc_lam_away": dc_lam_a,
            "pred_p_home": p_home, "pred_p_draw": p_draw, "pred_p_away": p_away,
            "pred_p_over_25": p_over_25,
            "pred_p_btts_yes": p_btts_yes,
            "lgbm_used": lgbm_trained,
        })

        # Step 5: update dynamic DC state. Use NPxG if available, else goals.
        target_h = m.get("home_npxg")
        target_a = m.get("away_npxg")
        if pd.isna(target_h) or pd.isna(target_a):
            target_h, target_a = int(hg), int(ag)
        else:
            target_h, target_a = float(target_h), float(target_a)
        update_dynamic_dc(state, h, a, target_h, target_a)

        # Step 6: add match to training buffer
        training_buffer.append((feat, int(hg), int(ag)))

        # Step 7: at season boundary, retrain LGBM head
        if last_train_season is None:
            last_train_season = season
        elif season != last_train_season and len(training_buffer) > 500:
            print(f"  Retraining LGBM head at season {season} (buffer size: {len(training_buffer)})")
            X_train = pd.DataFrame([f for f, _, _ in training_buffer])[LGBM_FEATURES]
            y_h = pd.Series([hg for _, hg, _ in training_buffer])
            y_a = pd.Series([ag for _, _, ag in training_buffer])
            lgbm_head.fit(X_train, y_h, y_a)
            lgbm_trained = True
            last_train_season = season

        if (i + 1) % 1000 == 0:
            print(f"  ... {i + 1}/{len(matches)} matches, lgbm_trained={lgbm_trained}")

    return pd.DataFrame(predictions)
