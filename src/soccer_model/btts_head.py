"""
btts_head.py — Phase 0 v4: Dedicated BTTS prediction head.

The main model uses independent Poisson with extended DC correction for the
scoreline matrix, but BTTS specifically depends on the JOINT distribution of
goals (when do both teams score?). The independent-Poisson assumption
structurally struggles here because:
1. Game-state effects: 0-0 is positively correlated with both teams being
   defensive (small λ_h × λ_a product isn't enough)
2. Style matchups: two attacking teams (high PPDA = less pressing) → more
   open games → more BTTS Yes
3. Tempo: matches with high combined xG but moderate defensive depth → more
   BTTS Yes

This dedicated head uses style matchup features (PPDA, deep, xpts differential)
plus the rolling BTTS rate to predict BTTS Yes/No directly with a LightGBM
binary classifier.

Target: y = 1 if (ft_home_goals > 0) and (ft_away_goals > 0), else 0

Features (per match):
- Both teams' rolling BTTS Yes rate (last 10 matches exp-decay)
- Both teams' rolling xpts (open attacking vs defensive style)
- PPDA differential (both teams' pressing intensity — defensive teams = high PPDA)
- Deep for and deep against for both teams
- Combined expected goals (dc_lam_home + dc_lam_away)
- Combined NPxG (rolling)
- PPDA ratio (one team's PPDA / other team's PPDA — captures style asymmetry)
- League encoded

The output is a calibrated P(BTTS yes) that we use directly (instead of deriving
from the scoreline matrix).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from typing import Optional
from collections import defaultdict


BTTS_FEATURES = [
    # Main model signals
    "dc_lam_home", "dc_lam_away",
    # Per-team rolling BTTS rate (need to add — computed in features.py)
    "rolling_home_btts_for", "rolling_home_btts_against",
    "rolling_away_btts_for", "rolling_away_btts_against",
    # Style matchup: PPDA (higher = less pressing, more open)
    "rolling_home_ppda_for", "rolling_home_oppda_against",
    "rolling_away_ppda_for", "rolling_away_oppda_against",
    # Touches in box (more = more attacking)
    "rolling_home_deep_for", "rolling_home_deep_against",
    "rolling_away_deep_for", "rolling_away_deep_against",
    # Expected points (attacking vs defensive style)
    "rolling_home_xpts_for", "rolling_home_xpts_against",
    "rolling_away_xpts_for", "rolling_away_xpts_against",
    # Combined xG (open games)
    "rolling_home_npxg_for", "rolling_home_npxg_against",
    "rolling_away_npxg_for", "rolling_away_npxg_against",
    # History (cold-start awareness)
    "home_matches_history", "away_matches_history",
]


class BTTSHead:
    """LightGBM binary classifier for BTTS prediction, with isotonic calibration."""

    def __init__(self):
        self.model = None
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_fit = False
        self.calibration_buffer_X = []
        self.calibration_buffer_y = []
        self.last_cal_count = 0

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 50,
            "verbose": -1,
        }
        self.params = params
        self.num_boost_round = 100

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        mask = y.notna()
        X_clean = X[mask].copy()
        y_clean = y[mask].astype(int)
        X_clean = X_clean.fillna(X_clean.median(numeric_only=True)).fillna(0)
        dtrain = lgb.Dataset(X_clean, label=y_clean, feature_name=BTTS_FEATURES)
        self.model = lgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)

    def update_calibrator(self, raw_prob: float, outcome: int) -> None:
        """Add an observation to the calibration buffer and refit if needed."""
        self.calibration_buffer_X.append(float(raw_prob))
        self.calibration_buffer_y.append(int(outcome))
        if len(self.calibration_buffer_X) - self.last_cal_count >= 100 and len(self.calibration_buffer_X) >= 300:
            X = np.array(self.calibration_buffer_X)
            y = np.array(self.calibration_buffer_y)
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(X, y)
            self.calibrator_fit = True
            self.last_cal_count = len(self.calibration_buffer_X)

    def predict_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            # Return uninformative prior
            return np.full(len(X), 0.5)
        X_filled = X.fillna(X.median(numeric_only=True)).fillna(0)
        raw = self.model.predict(X_filled)
        if self.calibrator_fit:
            return self.calibrator.predict(raw)
        return raw
