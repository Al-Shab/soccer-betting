"""
37_prepare_live_model.py — Train the LGBM model on ALL historical data
and save: model + DC state + latest rolling features per team + calibrator.

This produces the artifacts needed for live predictions:
  models/lgbm_home.pkl, models/lgbm_away.pkl  — trained LightGBM models
  models/dc_state.pkl                          — final DynamicDCState
  models/team_features.pkl                     — latest rolling features per team
  models/calibrator.pkl                        — isotonic calibrators per market

The live Streamlit app loads these to make predictions on upcoming matches
without needing to re-run the full backtest.
"""
import os
import sys
import pickle
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from soccer_model.config import (
    DATA_PROCESSED, ELO_DIR, MODELS_DIR, DC_MAX_GOALS, BANKROLL,
)
from soccer_model.dixon_coles import (
    DynamicDCState, init_dynamic_from_elo,
    predict_dynamic_dc, update_dynamic_dc,
    scoreline_matrix,  # Use the canonical scoreline_matrix (handles tail overflow)
)
from soccer_model.lgbm_head import LGBMHead, LGBM_FEATURES
from soccer_model.markets import derive_1x2, derive_totals, derive_btts
from soccer_model.metrics import brier_score

import lightgbm as lgb


def main():
    print("=" * 70)
    print("Preparing live model artifacts")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    matches = pd.read_csv(DATA_PROCESSED / "matches_with_features.csv", low_memory=False)
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    elo = pd.read_csv(ELO_DIR / "ALL_elo.csv")
    elo["date"] = pd.to_datetime(elo["date"])

    matches = matches.sort_values("date").reset_index(drop=True)
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    print(f"Loaded {len(matches):,} matches, {len(teams)} teams")

    # Initialize DC state from Elo
    state = init_dynamic_from_elo(teams, elo)

    # Walk forward through ALL matches, updating DC state
    # Also collect training data for LGBM and calibrator
    print("\nWalking forward through all matches...")
    training_buffer = []
    calibrator_history = defaultdict(list)

    for i, m in matches.iterrows():
        h = m["home_team"]; a = m["away_team"]
        hg = m["ft_home_goals"]; ag = m["ft_away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue

        _, dc_lam_h, dc_lam_a = predict_dynamic_dc(state, h, a, DC_MAX_GOALS)

        # Build features
        feat = {"dc_lam_home": dc_lam_h, "dc_lam_away": dc_lam_a, "is_local_derby": 0}
        for c in LGBM_FEATURES:
            if c in feat:
                continue
            if c in matches.columns:
                feat[c] = m[c]
            else:
                feat[c] = np.nan

        # Derive raw market probabilities for calibrator
        # Use the canonical scoreline_matrix from dixon_coles.py.
        # The old hand-rolled version had two bugs:
        # 1. Missing tail overflow at i==max_goals (poisson_pmf vs survivor).
        # 2. Used BASIC DC tau (4-cell) only — scoreline_matrix defaults to that
        #    with apply_draw_buffer=False, which is what we want.
        P = scoreline_matrix(dc_lam_h, dc_lam_a, rho=state.rho, max_goals=DC_MAX_GOALS)

        n = P.shape[0]
        # Vectorized sums (much faster than nested Python loops)
        idx = np.arange(n)
        over_mask = idx[:, None] + idx[None, :] > 2.5
        btts_mask = (idx[:, None] >= 1) & (idx[None, :] >= 1)
        p_home_raw = float(np.tril(P, -1).sum())
        p_draw_raw = float(np.trace(P))
        p_away_raw = float(np.triu(P, 1).sum())
        p_over_raw = float(P[over_mask].sum())
        p_btts_raw = float(P[btts_mask].sum())

        # Record for calibrator
        home_win = int(hg > ag)
        draw = int(hg == ag)
        away_win = int(hg < ag)
        over = int(hg + ag > 2.5)
        btts = int(hg > 0 and ag > 0)

        calibrator_history["home_win"].append((float(p_home_raw), home_win))
        calibrator_history["draw"].append((float(p_draw_raw), draw))
        calibrator_history["away_win"].append((float(p_away_raw), away_win))
        calibrator_history["over_25"].append((float(p_over_raw), over))
        calibrator_history["btts_yes"].append((float(p_btts_raw), btts))

        # Add to LGBM training buffer
        training_buffer.append((feat, int(hg), int(ag)))

        # Update DC state on NPxG
        target_h = m.get("home_npxg")
        target_a = m.get("away_npxg")
        if pd.isna(target_h) or pd.isna(target_a):
            target_h, target_a = int(hg), int(ag)
        else:
            target_h, target_a = float(target_h), float(target_a)
        update_dynamic_dc(state, h, a, target_h, target_a)

        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{len(matches)}")

    print(f"\nWalk-forward complete. DC state has {state.matches_seen} updates.")
    print(f"Training buffer: {len(training_buffer)} matches")

    # Train LGBM model on ALL data
    print("\nTraining LGBM model on all data...")
    X_train = pd.DataFrame([f for f, _, _ in training_buffer])[LGBM_FEATURES]
    y_h = pd.Series([hg for _, hg, _ in training_buffer])
    y_a = pd.Series([ag for _, _, ag in training_buffer])

    params = {
        "objective": "poisson", "metric": "poisson",
        "num_leaves": 31, "learning_rate": 0.05,
        "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 5, "min_data_in_leaf": 30, "verbose": -1,
    }

    X_clean = X_train.fillna(X_train.median(numeric_only=True)).fillna(0)
    dtrain_h = lgb.Dataset(X_clean, label=y_h, feature_name=LGBM_FEATURES)
    dtrain_a = lgb.Dataset(X_clean, label=y_a, feature_name=LGBM_FEATURES)
    model_home = lgb.train(params, dtrain_h, num_boost_round=100)
    model_away = lgb.train(params, dtrain_a, num_boost_round=100)

    # Save LGBM models
    model_home.save_model(str(MODELS_DIR / "lgbm_home.txt"))
    model_away.save_model(str(MODELS_DIR / "lgbm_away.txt"))
    print(f"Saved LGBM models to {MODELS_DIR}")

    # Save DC state
    dc_state = {
        "attack": state.attack,
        "defence": state.defence,
        "home_adv": state.home_adv,
        "rho": state.rho,
        "matches_seen": state.matches_seen,
    }
    with open(MODELS_DIR / "dc_state.pkl", "wb") as f:
        pickle.dump(dc_state, f)
    print(f"Saved DC state (matches_seen={state.matches_seen})")

    # Train and save calibrators
    calibrators = {}
    for market, history in calibrator_history.items():
        if len(history) < 200:
            continue
        X_cal = np.array([p for p, _ in history])
        y_cal = np.array([o for _, o in history])
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(X_cal, y_cal)
        calibrators[market] = iso
        print(f"  Calibrator '{market}': {len(history)} observations")

    with open(MODELS_DIR / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)
    print(f"Saved calibrators")

    # Compute and save latest rolling features per team
    print("\nComputing latest rolling features per team...")
    from soccer_model.features import per_team_long_format, compute_team_rolling_features

    long_df = per_team_long_format(matches)
    long_with_features = compute_team_rolling_features(long_df)

    # Get the latest features per team
    latest_features = {}
    for team, grp in long_with_features.groupby("team"):
        grp = grp.sort_values("date")
        latest = grp.iloc[-1]
        rolling_cols = [c for c in grp.columns if c.startswith("rolling_")]
        feat_dict = {c: float(latest[c]) if pd.notna(latest[c]) else 0.0 for c in rolling_cols}
        feat_dict["matches_history"] = int(latest.get("matches_history", 0))
        feat_dict["last_match_date"] = str(latest["date"].date())
        latest_features[team] = feat_dict

    with open(MODELS_DIR / "team_features.pkl", "wb") as f:
        pickle.dump(latest_features, f)
    print(f"Saved latest features for {len(latest_features)} teams")

    # Also save the LGBM feature list
    with open(MODELS_DIR / "lgbm_features.pkl", "wb") as f:
        pickle.dump(LGBM_FEATURES, f)

    # Quick validation: predict on last 100 matches and check Brier
    print("\nQuick validation on last 100 matches...")
    test = matches.tail(100)
    correct = 0
    for _, m in test.iterrows():
        h = m["home_team"]; a = m["away_team"]
        hg = m["ft_home_goals"]; ag = m["ft_away_goals"]
        if pd.isna(hg) or pd.isna(ag):
            continue

        # Get DC lambdas
        a_h = state.attack.get(h, 0.0)
        a_a = state.attack.get(a, 0.0)
        d_h = state.defence.get(h, 0.0)
        d_a = state.defence.get(a, 0.0)
        lam_h = np.exp(a_h + d_a + state.home_adv)
        lam_a = np.exp(a_a + d_h)

        # Build features
        feat = {"dc_lam_home": lam_h, "dc_lam_away": lam_a, "is_local_derby": 0}
        for c in LGBM_FEATURES:
            if c in feat: continue
            feat[c] = m.get(c, np.nan)

        X = pd.DataFrame([feat])[LGBM_FEATURES]
        X_filled = X.fillna(X.median(numeric_only=True)).fillna(0)
        pred_h = float(model_home.predict(X_filled)[0])
        pred_a = float(model_away.predict(X_filled)[0])

        pred_home_win = 1 if pred_h > pred_a else 0
        actual_home_win = 1 if hg > ag else 0
        if pred_home_win == actual_home_win:
            correct += 1

    print(f"  Directional accuracy (home vs away): {correct}/{len(test)} = {correct/len(test)*100:.1f}%")

    print("\n" + "=" * 70)
    print("LIVE MODEL ARTIFACTS SAVED")
    print(f"  LGBM models: {MODELS_DIR / 'lgbm_home.txt'}, {MODELS_DIR / 'lgbm_away.txt'}")
    print(f"  DC state: {MODELS_DIR / 'dc_state.pkl'}")
    print(f"  Calibrators: {MODELS_DIR / 'calibrators.pkl'}")
    print(f"  Team features: {MODELS_DIR / 'team_features.pkl'}")
    print(f"  Feature list: {MODELS_DIR / 'lgbm_features.pkl'}")


if __name__ == "__main__":
    main()
