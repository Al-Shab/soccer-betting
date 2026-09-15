"""
betting.py — Layer 5: Market-anchored betting decision.

Two-probability approach (per expert review):
- P_F = fundamental probability (from Layer 1 dynamic Dixon-Coles)
- P_M = market de-vigged consensus probability (from Layer 4)
- correction = P_F - P_M
- uncertainty = sqrt(P_F * (1 - P_F) / N_eff) + dispersion_term
- net_edge = |correction| - K * uncertainty
- Bet if net_edge > edge_threshold and correction > 0
- Stake = ¼ Kelly on net edge, capped at 2% bankroll

This is the critical change from v0.1: we bet only when the model says the
market is wrong by more than the combined uncertainty, not whenever P_F × odds
is positive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from .config import (
    BANKROLL, KELLY_FRACTION, MAX_STAKE_FRACTION,
    UNCERTAINTY_K, N_EFF_FLOOR, DISPERSION_WEIGHT
)


@dataclass
class BetSignal:
    match_id: str
    date: object
    league: str
    season: str
    home_team: str
    away_team: str
    market: str
    side: str
    p_fundamental: float
    p_market: float
    correction: float
    uncertainty: float
    net_edge: float
    odds: float
    kelly_fraction: float
    stake_fraction: float
    stake: float
    n_books: int
    dispersion: float
    place_bet: bool


def kelly_fraction(p: float, odds: float) -> float:
    """Kelly stake fraction: f = (p*o - 1) / (o - 1)."""
    if odds <= 1.0:
        return 0.0
    return (p * odds - 1.0) / (odds - 1.0)


def compute_bet_signal(
    p_fundamental: float,
    p_market: float,
    dispersion: float,
    odds: float,
    n_books: int,
    edge_threshold: float,
    bankroll: float = BANKROLL,
    n_eff: int = N_EFF_FLOOR,
) -> dict:
    """
    Compute the market-anchored bet signal for a single (match, market, side).

    Returns dict with: correction, uncertainty, net_edge, kelly_fraction,
    stake_fraction, stake, place_bet.
    """
    correction = p_fundamental - p_market

    # Uncertainty band — model uncertainty + market dispersion
    model_uncertainty = np.sqrt(p_fundamental * (1 - p_fundamental) / max(n_eff, 1))
    market_uncertainty = dispersion * DISPERSION_WEIGHT
    uncertainty = model_uncertainty + market_uncertainty

    # Net edge after uncertainty discount
    net_edge = abs(correction) - UNCERTAINTY_K * uncertainty

    # Bet only in the direction of correction > 0 (model says market underestimates)
    # AND net_edge > threshold
    place_bet = (correction > 0) and (net_edge > edge_threshold)

    # Stake: ¼ Kelly on the conservative probability (P_F minus uncertainty buffer)
    # Only bet if kelly_fraction > 0 — if not, even at the conservative prob we
    # don't have positive EV at the offered odds, so skip.
    if place_bet:
        p_conservative = max(0.01, p_fundamental - UNCERTAINTY_K * uncertainty)
        kf = kelly_fraction(p_conservative, odds)
        if kf <= 0:
            # The offered odds aren't good enough even at our model probability
            place_bet = False
            kf = 0.0
            stake_fraction = 0.0
            stake = 0.0
        else:
            stake_fraction = min(kf * KELLY_FRACTION, MAX_STAKE_FRACTION)
            # Cap from below too (no negative stakes)
            stake_fraction = max(0.0, min(stake_fraction, MAX_STAKE_FRACTION))
            stake = stake_fraction * bankroll
    else:
        kf = 0.0
        stake_fraction = 0.0
        stake = 0.0

    return {
        "correction": correction,
        "uncertainty": uncertainty,
        "net_edge": net_edge,
        "kelly_fraction_raw": kf,
        "stake_fraction": stake_fraction,
        "stake": stake,
        "place_bet": place_bet,
        "n_books": n_books,
    }


def evaluate_bet_outcome(
    signal: dict,
    actual_outcome: bool,
    odds: float,
) -> dict:
    """
    After the match, compute the outcome of a bet.
    actual_outcome: True if the bet side won, False if lost, None if push.
    """
    if not signal["place_bet"]:
        return {"placed": False, "profit": 0.0, "won": None}

    stake = signal["stake"]
    if actual_outcome is None:
        # Push — return stake
        return {"placed": True, "profit": 0.0, "won": None, "stake": stake, "odds": odds}
    if actual_outcome:
        profit = stake * (odds - 1.0)
    else:
        profit = -stake
    return {
        "placed": True,
        "stake": stake,
        "odds": odds,
        "profit": profit,
        "won": actual_outcome,
    }


# ---------------------------------------------------------------------------
# Helpers to determine actual outcome from match results
# ---------------------------------------------------------------------------

def actual_outcome_1x2(row: pd.Series, side: str) -> bool | None:
    """side in {home, draw, away}. Returns True if won, False if lost."""
    res = row.get("ft_result")
    if pd.isna(res):
        # Reconstruct from goals
        h = row.get("ft_home_goals"); a = row.get("ft_away_goals")
        if pd.isna(h) or pd.isna(a):
            return None
        if h > a: res = "H"
        elif h < a: res = "A"
        else: res = "D"
    return {"home": res == "H", "draw": res == "D", "away": res == "A"}.get(side, False)


def actual_outcome_ou25(row: pd.Series, side: str) -> bool | None:
    h = row.get("ft_home_goals"); a = row.get("ft_away_goals")
    if pd.isna(h) or pd.isna(a):
        return None
    total = int(h) + int(a)
    if side == "over":
        return total > 2.5
    elif side == "under":
        return total < 2.5
    return False


def actual_outcome_btts(row: pd.Series, side: str) -> bool | None:
    h = row.get("ft_home_goals"); a = row.get("ft_away_goals")
    if pd.isna(h) or pd.isna(a):
        return None
    both_scored = (int(h) > 0 and int(a) > 0)
    if side == "yes":
        return both_scored
    elif side == "no":
        return not both_scored
    return False


def actual_outcome_ah(row: pd.Series, side: str, handicap: float) -> bool | None:
    """
    AH outcome. side in {home, away}. handicap is the home team's handicap
    (negative = home favored; e.g. -1.5 means home starts 1.5 goals behind).
    Adjusted home score = home_goals + handicap. Home covers if (home+handicap) > away.
    For quarter lines, we approximate as half-push/half-win (returns None for push
    half — handled by backtester with split-stake accounting).
    """
    h = row.get("ft_home_goals"); a = row.get("ft_away_goals")
    if pd.isna(h) or pd.isna(a):
        return None
    margin = int(h) + handicap - int(a)  # adjusted home score - away score
    if side == "home":
        if margin > 0.01: return True
        if margin < -0.01: return False
        return None  # push
    elif side == "away":
        if margin < -0.01: return True
        if margin > 0.01: return False
        return None
    return None
