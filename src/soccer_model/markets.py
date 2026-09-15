"""
markets.py — Layer 1+4 bridge: scoreline matrix → market odds.

Given the scoreline probability matrix P[i,j], derive fair odds for:
- 1X2 (home/draw/away)
- Asian handicap (any line, including quarter-line splitting)
- Totals (any line, including quarter-line splitting)
- BTTS (both teams to score)
- Correct score (direct read)
- HT/FT (via historical ratio adjustment — Phase 0 approximation)

All functions take the scoreline matrix and return:
- probabilities (list of dicts)
- fair_odds = 1/probability (decimal)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from .config import DC_MAX_GOALS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slice_with_tail(P: np.ndarray, max_goals: int) -> np.ndarray:
    """Truncate/keep matrix; tail row/col absorbs > max_goals mass."""
    return P[: max_goals + 1, : max_goals + 1]


# ---------------------------------------------------------------------------
# 1X2
# ---------------------------------------------------------------------------

def derive_1x2(P: np.ndarray) -> dict[str, float]:
    """Return P(home), P(draw), P(away) from scoreline matrix."""
    n = P.shape[0]
    p_home = sum(P[i, j] for i in range(n) for j in range(n) if i > j)
    p_draw = sum(P[i, i] for i in range(n))
    p_away = sum(P[i, j] for i in range(n) for j in range(n) if i < j)
    return {
        "home": p_home,
        "draw": p_draw,
        "away": p_away,
    }


# ---------------------------------------------------------------------------
# Totals (over/under)
# ---------------------------------------------------------------------------

def derive_totals(P: np.ndarray, line: float) -> dict[str, float]:
    """
    Compute P(over) and P(under) for a totals line.
    Supports half-lines (e.g. 2.5 — no push), integer lines (push possible),
    and quarter-lines (e.g. 2.25 — split into 2.0 and 2.5 half-bets).

    For quarter lines, return the "blended" probability — for the bet decision
    layer, the backtester handles split-stake accounting.
    """
    n = P.shape[0]
    if line % 1 == 0.5:
        # Half line — no push possible
        p_over = sum(P[i, j] for i in range(n) for j in range(n) if i + j > line)
        p_under = 1.0 - p_over
        return {"over": p_over, "under": p_under, "push": 0.0}
    elif line % 1 == 0.0:
        # Integer line — push possible
        p_over = sum(P[i, j] for i in range(n) for j in range(n) if i + j > line)
        p_push = sum(P[i, j] for i in range(n) for j in range(n) if i + j == line)
        p_under = 1.0 - p_over - p_push
        return {"over": p_over, "under": p_under, "push": p_push}
    else:
        # Quarter line (e.g. 2.25, 2.75) — split into two adjacent half-lines
        lower = float(np.floor(line * 2) / 2)
        upper = lower + 0.5
        lower_p = derive_totals(P, lower)
        upper_p = derive_totals(P, upper)
        return {
            "over": 0.5 * (lower_p["over"] + upper_p["over"]),
            "under": 0.5 * (lower_p["under"] + upper_p["under"]),
            "push": 0.0,
        }


# ---------------------------------------------------------------------------
# BTTS (both teams to score)
# ---------------------------------------------------------------------------

def derive_btts(P: np.ndarray) -> dict[str, float]:
    """P(BTTS yes) and P(BTTS no) from scoreline matrix."""
    n = P.shape[0]
    # P(both teams scored) = 1 - P(home=0) - P(away=0) + P(home=0, away=0)
    p_home_zero = sum(P[0, j] for j in range(n))
    p_away_zero = sum(P[i, 0] for i in range(n))
    p_both_zero = P[0, 0]
    p_yes = 1.0 - p_home_zero - p_away_zero + p_both_zero
    return {"yes": p_yes, "no": 1.0 - p_yes}


# ---------------------------------------------------------------------------
# Asian handicap
# ---------------------------------------------------------------------------

def derive_asian_handicap(P: np.ndarray, line: float) -> dict[str, float]:
    """
    Compute P(home win | AH line), P(away win | AH line), P(push).
    Convention: line is the home team's handicap. e.g. line=-1.5 means home
    must win by 2+ to cover.

    For half/quarter lines, split accordingly.
    """
    n = P.shape[0]

    def p_outcome(line_arg: float) -> tuple[float, float, float]:
        """P(home_win | line_arg), P(away_win | line_arg), P(push | line_arg).
        Convention: `line_arg` is the home team's handicap (e.g. -1.5 means home
        starts 1.5 goals behind). Adjusted home score = i + line_arg. Home covers
        if (i + line_arg) > j, i.e., i - j + line_arg > 0.

        NOTE: uses the parameter `line_arg`, NOT the outer `line` closure
        variable — this is critical for quarter-line split-stake calculation
        where the two split half-bets use different `lower` and `upper` values.
        """
        p_home = 0.0; p_away = 0.0; p_push = 0.0
        for i in range(n):
            for j in range(n):
                margin = i + line_arg - j  # = i - j + line_arg
                if margin > 0.001:
                    p_home += P[i, j]
                elif margin < -0.001:
                    p_away += P[i, j]
                else:
                    p_push += P[i, j]
        return p_home, p_away, p_push

    if line * 2 == int(line * 2):
        # Integer or half line
        p_home, p_away, p_push = p_outcome(line)
        return {"home": p_home, "away": p_away, "push": p_push}
    else:
        # Quarter line — split into two adjacent half-lines
        # Each half-bet uses one of the two adjacent half-lines.
        # E.g. line=-1.25 → lower=-1.5, upper=-1.0
        # Final outcome is the AVERAGE of the two half-bet outcomes.
        # Push is the average too (lower half-bet may have push, upper usually doesn't).
        lower = float(np.floor(line * 2) / 2)
        upper = lower + 0.5
        l_home, l_away, l_push = p_outcome(lower)
        u_home, u_away, u_push = p_outcome(upper)
        return {
            "home": 0.5 * (l_home + u_home),
            "away": 0.5 * (l_away + u_away),
            "push": 0.5 * (l_push + u_push),
        }


# ---------------------------------------------------------------------------
# Correct score
# ---------------------------------------------------------------------------

def derive_correct_score(P: np.ndarray, home_goals: int, away_goals: int) -> float:
    """P(home_goals, away_goals) directly from the matrix."""
    if home_goals >= P.shape[0] or away_goals >= P.shape[1]:
        return 0.0
    return P[home_goals, away_goals]


# ---------------------------------------------------------------------------
# HT/FT (Phase 0 approximation: historical ratio per FT scoreline)
# ---------------------------------------------------------------------------

def fit_htft_ratios(matches: pd.DataFrame) -> dict:
    """
    From historical matches with HT/FT scores, build a lookup:
    P(HT=i, FT=j | FT=k) for each FT scoreline k.

    Returns a dict keyed by (ft_home, ft_away) → dict of (ht_home, ht_away) → prob.
    """
    valid = matches[
        matches["ht_home_goals"].notna() &
        matches["ht_away_goals"].notna() &
        matches["ft_home_goals"].notna() &
        matches["ft_away_goals"].notna()
    ].copy()
    if valid.empty:
        return {}
    valid["ht_home_goals"] = valid["ht_home_goals"].astype(int)
    valid["ht_away_goals"] = valid["ht_away_goals"].astype(int)
    valid["ft_home_goals"] = valid["ft_home_goals"].astype(int)
    valid["ft_away_goals"] = valid["ft_away_goals"].astype(int)
    ratios = {}
    grouped = valid.groupby(["ft_home_goals", "ft_away_goals"])
    for (fh, fa), grp in grouped:
        sub = grp.groupby(["ht_home_goals", "ht_away_goals"]).size()
        total = sub.sum()
        ratios[(fh, fa)] = {(h, a): c / total for (h, a), c in sub.items()}
    return ratios


def derive_htft(P: np.ndarray, htft_ratios: dict) -> dict:
    """
    Approximate HT/FT probabilities by applying historical ratio per FT scoreline.
    Returns a dict of (ht_result, ft_result) → probability.
    ht_result and ft_result in {H, D, A} (home/draw/away).
    """
    n = P.shape[0]
    probs = {}
    for ft_h in range(n):
        for ft_a in range(n):
            p_ft = P[ft_h, ft_a]
            if p_ft < 1e-6:
                continue
            ratios = htft_ratios.get((ft_h, ft_a), {})
            if not ratios:
                # Fallback: equal split — naive, just to avoid zero
                ratios = {(0, 0): 1.0}
            for (ht_h, ht_a), r in ratios.items():
                # HT result
                if ht_h > ht_a:
                    ht_res = "H"
                elif ht_h == ht_a:
                    ht_res = "D"
                else:
                    ht_res = "A"
                # FT result
                if ft_h > ft_a:
                    ft_res = "H"
                elif ft_h == ft_a:
                    ft_res = "D"
                else:
                    ft_res = "A"
                key = f"{ht_res}/{ft_res}"
                probs[key] = probs.get(key, 0.0) + p_ft * r
    return probs


# ---------------------------------------------------------------------------
# Convenience: all markets from one scoreline matrix
# ---------------------------------------------------------------------------

def derive_all_markets(P: np.ndarray,
                        handicap_lines: list = [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5],
                        totals_lines: list = [0.5, 1.5, 2.5, 3.5]) -> dict:
    """Derive all standard markets from a scoreline matrix."""
    markets = {
        "1X2": derive_1x2(P),
        "BTTS": derive_btts(P),
    }
    for line in totals_lines:
        markets[f"O/U_{line}"] = derive_totals(P, line)
    for line in handicap_lines:
        markets[f"AH_{line}"] = derive_asian_handicap(P, line)
    return markets
