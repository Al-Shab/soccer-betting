"""
market_state.py — Layer 4: Market state.

Given a row of bookmaker odds for a match, compute:
- De-vigged consensus probability (using Shin's method)
- Cross-book dispersion (std of implied probabilities)
- Best available price per outcome
- Reference price (PSC pre-2024/25, multi-book max after)

Shin's method is empirically better than naive normalization because it
accounts for insider trading — the bookmaker spreads the margin more heavily
on the more probable outcome, not equally.

Reference: Shin (1993), "Developing an Operational Model for the Explanation
of Betting Behaviour and the Calculation of Bookmaker Margins".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable
from .config import PSC_TRUST_THROUGH_SEASON, REFERENCE_BOOKS


# ---------------------------------------------------------------------------
# Bookmaker column mappings for each market
# ---------------------------------------------------------------------------

# For 1X2: (home_col, draw_col, away_col) per bookmaker
BOOK_1X2_COLS = {
    "PSC":  ("psc_home", "psc_draw", "psc_away"),
    "BbMx": ("bbmx_home", "bbmx_draw", "bbmx_away"),
    "BbAv": ("bbav_home", "bbav_draw", "bbav_away"),
    "B365": ("b365_home", "b365_draw", "b365_away"),
    "BF":   ("bf_home", "bf_draw", "bf_away"),
    "WH":   ("wh_home", "wh_draw", "wh_away"),
    "VC":   ("vc_home", "vc_draw", "vc_away"),
    "LB":   ("lb_home", "lb_draw", "lb_away"),
    "SBO":  ("sbo_home", "sbo_draw", "sbo_away"),
    "IW":   ("iw_home", "iw_draw", "iw_away"),
}

# For O/U 2.5: (over_col, under_col) per bookmaker
BOOK_OU25_COLS = {
    "BbMx": ("bbmx_over_25", "bbmx_under_25"),
    "BbAv": ("bbav_over_25", "bbav_under_25"),
    "B365": ("b365_over_25", "b365_under_25"),
    "Max":  ("max_over_25", "max_under_25"),
    "Avg":  ("avg_over_25", "avg_under_25"),
}

# For Asian handicap: (home_col, away_col, handicap_col) per bookmaker
BOOK_AH_COLS = {
    "BbMx": ("bbmx_ah_home", "bbmx_ah_away", "bbah_handicap"),
    "BbAv": ("bbav_ah_home", "bbav_ah_away", "bbah_handicap"),
    "B365": ("b365_ah_home", "b365_ah_away", "b365_ah_handicap"),
    "Max":  ("max_ah_home", "max_ah_away"),
    "Avg":  ("avg_ah_home", "avg_ah_away"),
}

# For BTTS — football-data.co.uk covers this inconsistently
BOOK_BTTS_COLS = {
    "BbMx": ("bbmx_btts_yes", "bbmx_btts_no"),
    "BbAv": ("bbav_btts_yes", "bbav_btts_no"),
}


# ---------------------------------------------------------------------------
# De-vig using Shin's method
# ---------------------------------------------------------------------------

def shin_devig(odds: Iterable[float]) -> tuple[list[float], float]:
    """
    Apply Shin's method to remove the bookmaker margin from a set of odds.

    Vectorized with numpy for speed. For small arrays (2-4 outcomes) we use
    a fast path with pure Python (no numpy overhead), since numpy's per-call
    overhead is ~50μs which dominates for small inputs.

    Returns a list of de-vigged probabilities, one per input odd (in the same
    order). If an input is invalid, the corresponding output is 0 (and the
    bookmaker column set should not include it).
    """
    odds_list = list(odds)
    n_outcomes = len(odds_list)
    if n_outcomes == 0:
        return [], 0.0

    # Fast validation (no numpy for small arrays)
    valid_odds = []
    valid_mask = []
    for o in odds_list:
        is_valid = (o is not None) and (o == o) and (o > 1.0)
        valid_mask.append(is_valid)
        if is_valid:
            valid_odds.append(float(o))
    n_valid = len(valid_odds)

    if n_valid < 2:
        out = [0.0] * n_outcomes
        if n_valid == 1:
            s = 1.0 / valid_odds[0]
            for i, v in enumerate(valid_mask):
                if v:
                    out[i] = s
        return out, 0.0

    # For small arrays (2-4 outcomes), use pure-Python Shin — no numpy overhead
    # This is the hot path; numpy overhead is ~50μs per call which dominates
    # NOTE: do NOT clamp intermediate p_i values to zero — that breaks the Shin
    # iteration (it can oscillate between two states, and clamping forces it
    # into the wrong fixed point). The Shin formula naturally produces slightly
    # negative intermediate values that converge to sensible positive probs.
    if n_valid <= 4:
        raw_implied = [1.0 / o for o in valid_odds]
        s = sum(raw_implied)
        margin = s - 1.0
        if margin < 1e-4:
            valid_probs = [r / s for r in raw_implied]
        else:
            z = 0.0
            p = [r / s for r in raw_implied]
            for _ in range(50):
                sum_p1mp = sum(pi * (1.0 - pi) for pi in p)
                sum_1mp_sq = sum((1.0 - pi) ** 2 for pi in p)
                if sum_1mp_sq < 1e-10:
                    break
                z_new = sum_p1mp / sum_1mp_sq
                if abs(z_new - z) < 1e-6:
                    z = z_new
                    break
                z = z_new
                if (1.0 - z) <= 1e-10:
                    break
                # Compute new p — DO NOT clamp to zero (breaks iteration)
                p = [(r - z) / (1.0 - z) for r in raw_implied]
                ps = sum(p)
                if ps > 0:
                    p = [pi / ps for pi in p]
            # Final safety: clip any tiny negatives to zero for output
            valid_probs = [max(0.0, pi) for pi in p]
            # Renormalize after clipping
            ps = sum(valid_probs)
            if ps > 0:
                valid_probs = [pi / ps for pi in valid_probs]

        # Reconstruct output in original order
        out = []
        vi = 0
        for v in valid_mask:
            if v:
                out.append(valid_probs[vi])
                vi += 1
            else:
                out.append(0.0)
        return out, 0.0

    # Large arrays (5+ outcomes) — use numpy vectorization
    arr = np.array(odds_list, dtype=float)
    valid_mask_np = ~np.isnan(arr) & (arr > 1.0)
    valid_odds_np = arr[valid_mask_np]
    out = np.zeros(n_outcomes, dtype=float)

    raw_implied = 1.0 / valid_odds_np
    s = raw_implied.sum()
    margin = s - 1.0
    if margin < 1e-4:
        out[valid_mask_np] = raw_implied / s
        return out.tolist(), 0.0

    p = raw_implied / s
    z = 0.0
    for _ in range(50):
        z_new = (p * (1.0 - p)).sum() / max(((1.0 - p) ** 2).sum(), 1e-10)
        if abs(z_new - z) < 1e-6:
            z = z_new
            break
        z = z_new
        if (1.0 - z) <= 1e-10:
            break
        # NO clip — let iteration run naturally
        p = (raw_implied - z) / (1.0 - z)
        ps = p.sum()
        if ps > 0:
            p = p / ps
    # Final safety: clip tiny negatives to zero
    p = np.maximum(p, 0.0)
    ps = p.sum()
    if ps > 0:
        p = p / ps
    out[valid_mask_np] = p
    return out.tolist(), 0.0


# ---------------------------------------------------------------------------
# Per-market de-vigged consensus + dispersion
# ---------------------------------------------------------------------------

def _collect_book_odds(row: pd.Series, cols_dict: dict, book_key: str) -> list[float]:
    """Get odds for one bookmaker from a match row."""
    cols = cols_dict[book_key]
    return [row.get(c) for c in cols if c in row.index]


def build_market_state_for_1x2(row: pd.Series, season: str) -> dict:
    """
    For a 1X2 market row, compute:
    - devigged probabilities per book
    - consensus (mean across books)
    - dispersion (std across books)
    - best price per outcome
    - reference (sharp) price
    """
    books_with_odds = {}
    for book, cols in BOOK_1X2_COLS.items():
        # Skip PSC for 2024/25+ seasons (staleness — see config)
        if book == "PSC" and str(season) > str(PSC_TRUST_THROUGH_SEASON):
            continue
        odds = [row.get(c) for c in cols if c in row.index]
        if len(odds) >= 2 and all(o is not None and not pd.isna(o) and o > 1.0 for o in odds):
            books_with_odds[book] = odds

    if not books_with_odds:
        return {"available": False}

    # De-vig per book
    per_book_devig = {}
    for book, ods in books_with_odds.items():
        probs, _ = shin_devig(ods)
        per_book_devig[book] = probs  # [home, draw, away]

    # Consensus — mean of de-vigged probs
    consensus = [np.mean([per_book_devig[b][i] for b in per_book_devig])
                 for i in range(3)]
    # Dispersion — std across books
    dispersion = [np.std([per_book_devig[b][i] for b in per_book_devig])
                  for i in range(3)]

    # Best price (max) per outcome
    best_price = [max(books_with_odds[b][i] for b in books_with_odds)
                  for i in range(3)]

    # Reference — first available from REFERENCE_BOOKS
    reference = None
    for b in REFERENCE_BOOKS:
        if b in books_with_odds:
            reference = {"book": b, "odds": books_with_odds[b]}
            break

    return {
        "available": True,
        "consensus": {"home": consensus[0], "draw": consensus[1], "away": consensus[2]},
        "dispersion": {"home": dispersion[0], "draw": dispersion[1], "away": dispersion[2]},
        "best_price": {"home": best_price[0], "draw": best_price[1], "away": best_price[2]},
        "reference_book": reference,
        "per_book_devig": per_book_devig,
        "n_books": len(books_with_odds),
    }


def build_market_state_for_ou25(row: pd.Series, season: str) -> dict:
    books_with_odds = {}
    for book, cols in BOOK_OU25_COLS.items():
        if book == "PSC" and str(season) > str(PSC_TRUST_THROUGH_SEASON):
            continue
        odds = [row.get(c) for c in cols if c in row.index]
        if len(odds) >= 2 and all(o is not None and not pd.isna(o) and o > 1.0 for o in odds):
            books_with_odds[book] = odds
    if not books_with_odds:
        return {"available": False}

    per_book_devig = {b: shin_devig(o)[0] for b, o in books_with_odds.items()}
    consensus = [np.mean([per_book_devig[b][i] for b in per_book_devig])
                for i in range(2)]
    dispersion = [np.std([per_book_devig[b][i] for b in per_book_devig])
                  for i in range(2)]
    best_price = [max(books_with_odds[b][i] for b in books_with_odds)
                  for i in range(2)]
    return {
        "available": True,
        "consensus": {"over": consensus[0], "under": consensus[1]},
        "dispersion": {"over": dispersion[0], "under": dispersion[1]},
        "best_price": {"over": best_price[0], "under": best_price[1]},
        "per_book_devig": per_book_devig,
        "n_books": len(books_with_odds),
    }


def build_market_state_for_ah(row: pd.Series, season: str) -> dict:
    """
    Asian handicap market state. Note: handicap line varies per match, so we
    extract the line from the row and use the consensus.
    """
    books_with_odds = {}
    handicap = row.get("bbah_handicap")
    if pd.isna(handicap):
        return {"available": False}
    for book, cols in BOOK_AH_COLS.items():
        if book == "PSC" and str(season) > str(PSC_TRUST_THROUGH_SEASON):
            continue
        odds = [row.get(c) for c in cols[:2] if c in row.index]
        if len(odds) >= 2 and all(o is not None and not pd.isna(o) and o > 1.0 for o in odds):
            books_with_odds[book] = odds
    if not books_with_odds:
        return {"available": False}

    per_book_devig = {b: shin_devig(o)[0] for b, o in books_with_odds.items()}
    consensus = [np.mean([per_book_devig[b][i] for b in per_book_devig])
                 for i in range(2)]
    dispersion = [np.std([per_book_devig[b][i] for b in per_book_devig])
                  for i in range(2)]
    best_price = [max(books_with_odds[b][i] for b in books_with_odds)
                  for i in range(2)]
    return {
        "available": True,
        "handicap": float(handicap),
        "consensus": {"home": consensus[0], "away": consensus[1]},
        "dispersion": {"home": dispersion[0], "away": dispersion[1]},
        "best_price": {"home": best_price[0], "away": best_price[1]},
        "per_book_devig": per_book_devig,
        "n_books": len(books_with_odds),
    }


def build_market_state_for_btts(row: pd.Series, season: str) -> dict:
    books_with_odds = {}
    for book, cols in BOOK_BTTS_COLS.items():
        if book == "PSC" and str(season) > str(PSC_TRUST_THROUGH_SEASON):
            continue
        odds = [row.get(c) for c in cols if c in row.index]
        if len(odds) >= 2 and all(o is not None and not pd.isna(o) and o > 1.0 for o in odds):
            books_with_odds[book] = odds
    if not books_with_odds:
        return {"available": False}

    per_book_devig = {b: shin_devig(o)[0] for b, o in books_with_odds.items()}
    consensus = [np.mean([per_book_devig[b][i] for b in per_book_devig])
                for i in range(2)]
    dispersion = [np.std([per_book_devig[b][i] for b in per_book_devig])
                  for i in range(2)]
    best_price = [max(books_with_odds[b][i] for b in books_with_odds)
                  for i in range(2)]
    return {
        "available": True,
        "consensus": {"yes": consensus[0], "no": consensus[1]},
        "dispersion": {"yes": dispersion[0], "no": dispersion[1]},
        "best_price": {"yes": best_price[0], "no": best_price[1]},
        "per_book_devig": per_book_devig,
        "n_books": len(books_with_odds),
    }
