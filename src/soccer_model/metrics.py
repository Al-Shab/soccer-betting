"""
metrics.py — Calibration-first metrics per expert review.

Primary: Brier score, log-loss, reliability, calibration slope/intercept.
Secondary: yield, ROI, CLV with CIs, hit rate, drawdown, Sharpe-like ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable


# ---------------------------------------------------------------------------
# Primary metrics — calibration
# ---------------------------------------------------------------------------

def brier_score(predictions: pd.Series, outcomes: pd.Series) -> float:
    """Mean squared error of binary probabilities."""
    return float(np.mean((predictions.values - outcomes.values.astype(float)) ** 2))


def log_loss(predictions: pd.Series, outcomes: pd.Series, eps: float = 1e-10) -> float:
    """Negative log-likelihood, capped for stability."""
    p = np.clip(predictions.values, eps, 1 - eps)
    y = outcomes.values.astype(float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_curve(predictions: pd.Series, outcomes: pd.Series,
                       n_bins: int = 10) -> pd.DataFrame:
    """
    Bucket predictions by probability, return per-bin predicted vs actual hit rate.
    Output: DataFrame with columns: bin_low, bin_high, n, predicted, actual.
    """
    df = pd.DataFrame({"pred": predictions.values, "out": outcomes.values.astype(float)})
    bins = np.linspace(0, 1, n_bins + 1)
    df["bin_idx"] = pd.cut(df["pred"], bins=bins, labels=False, include_lowest=True)
    rows = []
    for i in range(n_bins):
        sub = df[df["bin_idx"] == i]
        if sub.empty:
            continue
        rows.append({
            "bin_idx": i,
            "bin_low": bins[i],
            "bin_high": bins[i + 1],
            "n": len(sub),
            "predicted": sub["pred"].mean(),
            "actual": sub["out"].mean(),
        })
    return pd.DataFrame(rows)


def calibration_slope_intercept(predictions: pd.Series, outcomes: pd.Series) -> dict:
    """
    Fit logistic regression of outcome ~ logit(predictions).
    Slope should be 1.0 (well-calibrated); intercept should be 0.0.
    Slope < 1 means overconfident; slope > 1 means underconfident.
    """
    from scipy.optimize import minimize
    p = np.clip(predictions.values, 1e-4, 1 - 1e-4)
    y = outcomes.values.astype(float)

    def nll(params):
        a, b = params
        logit_p = np.log(p / (1 - p))
        logit_q = a + b * logit_p
        q = 1 / (1 + np.exp(-logit_q))
        q = np.clip(q, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    res = minimize(nll, [0.0, 1.0], method="L-BFGS-B")
    return {"intercept": float(res.x[0]), "slope": float(res.x[1])}


# ---------------------------------------------------------------------------
# Secondary metrics — profitability
# ---------------------------------------------------------------------------

def yield_pct(bets: pd.DataFrame) -> float:
    """Yield = total profit / total staked, in %."""
    staked = bets["stake"].sum()
    profit = bets["profit"].sum()
    if staked == 0:
        return 0.0
    return float(profit / staked * 100)


def roi(bets: pd.DataFrame) -> float:
    """Same as yield — kept for compatibility."""
    return yield_pct(bets)


def hit_rate(bets: pd.DataFrame) -> float:
    """% of bets won (excluding pushes)."""
    decided = bets[bets["won"].notna()]
    if decided.empty:
        return 0.0
    return float(decided["won"].mean())


def closing_line_value(bets: pd.DataFrame) -> dict:
    """
    CLV = (your_odds / closing_odds) - 1, in %.
    In Phase 0 with closing-only odds, CLV ≡ 0 by construction.
    Phase 1 with time-stamped odds will compute real CLV.

    Bootstrap 95% CI included.
    """
    if "closing_odds" not in bets.columns or bets.empty:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    clv = (bets["odds"] / bets["closing_odds"] - 1.0) * 100
    if clv.empty:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = float(clv.mean())
    if len(clv) > 30:
        # Bootstrap CI
        boots = [np.random.choice(clv.values, len(clv), replace=True).mean()
                 for _ in range(1000)]
        ci_low = float(np.percentile(boots, 2.5))
        ci_high = float(np.percentile(boots, 97.5))
    else:
        ci_low = ci_high = mean
    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "n": len(clv)}


def max_drawdown(bankroll_series: pd.Series) -> float:
    """Peak-to-trough drawdown in %."""
    cummax = bankroll_series.cummax()
    drawdown = (bankroll_series - cummax) / cummax
    return float(drawdown.min() * 100)


def longest_losing_streak(bets: pd.DataFrame) -> int:
    """Longest run of consecutive losing bets (excluding pushes)."""
    decided = bets[bets["won"].notna()]
    if decided.empty:
        return 0
    losses = decided["won"].astype(bool).values  # True = won, False = lost
    longest = current = 0
    for w in losses:
        if not w:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def sharpe_like_ratio(bets: pd.DataFrame) -> float:
    """Mean bet return / std bet return — risk-adjusted performance."""
    if bets.empty:
        return 0.0
    returns = bets["profit"] / bets["stake"].replace(0, np.nan)
    returns = returns.dropna()
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std())


def summary_table(bets: pd.DataFrame, predictions: pd.DataFrame,
                   outcomes: pd.Series, market_name: str = "all") -> dict:
    """Compute all metrics for one market slice."""
    return {
        "market": market_name,
        "n_bets": len(bets),
        "n_predictions": len(predictions),
        "brier": brier_score(predictions, outcomes) if not predictions.empty else None,
        "log_loss": log_loss(predictions, outcomes) if not predictions.empty else None,
        "yield_pct": yield_pct(bets) if not bets.empty else None,
        "hit_rate": hit_rate(bets) if not bets.empty else None,
        "max_drawdown_pct": max_drawdown(bets["cum_bankroll"]) if not bets.empty and "cum_bankroll" in bets else None,
        "longest_losing_streak": longest_losing_streak(bets) if not bets.empty else None,
        "sharpe_like": sharpe_like_ratio(bets) if not bets.empty else None,
        "clv": closing_line_value(bets) if not bets.empty else None,
    }
