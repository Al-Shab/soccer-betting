"""
bivariate_poisson.py — Bivariate Poisson scoreline model.

Replaces the independent-Poisson assumption (Dixon-Coles) with a bivariate
Poisson that has a covariance parameter λ3.

Specification (Karlis & Ntzoufras 2003):
    (X, Y) ~ BP(λ1, λ2, λ3)
    X = X1 + X3  (home goals = home-only + shared)
    Y = X2 + X3  (away goals = away-only + shared)
    X1, X2, X3 ~ independent Poisson(λ1, λ2, λ3)

The λ3 parameter is the COVARIANCE between teams' goals:
- λ3 > 0: goals tend to happen together (high-scoring games, BTTS Yes, draws)
- λ3 = 0: independent (Dixon-Coles special case)
- λ3 < 0: anti-correlated (one team scores → other doesn't) — rare in soccer

Why this helps draws: when both teams score 1, the joint prob under indep
Poisson is P(X1=1)*P(X2=1). Under BP it's P(X1+X3=1)*P(X2+X3=1) which
includes the (X1=1, X2=1, X3=0) and (X1=0, X2=0, X3=1) cases. The (X3=1)
case adds the "both teams scored via shared chance generation" mass that
indep Poisson misses. Empirically, this inflates draws AND BTTS Yes.

Per-team attack/defence structure:
    λ1 = exp(attack_h + defence_a + home_adv)
    λ2 = exp(attack_a + defence_h)
    λ3 = a constant or a function of (combined xG, ppda matchup, etc.)

The λ3 parameter can be:
- A global constant (fitted from training data)
- A function of the match (e.g., λ3 = base_λ3 + γ * combined_xG_indicator)
- Estimated per-match by the LGBM head (most flexible)

For Phase 0 v5 we use λ3 as a global constant fitted from the residual
correlation between home_goals and away_goals in training data. The
LGBM head continues to predict λ1 and λ2; λ3 is added on top.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from .config import DC_MAX_GOALS


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_p = k * np.log(lam) - lam - sum(np.log(i + 1) for i in range(k))
    return np.exp(log_p)


def bivariate_poisson_pmf(i: int, j: int, lam1: float, lam2: float, lam3: float) -> float:
    """
    P(X=i, Y=j) for bivariate Poisson BP(λ1, λ2, λ3).

    P(X=i, Y=j) = Σ_{k=0}^{min(i,j)} P(X1=i-k) * P(X2=j-k) * P(X3=k)
                = Σ_{k=0}^{min(i,j)} exp(-λ1) λ1^(i-k)/(i-k)!
                                * exp(-λ2) λ2^(j-k)/(j-k)!
                                * exp(-λ3) λ3^k / k!
    """
    total = 0.0
    for k in range(min(i, j) + 1):
        p1 = poisson_pmf(i - k, lam1)
        p2 = poisson_pmf(j - k, lam2)
        p3 = poisson_pmf(k, lam3)
        total += p1 * p2 * p3
    return total


def bivariate_poisson_pmf_vec(i_arr, j_arr, lam1: float, lam2: float, lam3: float) -> np.ndarray:
    """
    Vectorized P(X=i, Y=j) for arrays i, j.

    P(X=i, Y=j) = Σ_{k=0}^{min(i,j)} P(X1=i-k) * P(X2=j-k) * P(X3=k)

    We precompute Poisson PMFs for all needed k values once, then accumulate.
    For our use case (max_goals = 8), the convolution loop is at most 8
    iterations — fast even with Python overhead.
    """
    i_arr = np.asarray(i_arr)
    j_arr = np.asarray(j_arr)
    max_k = int(np.minimum(i_arr, j_arr).max()) if len(i_arr) > 0 else 0
    if max_k == 0 and len(i_arr) > 0:
        # All entries have min(i,j)=0; just compute p1*p2
        return poisson_pmf_vec(i_arr, lam1) * poisson_pmf_vec(j_arr, lam2)

    # Precompute pmfs for all k up to max_k (vectorized Poisson PMF)
    # Also need pmf1 at i-k for all i in i_arr, k in 0..max_k
    max_i = int(i_arr.max())
    max_j = int(j_arr.max())
    pmf1 = poisson_pmf_vec(np.arange(max_i + 1), lam1)  # array of length max_i+1
    pmf2 = poisson_pmf_vec(np.arange(max_j + 1), lam2)
    pmf3 = poisson_pmf_vec(np.arange(max_k + 1), lam3)

    # For each k, add the contribution to entries where k <= min(i, j)
    result = np.zeros(len(i_arr))
    for k in range(max_k + 1):
        mask = (i_arr >= k) & (j_arr >= k)
        if not mask.any():
            continue
        result[mask] += pmf1[i_arr[mask] - k] * pmf2[j_arr[mask] - k] * pmf3[k]
    return result


def poisson_pmf_vec(k_arr, lam: float) -> np.ndarray:
    """Vectorized Poisson PMF using log-factorial table for speed."""
    k_arr = np.asarray(k_arr, dtype=int)
    if lam <= 0:
        return (k_arr == 0).astype(float)
    # Compute log(k!) for all k using a table (avoid recomputing factorial)
    max_k = int(k_arr.max()) if len(k_arr) > 0 else 0
    log_factorials = np.zeros(max_k + 1)
    for i in range(1, max_k + 1):
        log_factorials[i] = log_factorials[i - 1] + np.log(i)
    # Vectorized log PMF
    log_p = k_arr * np.log(lam) - lam - log_factorials[k_arr]
    return np.exp(log_p)


def bivariate_poisson_scoreline_matrix(lam1: float, lam2: float, lam3: float,
                                          max_goals: int = DC_MAX_GOALS) -> np.ndarray:
    """
    Build the full scoreline probability matrix using bivariate Poisson.

    λ1 + λ3 = E[home goals] (matches our LGBM-predicted home λ)
    λ2 + λ3 = E[away goals] (matches our LGBM-predicted away λ)
    λ3 controls the covariance between home and away goals.
    """
    # Ensure λ1, λ2 > 0 (λ3 must be less than both)
    lam3 = max(0.0, min(lam3, lam1 * 0.3, lam2 * 0.3))
    lam1_adj = max(0.01, lam1 - lam3)
    lam2_adj = max(0.01, lam2 - lam3)

    # Generate all (i, j) pairs
    n = max_goals + 1
    i_grid, j_grid = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    # Vectorized PMF computation
    p_flat = bivariate_poisson_pmf_vec(i_flat, j_flat, lam1_adj, lam2_adj, lam3)

    # Reshape
    P = p_flat.reshape(n, n)

    # Handle the tail (last row/col): absorb mass beyond max_goals
    # Sum of all mass in non-tail cells
    non_tail_sum = P[:max_goals, :max_goals].sum()
    # Tail row (i=max_goals, j<max_goals): 1 - sum(i<max_goals, j fixed)
    for j in range(max_goals):
        P[max_goals, j] = max(0.0, P[:max_goals, j].sum() - 0)  # approximate: just compute directly
    # Actually let's just renormalize — the bivariate Poisson is a proper
    # distribution so the total should be close to 1; small loss in the tail is fine
    # for the scoreline matrix
    # The proper way: the tail cells should absorb all mass beyond max_goals
    # But that's expensive to compute exactly. We'll renormalize and move on.
    # Re-compute tail cells as complements:
    for j in range(max_goals):
        # P(i>=max_goals, j) = 1 - P(i<max_goals, j) — across all j (marginal)
        # But we want the joint: P(i=max_goals, j) ≈ P(i>=max_goals, j) since tail is small
        non_tail_j = P[:max_goals, j].sum()
        # Marginal of i at i>=max_goals (with j fixed) — need marginal of i
        # Simplest: assume tail is small, set to remaining mass
        P[max_goals, j] = max(0.0, poisson_pmf_at_least(max_goals, lam1_adj + lam3) * poisson_pmf(j, lam2_adj))
    for i in range(max_goals):
        P[i, max_goals] = max(0.0, poisson_pmf(i, lam1_adj) * poisson_pmf_at_least(max_goals, lam2_adj + lam3))
    # Corner (max_goals, max_goals): both teams score max_goals or more (essentially 0)
    P[max_goals, max_goals] = max(0.0,
        poisson_pmf_at_least(max_goals, lam1_adj + lam3) * poisson_pmf_at_least(max_goals, lam2_adj + lam3))

    # Normalize
    total = P.sum()
    if total > 0:
        P = P / total
    return P


def poisson_pmf_at_least(k: int, lam: float) -> float:
    """P(X >= k) for Poisson(lam)."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    # 1 - CDF(k-1)
    cdf = sum(poisson_pmf(i, lam) for i in range(k))
    return max(0.0, 1.0 - cdf)


def fit_lambda3(matches: pd.DataFrame, lam_home_col: str = "pred_lam_home",
                  lam_away_col: str = "pred_lam_away",
                  prior_lambda3: float = 0.05,
                  prior_strength: float = 100.0,
                  max_lambda3: float = 0.15) -> float:
    """
    Fit the global λ3 parameter by MAP (maximum a posteriori) with a
    Gaussian prior pulling toward `prior_lambda3`.

    Bayesian shrinkage prevents the unstable swings seen in v5 (λ3 jumping
    from 0.05 → 0.21 → 0.008 across seasons). The prior_strength controls how
    strongly the prior pulls: prior_strength=100 means the prior is worth
    ~100 pseudo-matches of evidence at the prior value.

    Specifically:
        loss = -log_likelihood + (prior_strength / 2) * (λ3 - prior)² / scale²

    where scale = 0.05 (typical empirical range). The (prior_strength / scale²)
    ratio effectively weights the prior like `prior_strength` matches.

    The max_lambda3 cap prevents pathological fits that wreck the next season's
    scoreline matrix.

    λ1 and λ2 are per-match (different for each row), so we need a per-row
    scalar PMF computation. Vectorize across matches with a Python loop
    over the bivariate Poisson convolution (max_k is small, ~5-6).
    """
    from scipy.optimize import minimize_scalar

    valid = matches[
        matches[lam_home_col].notna() &
        matches[lam_away_col].notna() &
        matches["ft_home_goals"].notna() &
        matches["ft_away_goals"].notna()
    ].copy()
    if valid.empty:
        return prior_lambda3

    lam_h = valid[lam_home_col].values
    lam_a = valid[lam_away_col].values
    h = valid["ft_home_goals"].values.astype(int)
    a = valid["ft_away_goals"].values.astype(int)
    n = len(valid)

    # Prior standard deviation — smaller means stronger shrinkage
    prior_scale = 0.03  # λ3 typically lives in [0, 0.15]; 0.03 SD is reasonable

    def neg_map_objective(lam3):
        if lam3 < 0 or lam3 > max_lambda3:
            return 1e10
        nll = 0.0
        for i in range(n):
            l1 = max(0.01, lam_h[i] - lam3)
            l2 = max(0.01, lam_a[i] - lam3)
            p = bivariate_poisson_pmf(int(h[i]), int(a[i]), l1, l2, lam3)
            p = max(p, 1e-10)
            nll -= np.log(p)
        # Add Gaussian prior penalty (log-prior is quadratic in lam3)
        nll += (prior_strength / 2.0) * ((lam3 - prior_lambda3) / prior_scale) ** 2
        return nll

    # Search in [0, max_lambda3] with tight tolerance
    result = minimize_scalar(
        neg_map_objective,
        bounds=(0.0, max_lambda3),
        method="bounded",
        options={"xatol": 0.005},
    )
    return float(result.x)


def fit_lambda3_emwa(prev_lambda3: float, new_lambda3: float, alpha: float = 0.4) -> float:
    """
    Exponentially-weighted moving average smoothing across season refits.

    The new λ3 is a weighted blend of the previous λ3 and the freshly-fit λ3:
        λ3_smoothed = alpha * new_lambda3 + (1 - alpha) * prev_lambda3

    alpha=0.4 means 40% new evidence + 60% prior belief — strong enough to
    update but not so strong that one bad season swings the model entirely.
    """
    return alpha * new_lambda3 + (1.0 - alpha) * prev_lambda3
