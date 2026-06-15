"""
views.py
========
Systematic view generation for the Black-Litterman model.

This module turns raw return data into the three BL view inputs:
  - P  (K x N pick matrix)
  - Q  (K x 1 view returns)
  - Omega (K x K view uncertainty)

We use 12-1 cross-sectional momentum as our primary signal, with
an optional manual view override for the interactive dashboard.

Why 12-1 Momentum?
------------------
Momentum (Jegadeesh & Titman 1993) is one of the most robust and
replicated factors in finance. The "12-1" variant uses the past 12
months of returns, skipping the most recent month (to avoid short-term
reversal). Applied cross-sectionally across sectors, it tells us:
"rank the sectors by recent performance; expect winners to keep winning
and losers to keep lagging."

This gives us a fully systematic, data-driven way to populate Q and P
-- no subjective guesses needed. The strength of the view (Q) is
calibrated to the cross-sectional spread of momentum scores, and the
uncertainty (Omega) scales inversely with confidence.

Author: Lewis Dang
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. MOMENTUM SIGNAL
# ---------------------------------------------------------------------------

def momentum_signal(returns, lookback=12, skip=1):
    """
    Compute 12-1 cross-sectional momentum scores for each sector.

    For each period t:
      score_i = cumulative return of sector i over [t-lookback, t-skip]

    Parameters
    ----------
    returns  : pd.DataFrame  Monthly returns (T x N).
    lookback : int           Lookback window in months (default 12).
    skip     : int           Months to skip at the end (default 1).

    Returns
    -------
    pd.DataFrame  Momentum scores, same shape as returns (NaN for early rows).
    """
    # Cumulative return over [t-lookback : t-skip]
    # i.e. product of (1+r) from t-lookback to t-skip, minus 1
    scores = (
        (1 + returns)
        .rolling(lookback - skip)
        .apply(lambda x: x.prod() - 1, raw=True)
        .shift(skip)
    )
    return scores


# ---------------------------------------------------------------------------
# 2. VIEW CONSTRUCTION FROM MOMENTUM
# ---------------------------------------------------------------------------

def views_from_momentum(returns, lookback=12, skip=1,
                        n_long=3, n_short=3,
                        view_scale=0.02, confidence=0.5):
    """
    Convert momentum scores into BL view matrices (P, Q, Omega).

    Strategy:
      - Go LONG on the top n_long sectors (expected to outperform)
      - Go SHORT on the bottom n_short sectors (expected to underperform)
      - Each view is a RELATIVE view: long sector outperforms short sector by Q_k

    This produces K = n_long * n_short relative views (each long paired
    with each short), though in practice we reduce to the strongest K views
    to keep the model tractable.

    For simplicity and interpretability we use K = min(n_long, n_short)
    views: pair rank-1 long vs rank-1 short, rank-2 long vs rank-2 short, etc.

    Parameters
    ----------
    returns    : pd.DataFrame  Monthly returns (T x N).
    lookback   : int           Momentum lookback window (months).
    skip       : int           Skip window (months).
    n_long     : int           Number of top sectors to go long.
    n_short    : int           Number of bottom sectors to short.
    view_scale : float         Expected outperformance per view (annualised).
                               We scale Q by the momentum score spread so
                               stronger signals produce stronger views.
    confidence : float         View confidence in [0, 1]. Higher = tighter
                               Omega (more weight on views vs prior).

    Returns
    -------
    P     : np.ndarray  (K x N) pick matrix.
    Q     : np.ndarray  (K,)    view returns (annualised).
    omega : np.ndarray  (K x K) diagonal view uncertainty matrix.
    long_sectors  : list of str
    short_sectors : list of str
    scores        : pd.Series  latest momentum scores
    """
    scores = momentum_signal(returns, lookback=lookback, skip=skip)
    latest_scores = scores.iloc[-1].dropna()

    ranked = latest_scores.sort_values(ascending=False)
    long_sectors  = ranked.index[:n_long].tolist()
    short_sectors = ranked.index[-n_short:].tolist()

    tickers = returns.columns.tolist()
    N = len(tickers)
    K = min(n_long, n_short)

    P     = np.zeros((K, N))
    Q_arr = np.zeros(K)

    for k in range(K):
        long_idx  = tickers.index(long_sectors[k])
        short_idx = tickers.index(short_sectors[k])

        P[k, long_idx]  =  1.0
        P[k, short_idx] = -1.0

        # Scale view magnitude by the momentum score spread
        spread = ranked.iloc[k] - ranked.iloc[-(k+1)]
        Q_arr[k] = view_scale * (1 + spread)   # annualised outperformance

    # Omega: diagonal, inversely scaled by confidence
    # High confidence => small Omega => posterior close to views
    # Low confidence  => large Omega => posterior stays close to prior
    base_variance = (view_scale ** 2) * (1 - confidence) / confidence
    omega = np.diag(np.full(K, base_variance))

    return P, Q_arr, omega, long_sectors[:K], short_sectors[:K], latest_scores


# ---------------------------------------------------------------------------
# 3. MANUAL VIEW OVERRIDE (for the interactive dashboard)
# ---------------------------------------------------------------------------

def views_from_manual(tickers, manual_views, confidence=0.5):
    """
    Build BL view matrices from user-specified absolute views.

    Each view is: "I expect sector X to return Y% annually."
    These become absolute views in BL (P row has a single 1 at position X).

    Parameters
    ----------
    tickers      : list of str   All sector tickers in order.
    manual_views : dict          {ticker: expected_annual_return (decimal)}
                                 e.g. {"XLK": 0.18, "XLE": 0.06}
    confidence   : float         View confidence in [0, 1].

    Returns
    -------
    P     : np.ndarray  (K x N).
    Q     : np.ndarray  (K,).
    omega : np.ndarray  (K x K).
    """
    N = len(tickers)
    K = len(manual_views)

    P     = np.zeros((K, N))
    Q_arr = np.zeros(K)

    for k, (ticker, view_return) in enumerate(manual_views.items()):
        if ticker in tickers:
            idx = tickers.index(ticker)
            P[k, idx] = 1.0
            Q_arr[k]  = view_return

    base_variance = (0.02 ** 2) * (1 - confidence) / confidence
    omega = np.diag(np.full(K, base_variance))

    return P, Q_arr, omega


# ---------------------------------------------------------------------------
# 4. COMBINED VIEW BUILDER (momentum + optional manual overrides)
# ---------------------------------------------------------------------------

def build_views(returns, manual_overrides=None, confidence=0.5,
                n_long=3, n_short=3):
    """
    Primary entry point for view generation.

    Generates momentum-based views, then optionally merges in any
    manual overrides from the dashboard. Manual overrides replace the
    corresponding momentum view if the same sector appears.

    Parameters
    ----------
    returns          : pd.DataFrame  Monthly returns.
    manual_overrides : dict or None  {ticker: expected_return}.
    confidence       : float         Overall view confidence.
    n_long, n_short  : int           Momentum top/bottom N.

    Returns
    -------
    P, Q, omega, metadata_dict
    """
    P_mom, Q_mom, omega_mom, longs, shorts, scores = views_from_momentum(
        returns, n_long=n_long, n_short=n_short, confidence=confidence
    )

    if not manual_overrides:
        meta = {"long_sectors": longs, "short_sectors": shorts,
                "momentum_scores": scores, "view_type": "momentum"}
        return P_mom, Q_mom, omega_mom, meta

    # Stack momentum views + manual absolute views
    tickers = returns.columns.tolist()
    P_man, Q_man, omega_man = views_from_manual(
        tickers, manual_overrides, confidence=confidence
    )

    P_combined     = np.vstack([P_mom, P_man])
    Q_combined     = np.concatenate([Q_mom, Q_man])
    omega_combined = np.block([
        [omega_mom, np.zeros((len(Q_mom), len(Q_man)))],
        [np.zeros((len(Q_man), len(Q_mom))), omega_man]
    ])

    meta = {"long_sectors": longs, "short_sectors": shorts,
            "momentum_scores": scores, "manual_overrides": manual_overrides,
            "view_type": "momentum + manual"}

    return P_combined, Q_combined, omega_combined, meta
