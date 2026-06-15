"""
backtest.py
===========
Rolling backtest engine for the Black-Litterman sector rotation strategy.

Compares four strategies on the same data:
  1. BL Momentum    -- BL posterior + max Sharpe optimiser
  2. Naive Markowitz -- sample mu/sigma directly into max Sharpe (no BL)
  3. Equal Weight   -- 1/N across all sectors
  4. Cap Weighted   -- fixed S&P 500 sector weights (buy-and-hold market)

Why compare these four?
-----------------------
- Equal Weight is the naive diversification baseline (no estimation needed).
- Cap Weighted is the "do nothing" market benchmark.
- Naive Markowitz shows what BL is improving upon -- it uses the same
  optimiser but with raw historical return estimates, which are notoriously
  noisy and unstable.
- BL Momentum is our strategy: it stabilises Markowitz by anchoring expected
  returns to the market equilibrium and only deviating in the direction of
  the momentum signal.

Author: Lewis Dang
"""

import numpy as np
import pandas as pd
from bl_model import (
    ledoit_wolf_cov, reverse_optimise, black_litterman,
    max_sharpe, portfolio_metrics, MARKET_WEIGHTS, TICKERS
)
from views import build_views


# ---------------------------------------------------------------------------
# ROLLING BACKTEST
# ---------------------------------------------------------------------------

def run_backtest(returns, train_window=36, rf=0.04,
                 n_long=3, n_short=3, confidence=0.5,
                 delta=2.5, tau=0.05):
    """
    Walk-forward backtest with monthly rebalancing.

    At each month t we:
      1. Use the past `train_window` months as the estimation window.
      2. Estimate covariance, implied returns, views.
      3. Compute portfolio weights for each of the four strategies.
      4. Record the NEXT month's realised return for each portfolio.

    Parameters
    ----------
    returns      : pd.DataFrame  Monthly returns (T x N).
    train_window : int           Months of history used for estimation (default 36).
    rf           : float         Risk-free rate (annualised).
    n_long       : int           Momentum long positions.
    n_short      : int           Momentum short positions.
    confidence   : float         BL view confidence.
    delta        : float         Risk aversion parameter.
    tau          : float         BL tau (prior uncertainty scalar).

    Returns
    -------
    results : pd.DataFrame  Monthly returns for each strategy.
    weights : pd.DataFrame  Portfolio weights over time (BL strategy only).
    """
    dates   = returns.index
    n_total = len(dates)

    bl_rets, mvo_rets, ew_rets, cw_rets = [], [], [], []
    bl_weights_history = []
    idx_dates = []

    ew_w = pd.Series(1 / len(TICKERS), index=TICKERS)
    cw_w = pd.Series(MARKET_WEIGHTS)[TICKERS]
    cw_w = cw_w / cw_w.sum()

    for t in range(train_window, n_total - 1):
        train = returns.iloc[t - train_window: t]
        next_ret = returns.iloc[t + 1]

        # -- Covariance (shared by BL and Naive MVO) --
        cov = ledoit_wolf_cov(train)

        # -- BL Strategy --
        try:
            pi  = reverse_optimise(cov, delta=delta)
            P, Q, omega, _ = build_views(train, confidence=confidence,
                                         n_long=n_long, n_short=n_short)
            mu_bl, sigma_bl = black_litterman(pi, cov, P, Q, omega, tau=tau)
            w_bl = max_sharpe(mu_bl, sigma_bl, rf=rf)
            bl_rets.append((next_ret * w_bl).sum())
            bl_weights_history.append(w_bl)
        except Exception:
            bl_rets.append(np.nan)
            bl_weights_history.append(pd.Series(np.nan, index=TICKERS))

        # -- Naive MVO (historical mean + LW cov, no BL) --
        try:
            mu_hist = train.mean() * 12   # annualised
            w_mvo   = max_sharpe(mu_hist, cov, rf=rf)
            mvo_rets.append((next_ret * w_mvo).sum())
        except Exception:
            mvo_rets.append(np.nan)

        # -- Equal Weight --
        ew_rets.append((next_ret * ew_w).sum())

        # -- Cap Weighted --
        cw_rets.append((next_ret * cw_w).sum())

        idx_dates.append(dates[t + 1])

    results = pd.DataFrame({
        "BL Momentum":      bl_rets,
        "Naive Markowitz":  mvo_rets,
        "Equal Weight":     ew_rets,
        "Cap Weighted":     cw_rets,
    }, index=idx_dates)

    weights_df = pd.DataFrame(bl_weights_history, index=idx_dates, columns=TICKERS)

    return results, weights_df


# ---------------------------------------------------------------------------
# PERFORMANCE SUMMARY TABLE
# ---------------------------------------------------------------------------

def performance_summary(results, rf=0.04):
    """
    Compute annualised performance metrics for each strategy.

    Returns
    -------
    pd.DataFrame  Rows = strategies, Columns = metrics.
    """
    rows = {}
    for col in results.columns:
        r = results[col].dropna()
        ann_ret  = r.mean() * 12
        ann_vol  = r.std()  * np.sqrt(12)
        sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan

        cum = (1 + r).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

        rows[col] = {
            "Ann. Return":  f"{ann_ret:.1%}",
            "Ann. Vol":     f"{ann_vol:.1%}",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.1%}",
            "Calmar Ratio": f"{calmar:.2f}",
        }

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# CUMULATIVE RETURNS
# ---------------------------------------------------------------------------

def cumulative_returns(results):
    """Convert period returns to cumulative wealth index (starts at 1)."""
    return (1 + results.fillna(0)).cumprod()
