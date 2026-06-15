"""
bl_model.py
===========
Core Black-Litterman implementation for S&P 500 Sector Rotation.

Covers:
  - Data download (sector ETFs via yfinance)
  - Covariance estimation (sample + Ledoit-Wolf shrinkage)
  - Reverse optimisation  (equilibrium implied returns)
  - BL master formula     (posterior mu and Sigma)
  - Mean-Variance optimiser (max Sharpe)

Author: Lewis Dang
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# 1. UNIVERSE DEFINITION
# ---------------------------------------------------------------------------

# The 11 S&P 500 SPDR sector ETFs and their human-readable names.
# We use these because:
#   - They cover the entire S&P 500 with no overlap
#   - Sector weights within the index are published and stable
#   - 11 assets => a clean 11x11 covariance matrix (far more reliable than
#     estimating a 500x500 matrix from the same data)
#   - Sector rotation is a well-documented and practically used strategy

SECTORS = {
    "XLB":  "Materials",
    "XLC":  "Communication Services",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLK":  "Technology",
    "XLP":  "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU":  "Utilities",
    "XLV":  "Health Care",
    "XLY":  "Consumer Discretionary",
}

TICKERS = list(SECTORS.keys())

# S&P 500 sector weights (approximate, as of early 2024).
# Source: SPDR / State Street. These represent the equilibrium "market portfolio"
# weights -- the starting point for the BL reverse-optimisation step.
# In a production system you would pull these live; here we fix them to keep
# the model transparent and reproducible.

MARKET_WEIGHTS = {
    "XLB":  0.025,   # Materials         ~2.5%
    "XLC":  0.085,   # Comm. Services    ~8.5%
    "XLE":  0.040,   # Energy            ~4.0%
    "XLF":  0.130,   # Financials       ~13.0%
    "XLI":  0.085,   # Industrials       ~8.5%
    "XLK":  0.290,   # Technology       ~29.0%
    "XLP":  0.060,   # Consumer Staples  ~6.0%
    "XLRE": 0.025,   # Real Estate       ~2.5%
    "XLU":  0.025,   # Utilities         ~2.5%
    "XLV":  0.125,   # Health Care      ~12.5%
    "XLY":  0.110,   # Consumer Disc.   ~11.0%
}

# ---------------------------------------------------------------------------
# 2. DATA DOWNLOAD
# ---------------------------------------------------------------------------

def get_returns(start="2010-01-01", end="2024-12-31", freq="ME"):
    """
    Download adjusted closing prices for all sector ETFs and convert to
    periodic returns.

    Parameters
    ----------
    start, end : str   Date range in 'YYYY-MM-DD' format.
    freq       : str   Pandas offset alias. 'ME' = month-end (default).

    Returns
    -------
    pd.DataFrame  shape (T, 11) -- period returns indexed by date.
    """
    raw = yf.download(TICKERS, start=start, end=end,
                      auto_adjust=True, progress=False)["Close"]
    prices = raw.resample(freq).last()
    returns = prices.pct_change().dropna()
    returns = returns[TICKERS]
    return returns


# ---------------------------------------------------------------------------
# 3. COVARIANCE ESTIMATION
# ---------------------------------------------------------------------------

def sample_cov(returns):
    """Annualised sample covariance matrix (x12 for monthly data)."""
    return returns.cov() * 12


def ledoit_wolf_cov(returns):
    """
    Ledoit-Wolf shrinkage covariance matrix (annualised).

    Why shrinkage?
    The sample covariance matrix is an unbiased but noisy estimator -- it
    amplifies estimation error, especially when the number of assets (p) is
    large relative to the number of observations (T). Ledoit-Wolf shrinks
    the sample matrix toward a structured target (scaled identity), reducing
    estimation error at the cost of a small bias. This produces more stable
    portfolio weights downstream.
    """
    lw = LedoitWolf().fit(returns)
    cov = pd.DataFrame(lw.covariance_ * 12,
                       index=returns.columns,
                       columns=returns.columns)
    return cov


# ---------------------------------------------------------------------------
# 4. REVERSE OPTIMISATION (market-implied equilibrium returns)
# ---------------------------------------------------------------------------

def reverse_optimise(cov, weights=None, delta=2.5):
    """
    Compute the market-equilibrium implied returns vector pi.

    Formula:  pi = delta * Sigma * w

    Intuition: if we assume the market cap-weighted portfolio is mean-variance
    efficient, we can work backwards from the weights to infer what expected
    returns must have produced those weights. This gives us a sensible,
    diversified prior instead of relying on noisy historical return estimates.

    Parameters
    ----------
    cov     : pd.DataFrame  Annualised covariance matrix (N x N).
    weights : dict          Market cap weights keyed by ticker.
                            Defaults to MARKET_WEIGHTS.
    delta   : float         Market risk-aversion coefficient (default 2.5).

    Returns
    -------
    pd.Series  Annualised implied returns for each sector (N,).
    """
    if weights is None:
        weights = MARKET_WEIGHTS

    w = pd.Series(weights)[cov.columns]
    w = w / w.sum()
    pi = delta * cov.values @ w.values
    return pd.Series(pi, index=cov.columns, name="pi")


# ---------------------------------------------------------------------------
# 5. BLACK-LITTERMAN MASTER FORMULA
# ---------------------------------------------------------------------------

def black_litterman(pi, cov, P, Q, omega=None, tau=0.05):
    """
    Black-Litterman posterior expected returns and covariance.

    Uses the numerically stable form (Walters 2011) that avoids
    inverting Omega directly:

        mu_BL    = pi + tau*Sigma*P' * [P*tau*Sigma*P' + Omega]^-1 * [Q - P*pi]
        Sigma_BL = Sigma + tau*Sigma - tau*Sigma*P' * [P*tau*Sigma*P' + Omega]^-1 * P*tau*Sigma

    Parameters
    ----------
    pi    : pd.Series    Prior (equilibrium) expected returns  (N,).
    cov   : pd.DataFrame Prior covariance matrix               (N x N).
    P     : np.ndarray   Pick matrix linking views to assets   (K x N).
    Q     : np.ndarray   View returns vector                   (K,).
    omega : np.ndarray   View uncertainty matrix               (K x K).
                         If None, uses He-Litterman default:
                         Omega = diag(P * tau*Sigma * P')
    tau   : float        Uncertainty scalar on the prior (default 0.05).

    Returns
    -------
    mu_bl    : pd.Series     Posterior expected returns (N,).
    sigma_bl : pd.DataFrame  Posterior covariance matrix (N x N).
    """
    pi_vec = pi.values.reshape(-1, 1)
    Q_vec  = np.array(Q).reshape(-1, 1)
    Sigma  = cov.values

    tau_sigma = tau * Sigma
    M = P @ tau_sigma @ P.T

    if omega is None:
        omega = np.diag(np.diag(M))

    # Add small regularisation to ensure the matrix is invertible
    # (prevents divide-by-zero and overflow in edge cases)
    K = M.shape[0]
    reg = 1e-8 * np.eye(K)
    inner_inv = np.linalg.inv(M + omega + reg)

    mu_bl    = pi_vec + tau_sigma @ P.T @ inner_inv @ (Q_vec - P @ pi_vec)
    sigma_bl = Sigma + tau_sigma - tau_sigma @ P.T @ inner_inv @ P @ tau_sigma

    # Clamp any NaN/inf that slipped through back to the prior
    mu_bl_flat = mu_bl.flatten()
    if not np.all(np.isfinite(mu_bl_flat)):
        mu_bl_flat = pi.values.copy()

    mu_bl_series = pd.Series(mu_bl_flat, index=pi.index, name="mu_BL")
    sigma_bl_df  = pd.DataFrame(sigma_bl, index=cov.index, columns=cov.columns)

    return mu_bl_series, sigma_bl_df


# ---------------------------------------------------------------------------
# 6. MEAN-VARIANCE OPTIMISER (Max Sharpe)
# ---------------------------------------------------------------------------

def max_sharpe(mu, cov, rf=0.04, weight_bounds=(0.0, 0.4)):
    """
    Compute the Maximum Sharpe Ratio portfolio weights via numerical
    optimisation.

    Weights are constrained to [0, 40%] per sector -- a standard
    practical constraint to prevent extreme concentration.

    Parameters
    ----------
    mu            : pd.Series    Expected returns vector.
    cov           : pd.DataFrame Covariance matrix.
    rf            : float        Risk-free rate (annualised). Default 4%.
    weight_bounds : tuple        (min, max) weight per asset.

    Returns
    -------
    pd.Series  Optimal portfolio weights summing to 1.
    """
    n = len(mu)
    mu_e = mu.values - rf

    def neg_sharpe(w):
        port_ret = w @ mu_e
        port_vol = np.sqrt(w @ cov.values @ w)
        return -port_ret / port_vol if port_vol > 1e-8 else 1e8

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [weight_bounds] * n
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-9, "maxiter": 1000})

    weights = pd.Series(result.x, index=mu.index, name="BL_weights")
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# 7. PORTFOLIO METRICS
# ---------------------------------------------------------------------------

def portfolio_metrics(weights, returns, rf=0.04):
    """
    Compute annualised performance metrics for a static weight vector.

    Returns dict with: annualised_return, annualised_vol, sharpe_ratio,
    max_drawdown, calmar_ratio.
    """
    port_ret    = (returns * weights).sum(axis=1)
    ann_ret     = port_ret.mean() * 12
    ann_vol     = port_ret.std() * np.sqrt(12)
    sharpe      = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0

    cumulative  = (1 + port_ret).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    max_dd      = drawdown.min()
    calmar      = ann_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        "annualised_return": ann_ret,
        "annualised_vol":    ann_vol,
        "sharpe_ratio":      sharpe,
        "max_drawdown":      max_dd,
        "calmar_ratio":      calmar,
    }
