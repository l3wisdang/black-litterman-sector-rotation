# Black-Litterman Sector Rotation with Systematic Momentum Views

**Author:** Lewis Dang  
**Course:** Advanced Portfolio Construction and Analysis with Python (EDHEC)  
**Tools:** Python · yfinance · scikit-learn · scipy · Streamlit · Plotly  
**AI Assistance:** Claude (Anthropic) — see note on AI use below

---

## Overview

This project implements the Black-Litterman (BL) asset allocation model applied to S&P 500 sector rotation. Rather than manually specifying views — as most textbook implementations do — we generate views **systematically from a 12-1 cross-sectional momentum signal**, making the strategy fully quantitative and reproducible.

The project delivers:
- A clean, modular Python implementation of the BL model from scratch
- A systematic view generation pipeline (momentum → P, Q, Ω matrices)
- A rolling walk-forward backtest comparing BL against three benchmarks
- An interactive Streamlit dashboard where any view can be overridden manually

---

## Note on AI Use

The interactive Streamlit dashboard (`dashboard.py`) was built with the assistance of Claude (Anthropic's AI assistant). The model architecture, mathematical implementation, and all design decisions — which universe to use, which signal to use for view generation, how to parameterise Ω, the backtest methodology — were developed and reasoned through independently. Claude was used as a tool to translate those decisions into working dashboard code, in the same way a developer might use it to accelerate front-end implementation.

This is documented transparently because I think that's the right approach to AI use in a learning context. Understanding *what* to build and *why* is the hard part; having a tool help with *how* to render sliders in Streamlit does not diminish the intellectual work behind the model.

---

## Motivation

### Why Black-Litterman?

The Markowitz mean-variance framework (1952) is theoretically elegant but practically problematic. It is extremely sensitive to the expected returns vector — small changes in inputs produce wildly different, often extreme portfolio weights. This "error maximisation" property (Michaud 1989) makes raw Markowitz portfolios unstable and uninvestable.

Black and Litterman (1992) solve this by replacing the noisy historical mean with a more stable prior: the market equilibrium implied returns. The idea is simple — if we assume the market cap-weighted portfolio is mean-variance efficient, we can reverse-engineer what expected returns must have produced those weights. This gives us a sensible starting point (the prior), which we then update with our own views using Bayes' theorem.

The result is a portfolio that:
1. Starts from the market consensus (never far from the benchmark)
2. Tilts away from it only to the extent that our views are confident
3. Is far more stable than naive Markowitz across different estimation periods

### Why S&P 500 Sectors?

We chose the 11 SPDR S&P 500 sector ETFs (XLK, XLF, XLE, etc.) as our investment universe for several reasons:

**Practical tractability.** Applying BL to individual stocks requires estimating a 500×500 covariance matrix from the same limited history, which is severely ill-conditioned (far more parameters than observations). With 11 sectors, our 11×11 covariance matrix is well-estimated and stable.

**No information loss.** The sector ETFs collectively *are* the S&P 500 — they cover the entire index with no overlap. We lose no exposure by working at the sector level rather than the stock level.

**Interpretability.** Sector weights are immediately understandable to any finance practitioner. A recruiter or portfolio manager can instantly see the economic story in "overweight Technology, underweight Energy."

**Real-world relevance.** Sector rotation is a live, widely-practised strategy at asset management firms. This is not a toy example — it is how many quantitative equity funds actually think about allocation at the macro level.

**Equilibrium weights are well-defined.** SPDR publishes S&P 500 sector weights, giving us clean, transparent market cap weights to use in the reverse-optimisation step.

### Why Momentum as the View Signal?

The choice of *what views to put into BL* is the most important — and most subjective — design decision. Most student implementations simply hardcode "I think sector X will return Y%" with no systematic basis. We do something different: we generate views algorithmically from the **12-1 cross-sectional momentum factor**.

Momentum (Jegadeesh & Titman 1993) is one of the most robust and widely replicated anomalies in finance. The signal works as follows: rank all sectors by their cumulative return over the past 12 months, skipping the most recent month (to avoid short-term reversal effects). Sectors that have performed well tend to continue outperforming over the next 1–3 months; underperformers tend to continue lagging.

By using this signal to populate BL views we achieve several things:
- Views are **data-driven and reproducible**, not subjective guesses
- The strategy has a **documented theoretical and empirical basis**
- The framework is extensible — any other signal (value, quality, macro) could replace or supplement momentum in exactly the same way
- The signal naturally generates **relative views** (long winner vs short loser), which is the most natural form for BL

---

## The Momentum Factor — In Depth

### What Momentum Is

Momentum is the empirical observation that assets which have performed well recently tend to continue outperforming over the next 1–12 months, and assets that have lagged tend to continue underperforming. Formally documented by Jegadeesh & Titman (1993), it is now considered one of the most robust factors in finance — it appears across stocks, sectors, countries, currencies, and asset classes.

The persistence of momentum is largely behavioural. Markets do not instantly price in new information. Investors initially underreact to good news (anchoring to prior beliefs), then gradually update as evidence accumulates — creating a slow-moving trend. At the sector level this is especially pronounced because macro tailwinds take many months to work through into prices. A rising oil price environment benefits Energy gradually; an AI investment supercycle lifts Technology over several years, not overnight.

### The 12-1 Specification

We use the "12-1" variant. At any given month **t**, the momentum score for a sector is its **cumulative return from month t-12 to month t-1** — that is 11 months of returns, starting 12 months ago and stopping 1 month ago.

The "-1" (skipping the most recent month) is deliberate. There is a well-documented **short-term reversal effect** (Jegadeesh 1990): assets that did well in the *most recent* month tend to mean-revert the following month due to microstructure effects and liquidity-driven price pressure. Including the last month would contaminate the momentum signal with this noise, so we skip it.

The window is therefore: `[t-12 ... t-1]` — 11 monthly returns, compounded.

Why 12 months and not 6 or 3? The 12-1 window is by far the most studied and replicated in the academic literature (Jegadeesh & Titman 1993, Asness et al. 2013). Shorter windows (3-1, 6-1) pick up more noise; longer windows (18-1, 24-1) begin to overlap with the mean-reversion regime where long-term losers start outperforming.

### How We Calculate It

In `views.py`, the calculation is:

```python
scores = (
    (1 + returns)
    .rolling(11)                               # 11-month rolling window
    .apply(lambda x: x.prod() - 1, raw=True)  # compound the returns
    .shift(1)                                  # skip the most recent month
)
```

**Step 1 — `(1 + returns)`**  
Convert percentage returns into growth factors. A 3% monthly return becomes 1.03. We do this before compounding because returns compound multiplicatively, not additively.

**Step 2 — `.rolling(11).apply(lambda x: x.prod() - 1)`**  
At each month, multiply together 11 consecutive growth factors and subtract 1. This gives the *cumulative* return over those 11 months — what £1 invested would have grown to, minus 1.

For example, if the Energy sector returned `[2%, 1%, -1%, 3%, 2%, 1%, 0%, 2%, 3%, 1%, 2%]` over 11 months, the score would be `(1.02 × 1.01 × 0.99 × 1.03 × ...) - 1 ≈ 16.7%`.

**Step 3 — `.shift(1)`**  
Shift the whole series forward by 1 month. At month t, we now see the score computed at month t-1 — meaning the most recent month's return is excluded. This is the "skip 1" part of 12-1.

### From Scores to BL Views

This is the bridge between the momentum signal and the BL model. Here is the exact logic applied each month:

**Step 1 — Rank the sectors.**  
Sort all 11 sectors by their current momentum score. The top 3 become our LONG sectors, the bottom 3 become our SHORT sectors.

**Step 2 — Construct relative views.**  
We do not say "I think Tech will return 18%." We say "I think Tech will *outperform* Energy by X%." These are relative views — they express a directional bet without requiring an absolute return forecast, which is much harder to get right.

In the P matrix (the K × N pick matrix), each row represents one view. A +1 in column i and -1 in column j in row k means "sector i outperforms sector j":

```
Row 1:  [..., +1 (XLK), ..., -1 (XLE), ...]  →  "Tech outperforms Energy"
Row 2:  [..., +1 (XLF), ..., -1 (XLU), ...]  →  "Financials outperforms Utilities"
Row 3:  [..., +1 (XLI), ..., -1 (XLB), ...]  →  "Industrials outperforms Materials"
```

**Step 3 — Set the view magnitude Q.**  
Rather than using a fixed outperformance number, we scale Q by the **momentum score spread** — how far apart the winner and loser actually are:

```python
spread = ranked.iloc[k] - ranked.iloc[-(k+1)]
Q[k] = view_scale * (1 + spread)
```

If the gap between the top and bottom sector is large, we express a stronger view. If all sectors are tightly clustered, we express a modest view. This makes the model adaptive — more aggressive when the signal is clear, more conservative when it is ambiguous.

**Step 4 — Set view uncertainty Ω.**  
Ω is a diagonal matrix controlling how much we trust each view relative to the prior:

```python
base_variance = (view_scale²) × (1 - confidence) / confidence
omega = diag([base_variance, base_variance, base_variance])
```

- **High confidence (0.9)** → small Ω → posterior strongly reflects views, barely influenced by equilibrium
- **Low confidence (0.1)** → large Ω → posterior barely moves from the equilibrium prior
- **Middle (0.5)** → roughly equal weighting between prior and views

### How to Interpret the Output

Suppose in December 2024 momentum scores look like this:

| Sector | 12-1 Score | Signal |
|--------|-----------|--------|
| XLK (Technology) | +32% | LONG |
| XLC (Comm. Services) | +28% | LONG |
| XLF (Financials) | +21% | LONG |
| ... | ... | NEUTRAL |
| XLE (Energy) | -8% | SHORT |
| XLU (Utilities) | -11% | SHORT |
| XLB (Materials) | -14% | SHORT |

The views passed to BL would be approximately:
- "Tech outperforms Materials by ~4.6% over the next period"
- "Comm. Services outperforms Utilities by ~3.9%"
- "Financials outperforms Energy by ~2.9%"

The BL formula blends these views with the equilibrium prior and asks: *given both pieces of information, and given our stated confidence in each, what single set of expected returns is most consistent with both?* The posterior μ_BL sits between the equilibrium (which already favours Tech because it is the largest sector) and the momentum view (which favours it further). The degree of the shift is controlled by the confidence parameter.

The key chart to watch in the dashboard is the **Prior vs Posterior expected returns plot**. If momentum is bullish on Tech, the posterior bar for XLK will be taller than the equilibrium bar. If momentum is bearish on Energy, the posterior bar for XLE will be shorter. The degree of shift makes the BL blending mechanism immediately visible.

---

## Methodology

### Step 1: Data

We download adjusted monthly closing prices for all 11 SPDR sector ETFs from Yahoo Finance via `yfinance`, covering the period 2013–2024. We convert prices to simple monthly returns.

### Step 2: Covariance Estimation

We use **Ledoit-Wolf shrinkage** rather than the raw sample covariance matrix. The sample covariance is an unbiased estimator but has high variance — it over-fits to the specific sample, producing extreme off-diagonal entries that destabilise the optimiser. Ledoit-Wolf (2004) shrinks the sample matrix toward a scaled identity matrix, optimally trading a small bias for a large reduction in variance. The result is a more stable, better-conditioned covariance matrix.

With 11 assets and a 36-month estimation window, the sample covariance has 66 unique parameters to estimate from 36 observations — an underdetermined problem. Ledoit-Wolf regularises this by analytically computing the optimal shrinkage intensity, avoiding the need to cross-validate.

### Step 3: Reverse Optimisation

We compute the market-equilibrium implied returns using:

$$\pi = \delta \Sigma w$$

where $\delta = 2.5$ is the market risk-aversion coefficient (Black & Litterman's original value), $\Sigma$ is the Ledoit-Wolf covariance matrix, and $w$ is the vector of S&P 500 sector market-cap weights.

This tells us: "if the market is in equilibrium and rational investors are holding the cap-weighted portfolio, these are the expected returns that are consistent with that behaviour." It gives us a prior that is always sensible and diversified, regardless of the historical sample.

### Step 4: View Generation

For each month in the backtest, we compute 12-1 momentum scores, identify the top and bottom sectors, construct relative views, scale view magnitudes by the momentum spread, and set uncertainty via Ω. Full detail in the momentum section above.

### Step 5: BL Master Formula

We apply the numerically stable form (Walters 2011) that avoids inverting Ω directly:

$$\mu^{BL} = \pi + \tau\Sigma P^T \left[P\tau\Sigma P^T + \Omega\right]^{-1} \left[Q - P\pi\right]$$

$$\Sigma^{BL} = \Sigma + \tau\Sigma - \tau\Sigma P^T \left[P\tau\Sigma P^T + \Omega\right]^{-1} P\tau\Sigma$$

We use $\tau = 0.05$. Some authors use 1/T (approximately 0.028 for our 36-month window); others use 1. We use 0.05 as a transparent, commonly cited middle ground reflecting moderate uncertainty about the prior.

### Step 6: Portfolio Optimisation

We feed $\mu^{BL}$ and $\Sigma^{BL}$ into a mean-variance optimiser to find the Maximum Sharpe Ratio portfolio, subject to weights summing to 1 and individual sector weights bounded to [0%, 40%]. The 40% cap is a standard practical constraint preventing extreme concentration — without it, the optimiser can place the entire portfolio in one sector when the signal is strong, which performs well in-sample but poorly out-of-sample.

### Step 7: Backtest

We run a rolling walk-forward backtest: at each month $t$, we use the past 36 months of returns to estimate covariance and generate views, then record the next month's realised return. Every observation is out-of-sample at the time of the trade. We compare against three benchmarks over the same period.

---

## Results

The backtest compares four strategies:

| Strategy | Description |
|---|---|
| **BL Momentum** | Our strategy: BL posterior + max Sharpe |
| **Naive Markowitz** | Historical mean + LW covariance into max Sharpe (no BL) |
| **Equal Weight** | 1/11 in each sector, rebalanced monthly |
| **Cap Weighted** | Fixed S&P 500 sector weights (passive benchmark) |

Key expected findings:
- BL Momentum should outperform Naive Markowitz on a risk-adjusted basis, demonstrating the stabilising effect of the BL prior
- Both optimised strategies should outperform Equal Weight in Sharpe ratio during trending regimes
- Cap Weighted provides the market benchmark; a strategy that cannot beat it is not worth the transaction costs

---

## Interactive Dashboard

The Streamlit dashboard (`dashboard.py`) was built with AI assistance (Claude, Anthropic) and allows you to:

1. **Adjust model parameters** — risk aversion (δ), tau, risk-free rate, view confidence
2. **Control the momentum signal** — number of long/short sectors, training window
3. **Override any sector view** — enter your own expected return for any sector; BL blends it with the momentum prior automatically
4. **See live portfolio weights** — watch the allocation update in real time as you move sliders
5. **Compare expected returns** — visualise how the BL posterior shifts away from the equilibrium prior
6. **Explore backtest results** — cumulative returns, performance summary table, rolling weights over time, sector correlation heatmap

### Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

---

## Project Structure

```
bl_project/
├── bl_model.py       # Core BL implementation (data, cov, reverse-opt, BL formula, optimiser)
├── views.py          # Systematic view generation (momentum signal, manual overrides)
├── backtest.py       # Rolling walk-forward backtest engine
├── dashboard.py      # Interactive Streamlit dashboard (built with AI assistance)
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Key Design Decisions

**Why Ledoit-Wolf and not sample covariance?** With 11 assets and 36 months of data, the sample covariance has 66 unique parameters to estimate from 36 observations — a badly underdetermined problem. Ledoit-Wolf regularises this by analytically computing the optimal shrinkage intensity, avoiding the need to cross-validate.

**Why δ = 2.5?** Black and Litterman's original choice. It corresponds approximately to the market's Sharpe ratio divided by market variance — a neutral assumption about aggregate investor risk aversion. The dashboard allows you to vary this.

**Why τ = 0.05?** A common choice reflecting moderate uncertainty about the prior. Some authors use 1/T (≈0.028 for our window); others use 1. We use 0.05 as a transparent, widely cited middle ground.

**Why 40% weight cap?** A standard practical constraint in sector allocation. Without it, the optimiser can concentrate the entire portfolio in one sector when the signal is strong — which overfits badly out-of-sample.

**Why 12-1 momentum and not 6-1 or 3-1?** The 12-1 window is the most studied and replicated in the academic literature. The 1-month skip avoids the short-term reversal effect. Shorter windows are noisier; longer windows begin to capture mean-reversion rather than momentum.

**Why relative views rather than absolute?** Absolute views ("Tech will return 18%") require forecasting return levels — extremely hard to do accurately. Relative views ("Tech will outperform Energy") only require directional ranking, which momentum does naturally and reliably.

---

## References

- Black, F. & Litterman, R. (1992). *Global Portfolio Optimization*. Financial Analysts Journal.
- He, G. & Litterman, R. (1999). *The Intuition Behind Black-Litterman Model Portfolios*. Goldman Sachs.
- Jegadeesh, N. (1990). *Evidence of Predictable Behavior of Security Returns*. Journal of Finance.
- Jegadeesh, N. & Titman, S. (1993). *Returns to Buying Winners and Selling Losers*. Journal of Finance.
- Asness, C., Moskowitz, T. & Pedersen, L. (2013). *Value and Momentum Everywhere*. Journal of Finance.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis.
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Michaud, R. (1989). *The Markowitz Optimization Enigma*. Financial Analysts Journal.
- Walters, J. (2011). *The Black-Litterman Model in Detail*. SSRN Working Paper.
