"""
dashboard.py
============
Interactive Streamlit dashboard for the Black-Litterman Sector Rotation model.

Run with:
    python3 -m streamlit run dashboard.py

Author: Lewis Dang
AI Assistance: Claude (Anthropic)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from bl_model import (
    get_returns, ledoit_wolf_cov, reverse_optimise,
    black_litterman, max_sharpe, SECTORS, TICKERS, MARKET_WEIGHTS
)
from views import build_views
from backtest import run_backtest, performance_summary, cumulative_returns

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Black-Litterman Sector Rotation",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.guide-box {
    background-color: #eef4fb;
    color: #1a1a1a;
    border-left: 4px solid #4a9eff;
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 12px;
}
.term-box {
    background-color: #f0f3f7;
    color: #1a1a1a;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.insight-box {
    background-color: #e7f7ee;
    color: #1a1a1a;
    border-left: 4px solid #00a578;
    padding: 12px 16px;
    border-radius: 6px;
    margin-top: 8px;
}
.warning-box {
    background-color: #fdf1e0;
    color: #1a1a1a;
    border-left: 4px solid #e08a00;
    padding: 12px 16px;
    border-radius: 6px;
    margin-top: 8px;
}
.guide-box b, .term-box b, .insight-box b, .warning-box b { color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.title("📊 Black-Litterman Sector Rotation")
st.markdown(
    "An interactive implementation of the **Black-Litterman model** applied to "
    "S&P 500 sector ETFs. Systematic views are generated from **12-1 momentum**, "
    "and you can override any sector with your own view using the sidebar."
)

st.markdown(
    "##### What is the Black-Litterman model?\n"
    "Developed at Goldman Sachs, it builds a portfolio by **combining two things**: "
    "the market's own *equilibrium* view (a sensible, diversified starting point) and "
    "*your views* on which assets will outperform, each weighted by how **confident** "
    "you are. Naive optimisers trust noisy historical returns and end up dumping "
    "everything into one or two sectors; Black-Litterman only moves away from the "
    "market when a view gives it a reason to. The two are blended with one formula:"
)
st.latex(r"\mu_{BL} = \left[(\tau\Sigma)^{-1} + P^{\top}\Omega^{-1}P\right]^{-1}"
         r"\left[(\tau\Sigma)^{-1}\pi + P^{\top}\Omega^{-1}Q\right]")
st.markdown(
    "where **π** = market-equilibrium returns (the prior), **Σ** = covariance of "
    "returns, **τ** = how much we trust the prior, **P** and **Q** = your views "
    "(which sectors, and by how much), **Ω** = your confidence in those views, and "
    "**μ&#95;BL** = the resulting blended expected returns that get fed into the "
    "optimiser. Full definitions are in the glossary below."
)

# ---------------------------------------------------------------------------
# USER GUIDE (expandable at the top)
# ---------------------------------------------------------------------------

with st.expander("📖 What is this dashboard? — Start here if you're new", expanded=False):
    st.markdown("""
### What this does in plain English

This dashboard builds a **smart portfolio** across the 11 sectors of the S&P 500
(Technology, Financials, Energy, etc.) using a method called **Black-Litterman**.

Most portfolio models ask: *"What have returns looked like historically?"* and
allocate based on that. The problem is that historical returns are noisy and
unreliable — the model ends up putting everything into one or two sectors that
happened to do well recently, producing extreme and impractical portfolios.

**Black-Litterman solves this** by starting from a sensible baseline —
the market itself — and only deviating when there is a specific reason to.
The market (i.e. the S&P 500 cap-weighted portfolio) is treated as the
"wisdom of crowds" starting point. We then layer on our own views about
which sectors we expect to outperform, and the model blends the two together
in a mathematically principled way.

Our views come from **momentum** — sectors that have been trending up over the
past 11 months (skipping the last month) tend to continue outperforming.
This is one of the most replicated findings in finance.

---

### How to use this dashboard

**Step 1 — Look at the current portfolio (Section 1)**
See how the BL model allocates across sectors right now, compared to the
market cap-weighted benchmark. The "BL vs Market Weights" bar chart shows
exactly where the model deviates from the market and by how much.

**Step 2 — Understand the model's expected returns (Section 2)**
The chart shows two bars per sector: the equilibrium prior (what the market
implies) and the BL posterior (what the model expects after incorporating
momentum views). The gap between them shows how much momentum is influencing
the model.

**Step 3 — Check the momentum signals (Section 3)**
See which sectors are currently flagged as LONG (green, expected to outperform)
and SHORT (red, expected to underperform) based on 11 months of past returns.

**Step 4 — Add your own views (Sidebar → Manual View Overrides)**
If you think a particular sector will do unusually well or badly, move its
slider away from 0%. The model will blend your view with the momentum signal.
Try moving Technology to +25% and watch Section 1 and 2 update live.

**Step 5 — Backtest (Section 4)**
See how this strategy would have performed historically vs three benchmarks.
Every data point is out-of-sample — the model never looks into the future.

**Step 6 — Adjust model parameters (Sidebar top)**
Change δ, τ, and confidence to see how sensitive the portfolio is to these
assumptions. A good model should be robust — results shouldn't change
dramatically with small parameter tweaks.
    """)

# ---------------------------------------------------------------------------
# GLOSSARY (expandable)
# ---------------------------------------------------------------------------

with st.expander("📚 Glossary — What do all these terms mean?", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Model Parameters")

        st.markdown("""
**δ (Delta) — Risk Aversion**
Controls how risk-averse the "average investor" is assumed to be.
A higher δ means the market demands more return per unit of risk.
Black & Litterman originally set this to 2.5. In practice it is
sometimes calibrated to the S&P 500's historical Sharpe ratio divided
by its variance. Changing δ shifts the equilibrium implied returns
up or down proportionally — it does not change the *relative* ranking
of sectors, only the *level* of expected returns.

---

**τ (Tau) — Prior Uncertainty**
A small scalar (typically 0.01–0.10) expressing how uncertain we are
about the equilibrium prior. A small τ means we trust the prior strongly;
a large τ means we think the prior is quite uncertain and are more willing
to deviate from it when views arrive. τ = 0.05 is a common default. It
affects how much the views can shift the posterior away from equilibrium.

---

**Risk-Free Rate**
The return available with zero risk — typically a short-term government
bond (e.g. US T-bills). Used to compute the Sharpe ratio (excess return
per unit of risk) in the optimiser and performance metrics. Set this to
approximately the current 3-month T-bill rate. As of 2024, ~4–5% is
appropriate for the US.
        """)

        st.markdown("#### View Parameters")

        st.markdown("""
**Momentum View Confidence**
How much weight to give your views (momentum signal) relative to
the market equilibrium prior. At 0.1, the portfolio barely moves from
the market. At 0.9, the portfolio almost entirely reflects the momentum
signal and ignores the prior. At 0.5, the two are roughly balanced.
Think of it as: *"How strongly do I believe in the momentum signal?"*

---

**# Long / Short Sectors**
How many sectors to go long (overweight, expect to outperform) and
short (underweight, expect to underperform) based on momentum ranking.
With 3 long and 3 short, we form 3 relative views: rank-1 long vs
rank-1 short, rank-2 long vs rank-2 short, and rank-3 long vs rank-3 short.
More views = more information fed into BL but also more noise.

---

**Backtest Training Window**
How many months of historical data to use at each rebalancing point
when estimating the covariance matrix and computing momentum scores.
36 months (3 years) is a common choice — enough data to estimate
covariance reliably while being responsive to recent regime changes.
        """)

    with col2:
        st.markdown("#### Mathematical Components")

        st.markdown("""
**π (Pi) — Equilibrium Implied Returns**
The expected returns that are *implied* by the market cap-weighted portfolio,
assuming that portfolio is mean-variance efficient. Computed as π = δΣw.
This is the "wisdom of the market" prior — the return the model uses before
any views are added. It is always sensible and diversified because it is
anchored to the market itself.

---

**μ_BL — BL Posterior Expected Returns**
The final expected returns output by the Black-Litterman model after blending
the equilibrium prior (π) with your views (momentum). This is what gets fed
into the optimiser. The further μ_BL is from π, the more the views are
pulling the portfolio away from the market.

---

**Σ (Sigma) — Covariance Matrix**
Measures how sectors move together. High covariance between two sectors
means they tend to rise and fall at the same time — the optimiser accounts
for this and avoids concentrating in correlated sectors. We estimate
this using Ledoit-Wolf shrinkage, which is more reliable than the raw
sample covariance.

---

**P Matrix (Pick Matrix)**
A mathematical way of encoding views. Each row is one view. A +1 in
column i and -1 in column j in that row means "sector i outperforms
sector j." For absolute views (your manual overrides), there is a +1
in the relevant sector's column and zeros elsewhere.

---

**Q Vector — View Returns**
The actual expected outperformance for each view. If view 1 says
"Tech outperforms Energy by 4%", then Q[1] = 0.04. The model uses
this alongside the Pick matrix to update expected returns.

---

**Ω (Omega) — View Uncertainty**
A diagonal matrix expressing how confident you are in each view.
Small diagonal values = high confidence = the posterior is pulled
strongly toward your view. Large values = low confidence = the posterior
stays close to equilibrium. The confidence slider controls this directly.

---

**Sharpe Ratio**
Excess return (above risk-free rate) divided by volatility. The
most common risk-adjusted performance measure. A Sharpe of 1.0 means
you earned 1% of excess return for every 1% of risk taken. Higher
is better; anything above 0.5 is considered reasonable.

---

**Max Drawdown**
The worst peak-to-trough loss experienced during the backtest period.
If your portfolio went from £100 to £70 at its worst point, max
drawdown is -30%. Crucial for understanding downside risk.

---

**Calmar Ratio**
Annualised return divided by absolute max drawdown. Measures how much
return you earn per unit of drawdown risk. Higher is better.
        """)

# ---------------------------------------------------------------------------
# SIDEBAR — DATA & MODEL SETTINGS
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Model Settings")
st.sidebar.markdown("Adjust these to control the Black-Litterman model. "
                    "All charts update live.")

st.sidebar.markdown("---")
st.sidebar.markdown("**📅 Date Range**")
start_date = st.sidebar.text_input("Start date", value="2013-01-01")
end_date   = st.sidebar.text_input("End date",   value="2024-12-31")

st.sidebar.markdown("---")
st.sidebar.markdown("**📐 Equilibrium Parameters**")

delta = st.sidebar.slider(
    "δ — Risk aversion", 1.0, 5.0, 2.5, 0.1,
    help="How risk-averse the market is assumed to be. Higher = market demands more "
         "return per unit of risk. Black & Litterman used 2.5."
)
tau = st.sidebar.slider(
    "τ — Prior uncertainty", 0.01, 0.20, 0.05, 0.01,
    help="How uncertain we are about the equilibrium prior. Small = trust the "
         "market strongly. Larger = more willing to deviate when views arrive."
)
rf = st.sidebar.slider(
    "Risk-free rate", 0.0, 0.08, 0.04, 0.005, format="%.3f",
    help="Annualised risk-free rate. Set to approximately the current T-bill rate. "
         "~4% is appropriate for the US as of 2024."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**📡 Momentum Signal**")

confidence = st.sidebar.slider(
    "View confidence", 0.1, 0.9, 0.5, 0.05,
    help="How strongly to trust the momentum views vs the market equilibrium. "
         "0.1 = barely deviate from market. 0.9 = almost entirely follow momentum."
)
n_long  = st.sidebar.slider("# Long sectors",  1, 5, 3,
    help="Number of top-ranked sectors to go long (overweight).")
n_short = st.sidebar.slider("# Short sectors", 1, 5, 3,
    help="Number of bottom-ranked sectors to underweight.")
train_window = st.sidebar.slider(
    "Training window (months)", 12, 60, 36,
    help="Months of history used to estimate covariance and momentum scores "
         "at each rebalancing. 36 months is a common choice."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**🎯 Manual View Overrides**")
st.sidebar.markdown(
    "Set your own expected annual return for any sector. "
    "Leave at **0%** to use the momentum signal. "
    "Move a slider to override — BL will blend your view with the prior."
)

manual_views = {}
for ticker, name in SECTORS.items():
    val = st.sidebar.slider(
        f"{name} ({ticker})",
        min_value=-0.20, max_value=0.40,
        value=0.0, step=0.01,
        format="%.0f%%",
        key=f"view_{ticker}"
    )
    if abs(val) > 0.001:
        manual_views[ticker] = val

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Downloading market data from Yahoo Finance...")
def load_data(start, end):
    return get_returns(start=start, end=end)

try:
    returns = load_data(start_date, end_date)
except Exception as e:
    st.error(f"Failed to download data: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# COMPUTE CURRENT BL PORTFOLIO
# ---------------------------------------------------------------------------

cov    = ledoit_wolf_cov(returns)
pi     = reverse_optimise(cov, delta=delta)
P, Q, omega, meta = build_views(
    returns,
    manual_overrides=manual_views if manual_views else None,
    confidence=confidence,
    n_long=n_long,
    n_short=n_short
)
mu_bl, sigma_bl = black_litterman(pi, cov, P, Q, omega, tau=tau)
w_bl = max_sharpe(mu_bl, sigma_bl, rf=rf)

w_ew = pd.Series(1 / len(TICKERS), index=TICKERS)
w_cw = pd.Series(MARKET_WEIGHTS)[TICKERS]
w_cw = w_cw / w_cw.sum()

# ---------------------------------------------------------------------------
# ACTIVE VIEWS SUMMARY
# ---------------------------------------------------------------------------

st.markdown("---")
longs  = meta.get("long_sectors", [])
shorts = meta.get("short_sectors", [])
overrides = meta.get("manual_overrides", {})

summary_parts = []
if longs and shorts:
    long_names  = " · ".join([f"**{SECTORS[t]}**" for t in longs])
    short_names = " · ".join([f"**{SECTORS[t]}**" for t in shorts])
    summary_parts.append(f"📈 Momentum longs: {long_names}")
    summary_parts.append(f"📉 Momentum shorts: {short_names}")
if overrides:
    override_str = " · ".join([f"**{SECTORS[t]}** {v*100:+.0f}%" for t, v in overrides.items()])
    summary_parts.append(f"✍️ Manual overrides: {override_str}")

if summary_parts:
    st.markdown("### 🧠 What the model currently believes")
    for part in summary_parts:
        st.markdown(part)
    st.markdown(
        "*The BL model is blending these views with the market equilibrium prior "
        f"at a confidence level of **{confidence:.0%}**. "
        "Scroll down to see how this affects portfolio weights and expected returns.*"
    )
    st.markdown("---")

# ---------------------------------------------------------------------------
# SECTION 1 — CURRENT PORTFOLIO
# ---------------------------------------------------------------------------

st.header("1. Current Portfolio Allocation")

st.markdown(
    "These charts show how the BL model allocates across the 11 S&P 500 sectors "
    "right now, based on the current momentum signal and your settings. "
    "Compare the **BL Momentum Portfolio** (left) to the **market benchmark** (centre) — "
    "any difference represents an active bet driven by our momentum views."
)

col1, col2, col3 = st.columns(3)

with col1:
    fig = px.pie(
        values=w_bl.values,
        names=[SECTORS[t] for t in w_bl.index],
        title="BL Momentum Portfolio",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch")

with col2:
    fig2 = px.pie(
        values=w_cw.values,
        names=[SECTORS[t] for t in w_cw.index],
        title="Cap Weighted (Market Benchmark)",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig2.update_traces(textposition="inside", textinfo="percent+label")
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, width="stretch")

with col3:
    compare = pd.DataFrame({
        "BL Portfolio": w_bl.values,
        "Cap Weighted": w_cw.values,
    }, index=[SECTORS[t] for t in TICKERS])

    fig3 = px.bar(
        compare.reset_index().melt(id_vars="index"),
        x="index", y="value", color="variable", barmode="group",
        title="Active Tilts vs Market",
        labels={"index": "Sector", "value": "Weight", "variable": "Strategy"},
        color_discrete_map={"BL Portfolio": "#EF553B", "Cap Weighted": "#636EFA"},
    )
    fig3.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%",
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig3, width="stretch")

with st.expander("💡 How to interpret these charts"):
    st.markdown("""
- **Left chart**: Where the BL model is putting your money right now.
- **Centre chart**: Where a passive investor tracking the S&P 500 would be.
- **Right chart**: The active bets. Bars taller on the left (red) than right (blue)
  mean the BL model is *overweighting* that sector vs the market.
  Bars taller on the right (blue) mean the model is *underweighting* it.
- A sector weight at **40%** means the model hit the concentration cap we set —
  the signal is very strong but we cap any single sector at 40% for prudence.
- A sector weight near **0%** means the model wants minimal exposure there,
  either due to weak momentum or negative views.
    """)

# ---------------------------------------------------------------------------
# SECTION 2 — EXPECTED RETURNS
# ---------------------------------------------------------------------------

st.header("2. Expected Returns: Prior vs Posterior")
st.markdown(
    "This is the core of what Black-Litterman does. The **blue bars** (π) show "
    "the expected returns implied by the market equilibrium — the 'wisdom of the "
    "market' prior. The **red bars** (μ_BL) show what the model expects *after* "
    "incorporating your momentum views. The gap between them is the BL update."
)

er_df = pd.DataFrame({
    "Equilibrium Prior (π)": pi.values,
    "BL Posterior (μ_BL)":   mu_bl.values,
}, index=[SECTORS[t] for t in TICKERS])

fig4 = px.bar(
    er_df.reset_index().melt(id_vars="index"),
    x="index", y="value", color="variable", barmode="group",
    title="Implied vs Posterior Expected Returns (Annualised)",
    labels={"index": "Sector", "value": "Expected Return", "variable": ""},
    color_discrete_map={
        "Equilibrium Prior (π)": "#636EFA",
        "BL Posterior (μ_BL)":  "#EF553B"
    }
)
fig4.update_layout(
    xaxis_tickangle=-45,
    yaxis_tickformat=".1%",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig4, width="stretch")

# Show biggest movers
diff = mu_bl - pi
biggest_up   = diff.idxmax()
biggest_down = diff.idxmin()

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        f'<div class="insight-box">📈 <b>Biggest upward revision:</b> {SECTORS[biggest_up]} '
        f'({diff[biggest_up]*100:+.1f}% vs equilibrium). '
        f'Momentum is strongly bullish on this sector.</div>',
        unsafe_allow_html=True
    )
with col_b:
    st.markdown(
        f'<div class="warning-box">📉 <b>Biggest downward revision:</b> {SECTORS[biggest_down]} '
        f'({diff[biggest_down]*100:+.1f}% vs equilibrium). '
        f'Momentum is bearish — the model is pulling expected returns below the market prior.</div>',
        unsafe_allow_html=True
    )

with st.expander("💡 How to interpret this chart"):
    st.markdown("""
- If a red bar is **taller** than its blue bar, the model expects that sector to
  do *better* than what the market implies — momentum is bullish on it.
- If a red bar is **shorter** than its blue bar, the model expects that sector to
  do *worse* than the market implies — momentum is bearish on it.
- If red and blue bars are the **same height**, that sector has no view attached —
  the model simply inherits the market's implied return.
- The magnitude of the gap is controlled by **view confidence** and **τ** in the sidebar.
  Try increasing confidence and watch the gaps widen.
- These posterior returns (red bars) are what get fed into the mean-variance
  optimiser to produce the portfolio weights in Section 1.
    """)

# ---------------------------------------------------------------------------
# SECTION 3 — MOMENTUM SCORES
# ---------------------------------------------------------------------------

st.header("3. Current Momentum Signals (12-1)")
st.markdown(
    "The 12-1 momentum score for each sector is its cumulative return over the "
    "past 11 months (skipping the most recent month to avoid short-term reversal). "
    "**Green = LONG** (model expects continued outperformance). "
    "**Red = SHORT** (model expects continued underperformance). "
    "**Blue = NEUTRAL** (not included in the current set of views)."
)

scores = meta["momentum_scores"].sort_values(ascending=False)
score_df = pd.DataFrame({
    "Sector": [SECTORS[t] for t in scores.index],
    "Ticker": scores.index.tolist(),
    "12-1 Momentum Score": scores.values,
    "Signal": [
        "LONG 📈"    if t in meta["long_sectors"]  else
        "SHORT 📉"   if t in meta["short_sectors"] else
        "NEUTRAL ➡️" for t in scores.index
    ]
})

color_map = {
    "LONG 📈":     "#00CC96",
    "SHORT 📉":    "#EF553B",
    "NEUTRAL ➡️":  "#636EFA"
}
fig5 = px.bar(
    score_df, x="Sector", y="12-1 Momentum Score", color="Signal",
    color_discrete_map=color_map,
    title="12-1 Momentum Scores — Latest Month",
    text="12-1 Momentum Score",
)
fig5.update_traces(texttemplate="%{text:.1%}", textposition="outside")
fig5.update_layout(xaxis_tickangle=-45, yaxis_tickformat=".0%",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig5, width="stretch")

with st.expander("💡 How momentum scores become views"):
    st.markdown(f"""
**How the score is calculated:**
At each month, we multiply together the monthly growth factors (1 + return) for
the past 11 months, then subtract 1. The result is the compounded return over
that window. We then skip the most recent month (shift by 1) to avoid
contaminating the signal with short-term reversal.

**How scores become BL views:**
1. Rank all 11 sectors by score — highest to lowest.
2. Top {n_long} sectors → LONG. Bottom {n_short} sectors → SHORT.
3. For each long-short pair, create a **relative view**: "long sector will outperform
   short sector by X%." The magnitude X scales with the gap between their scores —
   a bigger spread means a stronger view.
4. The uncertainty (Ω) around each view is set by your confidence slider.

**Why relative views?**
Saying "Tech will return 18%" is very hard to get right. Saying "Tech will
outperform Energy" is a much easier directional call — and it's what momentum
naturally tells us. Relative views are also more robust in the BL framework
because they do not require forecasting return *levels*, only *rankings*.
    """)

# ---------------------------------------------------------------------------
# SECTION 4 — BACKTEST
# ---------------------------------------------------------------------------

st.header("4. Backtest: BL vs Benchmarks")
st.markdown(
    f"Rolling walk-forward backtest with a **{train_window}-month** estimation window, "
    "rebalancing monthly. At each month, the model estimates covariance and views "
    "using only past data, then records the *next* month's return. "
    "Every data point shown is **genuinely out-of-sample** — the model never uses future information."
)

with st.spinner("Running backtest... (this takes ~20 seconds)"):
    bt_results, bt_weights = run_backtest(
        returns, train_window=train_window, rf=rf,
        n_long=n_long, n_short=n_short, confidence=confidence,
        delta=delta, tau=tau
    )

cum = cumulative_returns(bt_results)

fig6 = px.line(
    cum,
    title="Cumulative Wealth Index (£1 invested at start)",
    labels={"value": "Portfolio Value (£)", "index": "Date", "variable": "Strategy"},
    color_discrete_map={
        "BL Momentum":     "#EF553B",
        "Naive Markowitz": "#FF7F0E",
        "Equal Weight":    "#636EFA",
        "Cap Weighted":    "#00CC96",
    }
)
fig6.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    legend_title="Strategy",
    yaxis_tickprefix="£",
    yaxis_tickformat=".2f",
)
st.plotly_chart(fig6, width="stretch")

# Performance table
st.subheader("Performance Summary")
summary = performance_summary(bt_results, rf=rf)
st.dataframe(summary, use_container_width=True)

with st.expander("💡 How to interpret the backtest"):
    st.markdown("""
**The four strategies:**

| Strategy | What it does |
|---|---|
| **BL Momentum** | Our strategy: BL posterior expected returns → max Sharpe optimiser |
| **Naive Markowitz** | Same optimiser but uses raw historical mean instead of BL — shows what BL is improving |
| **Equal Weight** | Simply puts 1/11 in every sector and rebalances monthly — the naive diversification baseline |
| **Cap Weighted** | Holds the S&P 500 sector weights passively — the market benchmark |

**What to look for:**
- **BL vs Naive Markowitz**: If BL has a better Sharpe ratio, it confirms the BL prior is stabilising
  the optimiser — which is the whole point of the model.
- **BL vs Cap Weighted**: If BL cannot beat the passive market benchmark on a risk-adjusted
  basis, the momentum signal is not adding value.
- **Max Drawdown**: How much you would have lost at the worst point. A strategy with
  higher returns but also much larger drawdowns may not be worth it in practice.
- **Calmar Ratio**: Return per unit of drawdown. Useful for comparing strategies when
  both return and risk differ simultaneously.

**Caveat:** Past performance does not guarantee future results. This backtest
does not include transaction costs, bid-ask spreads, or taxes, all of which
would reduce real-world returns.
    """)

# ---------------------------------------------------------------------------
# SECTION 5 — ROLLING WEIGHTS
# ---------------------------------------------------------------------------

st.header("5. How the Portfolio Evolves Over Time")
st.markdown(
    "This chart shows how the BL model's sector allocation has changed month by month "
    "throughout the backtest. Periods where one sector dominates the stacked area reflect "
    "strong and persistent momentum signals in that sector."
)

bt_weights_named = bt_weights.rename(columns=SECTORS)
fig7 = px.area(
    bt_weights_named,
    title="Rolling BL Portfolio Weights",
    labels={"value": "Weight", "index": "Date", "variable": "Sector"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig7.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    yaxis_tickformat=".0%",
)
st.plotly_chart(fig7, width="stretch")

with st.expander("💡 How to interpret rolling weights"):
    st.markdown("""
- A sector that **grows in the stacked area** over time is being increasingly favoured
  by the momentum signal — its score has risen relative to others.
- **Sudden shifts** in allocation often correspond to macro events: the 2020 COVID crash
  (Tech surged, Energy collapsed), the 2022 rate-rise cycle (Financials and Energy
  outperformed Tech), the 2023-24 AI boom (Tech dominant again).
- The model naturally rotates toward what is working and away from what is not —
  this is the momentum effect in action within the BL framework.
- If you increase **view confidence** in the sidebar, you should see the stacked area
  become more concentrated (fewer sectors dominate). Lower confidence makes it look
  more like equal weight.
    """)

# ---------------------------------------------------------------------------
# SECTION 6 — CORRELATION HEATMAP
# ---------------------------------------------------------------------------

st.header("6. Sector Correlation Matrix")
st.markdown(
    "Estimated using Ledoit-Wolf shrinkage. Shows how closely sectors move together. "
    "High correlation (dark red) means two sectors tend to rise and fall together — "
    "the optimiser accounts for this and avoids doubling up on correlated bets."
)

corr = returns.corr()
corr.index   = [SECTORS[t] for t in corr.index]
corr.columns = [SECTORS[t] for t in corr.columns]

fig8 = px.imshow(
    corr, text_auto=".2f", aspect="auto",
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
    title="Pairwise Sector Return Correlations",
)
fig8.update_layout(
    xaxis_tickangle=-45,
    coloraxis_colorbar=dict(title="Correlation"),
)
st.plotly_chart(fig8, width="stretch")

with st.expander("💡 How to interpret the correlation matrix"):
    st.markdown("""
- **Dark red (close to +1.0)**: Sectors move together almost perfectly. Adding both
  to a portfolio does not add much diversification benefit.
- **White (close to 0)**: Sectors are uncorrelated — combining them meaningfully
  reduces portfolio volatility.
- **Dark blue (close to -1.0)**: Sectors tend to move in opposite directions — rare
  between equity sectors but a valuable diversification property.
- Notice that Technology and Communication Services tend to be highly correlated
  (both sensitive to growth expectations and interest rates). Similarly, Energy and
  Materials often move together (commodity price sensitivity).
- The optimiser uses this matrix to diversify: if two sectors have similar expected
  returns but one is less correlated with everything else, it will prefer that one.
    """)

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    "**Methodology**: Black-Litterman (1992) with He-Litterman (1999) Omega default. "
    "Views generated from 12-1 cross-sectional momentum (Jegadeesh & Titman 1993). "
    "Covariance: Ledoit-Wolf shrinkage (2004). Optimisation: Max Sharpe, 40% per-sector cap. "
    "Data: Yahoo Finance via yfinance."
)
st.markdown(
    "**Author**: Lewis Dang · "
    "**Course**: Advanced Portfolio Construction and Analysis with Python (EDHEC) · "
    "**AI Assistance**: Dashboard built with Claude (Anthropic)"
)
