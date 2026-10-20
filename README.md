# Trading Lab — multi-strategy backtest with honest multiple-testing correction

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![Backtesting](https://img.shields.io/badge/Backtesting-0E7C3E?style=flat-square) ![Statistics](https://img.shields.io/badge/Deflated_Sharpe-6E40C9?style=flat-square)


Built 2026-10-02. Educational research tool only — places no trades, touches no real money.

## Run it

```bash
cd ~/workspace/trading-lab
./venv/bin/python sweep.py      # ~45s: tests 10,996 strategy configs, writes results.csv
./venv/bin/python dashboard.py # rebuilds dashboard.html + charts in outputs/
```

Open `dashboard.html` in a browser (charts load from `outputs/`).

## What's inside

- `data/` — cached GC=F (gold futures, XAU/USD proxy): 1,123 hourly bars
  (2026-07-24 → 2026-10-02) + 504 daily bars (2024-10-02 → 2026-10-02).
- `backtest.py` — vectorized long-only engine. Costs: 0.35 USD/oz round-trip
  spread + 0.01% commission per side, applied on every position change.
- `strategies.py` — 5 families, 10,996 configs: SMA/EMA crossover (2,800),
  RSI mean-reversion (261), Donchian breakout (3,948), MACD (3,069),
  Bollinger mean-reversion (918).
- `stats.py` — total return, CAGR, Sharpe, max drawdown, win rate, profit
  factor + the **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014),
  which corrects for trying thousands of strategies.
- `results.csv` — one row per strategy config with all metrics.
- `dashboard.html` — headline stats, top-10 by Deflated Sharpe, equity curves
  of the top 3, Sharpe histogram vs the deflated bar, and the honest verdict.

## Headline result (2026-10-02 run)

10,996 strategies, 185,877 simulated trades, **0 survive** the Deflated Sharpe
bar. Best raw Sharpe 2.23 (RSI 2/20/70) sits *below* the 3.58 Sharpe that pure
luck is expected to produce across 10,996 trials — its Deflated Sharpe is
≈ 0.000. The "most profitable of 10,000" is selection bias, not an edge.

---

### 👤 About the author

**Prashant Shrestha** — Cybersecurity Specialist (CDI College, in progress) · 11+ years across SOC analysis, penetration testing & network security · Mississauga, ON 🇨🇦

🔗 [LinkedIn](https://www.linkedin.com/in/cto-prashant/) · 💻 [GitHub](https://github.com/prashantshrestha01)

*Building in public — every project here is my own lab work.*
