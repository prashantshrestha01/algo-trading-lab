"""Build the HTML dashboard: headline stats, survivors table, charts, honest verdict."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_ohlc
from backtest import backtest
import strategies
from stats import strategy_stats, deflated_threshold, MIN_TRADES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x*100:.1f}%"


def fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def pretty_params(p):
    return ", ".join(f"{k}={v}" for k, v in p.items())


def main():
    df = load_ohlc(os.path.join(HERE, "data", "GC_F_1h.csv"))
    close = df["Close"]
    res = pd.read_csv(os.path.join(HERE, "results.csv"))

    K = len(res)
    eligible = res[res["n_trades"] >= MIN_TRADES].copy()
    sr0 = deflated_threshold(eligible["sharpe"].values, K=K)
    total_trades = int(res["n_trades"].sum())
    survivors = eligible[eligible["deflated_sharpe"] > 0.95]
    ranked = eligible.sort_values("deflated_sharpe", ascending=False)
    top10 = ranked.head(10)
    top3 = ranked.head(3)

    # --- re-run backtests for the top 3 to get equity curves ---
    targets = {(r["family"], r["params"]) for _, r in top3.iterrows()}
    curves = {}
    for family, params, signal in strategies.iter_strategies(df):
        key = (family, json.dumps(params, sort_keys=True))
        if key in targets:
            bt = backtest(close, signal)
            curves[key] = bt["equity"]
            if len(curves) == len(targets):
                break

    equity_imgs = []
    for i, (_, r) in enumerate(top3.iterrows(), 1):
        key = (r["family"], r["params"])
        eq = curves[key]
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(df.index, eq, lw=1.2)
        ax.set_title(f"Top {i} by Deflated Sharpe — {r['family']} ({pretty_params(json.loads(r['params']))})")
        ax.set_ylabel("Equity (start = 1.0)")
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        p = os.path.join(OUT, f"equity_top{i}.png")
        fig.savefig(p, dpi=110)
        plt.close(fig)
        equity_imgs.append(os.path.basename(p))

    # --- histogram of all Sharpe ratios vs the deflated bar ---
    fig, ax = plt.subplots(figsize=(9, 4.2))
    srs = eligible["sharpe"].values
    srs = srs[np.isfinite(srs)]
    ax.hist(srs, bins=80, alpha=0.75, edgecolor="none")
    ax.axvline(sr0, color="red", lw=2, label=f"Deflated bar SR₀ = {sr0:.2f}")
    best = float(ranked.iloc[0]["sharpe"])
    ax.axvline(best, color="orange", lw=2, ls="--", label=f"Best raw Sharpe = {best:.2f}")
    ax.set_xlabel("Annualized Sharpe ratio (per strategy)")
    ax.set_ylabel("Number of strategies")
    ax.set_title("Distribution of Sharpe ratios across all strategies vs the multiple-testing bar")
    ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(OUT, "sharpe_hist.png")
    fig.savefig(hist_path, dpi=110)
    plt.close(fig)

    # --- buy & hold baseline ---
    bt_bh = backtest(close, np.ones(len(close), dtype=bool))
    st_bh = strategy_stats(df.index, bt_bh)

    # --- verdict ---
    best_r = ranked.iloc[0]
    if len(survivors) == 0:
        verdict = (
            f"<p><strong>Nothing survives.</strong> Out of {K:,} strategies and "
            f"{total_trades:,} simulated trades, <strong>zero</strong> strategies clear the "
            f"multiple-testing bar.</p>"
            f"<p>The best raw performer — {best_r['family']} "
            f"({pretty_params(json.loads(best_r['params']))}) — posts a Sharpe of "
            f"{best_r['sharpe']:.2f} and a {fmt_pct(best_r['cagr'])} CAGR. It looks like a "
            f"winner until you ask the key question: <em>out of {K:,} tries, how good "
            f"should the luckiest one look?</em></p>"
            f"<p>The Deflated Sharpe Ratio answers exactly that. With {K:,} trials, pure "
            f"luck alone is expected to produce a Sharpe of about <strong>{sr0:.2f}</strong> "
            f"(the red line). Our best raw Sharpe is {best:.2f} — <em>below</em> what chance "
            f"predicts. Its Deflated Sharpe is {best_r['deflated_sharpe']:.3f}, i.e. there is "
            f"effectively <strong>no statistical evidence</strong> its edge is real rather "
            f"than the luckiest draw from {K:,} attempts.</p>"
            "<p><strong>This is the real lesson.</strong> Picking the 'most profitable of "
            "10,000 strategies' without a multiple-testing correction is selection bias, "
            "not research. The reel this lab is modeled on shows the winner of its sweep "
            "as if the other 9,999 attempts never happened. Any strategy that 'survives' "
            "only under that framing should be treated as overfit until it proves itself "
            "on unseen, out-of-sample data — and even then, with fresh capital at risk, "
            "skepticism remains the correct default.</p>"
        )
    else:
        verdict = (f"<p>{len(survivors)} strategie(s) clear the Deflated Sharpe bar of 0.95. "
                   "Treat even these with caution: validate out-of-sample before risking capital.</p>")

    # --- top-10 table ---
    rows_html = []
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        p = json.loads(r["params"])
        rows_html.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><strong>{}</strong></td></tr>".format(
                i, r["family"], pretty_params(p), int(r["n_trades"]),
                fmt_pct(r["total_return"]), fmt_pct(r["cagr"]), fmt(r["sharpe"]),
                fmt_pct(r["max_drawdown"]), fmt_pct(r["win_rate"]), fmt(r["profit_factor"]),
                fmt(r["deflated_sharpe"], 3)))

    eq_sections = "".join(
        f"<h3>Equity curve — top {i}</h3><img src=\"outputs/{img}\" style=\"max-width:100%\"/>"
        for i, img in enumerate(equity_imgs, 1))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Backtest Lab — What Actually Survives</title>
<style>
body {{ font-family: -apple-system, Arial, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; color: #222; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: right; }}
th {{ background: #f4f4f4; }} td:nth-child(2), td:nth-child(3) {{ text-align: left; }}
.kpi {{ display: flex; gap: 1.5em; flex-wrap: wrap; }}
.kpi div {{ background: #f7f7f7; border: 1px solid #ddd; border-radius: 8px; padding: 0.8em 1.2em; }}
.kpi b {{ font-size: 1.4em; display: block; }}
.verdict {{ background: #fff8e6; border-left: 4px solid #e0a800; padding: 1em 1.2em; }}
.disclaimer {{ font-size: 0.85em; color: #666; border-top: 1px solid #ddd; margin-top: 2em; padding-top: 1em; }}
</style></head><body>
<h1>Backtest Lab — "10,000 strategies. Here's what survives."</h1>
<p>Gold futures (GC=F, proxy for XAU/USD) · hourly bars · {df.index.min().date()} → {df.index.max().date()} ·
long-only, no leverage · costs: 0.35 USD/oz round-trip spread + 0.01%/side commission.</p>

<h2>Headline</h2>
<div class="kpi">
<div><b>{K:,}</b>strategies tested</div>
<div><b>{total_trades:,}</b>trades simulated</div>
<div><b>{len(eligible):,}</b>with ≥{MIN_TRADES} trades</div>
<div><b>{len(survivors)}</b>survive the Deflated Sharpe bar</div>
<div><b>{sr0:.2f}</b>multiple-testing bar (SR₀)</div>
</div>

<h2>What survives — top 10 by Deflated Sharpe</h2>
<p>Ranked by Deflated Sharpe (probability the edge is real after correcting for {K:,} trials),
not by raw return. Strategies with fewer than {MIN_TRADES} trades are excluded as too noisy.</p>
<table>
<tr><th>#</th><th>Family</th><th>Parameters</th><th>Trades</th><th>Total ret.</th><th>CAGR</th>
<th>Sharpe</th><th>Max DD</th><th>Win rate</th><th>Profit factor</th><th>Deflated Sharpe</th></tr>
{''.join(rows_html)}
</table>

<h2>Equity curves — top 3</h2>
{eq_sections}

<h2>All Sharpe ratios vs the bar</h2>
<img src="outputs/sharpe_hist.png" style="max-width:100%"/>
<p>Buy-and-hold gold over the same window for context: total return {fmt_pct(st_bh['total_return'])},
CAGR {fmt_pct(st_bh['cagr'])}, Sharpe {fmt(st_bh['sharpe'])}, max drawdown {fmt_pct(st_bh['max_drawdown'])}.</p>

<h2>The honest verdict</h2>
<div class="verdict">{verdict}</div>

<div class="disclaimer">
<strong>Disclaimer:</strong> this is an educational research tool. Nothing here places trades,
touches real money, or constitutes financial advice. Past (and especially backtested) performance
does not predict future results. Do not trade real capital on any strategy shown here.
</div>
</body></html>"""

    dash = os.path.join(HERE, "dashboard.html")
    with open(dash, "w") as f:
        f.write(html)
    print("dashboard:", dash)
    print("charts:", [os.path.join(OUT, i) for i in equity_imgs], hist_path)


if __name__ == "__main__":
    main()
