"""Vectorized long-only backtest engine with realistic trading costs.

Position convention: a boolean `signal` means "be long during this bar".
Trades are executed at the bar close: the position held during bar t is
signal[t-1]. Each signal change (entry or exit) is one side of a trade and
pays half the round-trip spread plus one side of commission.

Costs (per the spec):
  - spread: 0.35 USD/oz round-trip  -> 0.175 USD/oz per side
  - commission: 0.01% per side
"""
import numpy as np
import pandas as pd

SPREAD_OZ_RT = 0.35       # USD per oz, round-trip
COMMISSION_SIDE = 0.0001  # 0.01% per side


def backtest(close, signal, spread_oz_rt=SPREAD_OZ_RT, commission_side=COMMISSION_SIDE):
    close = pd.Series(np.asarray(close, dtype=float))
    n = len(close)
    sig = pd.Series(np.asarray(signal, dtype=bool)).fillna(False).values

    pos = np.zeros(n)          # position actually held during each bar
    pos[1:] = sig[:-1]         # trade at close -> effective next bar
    if n > 0:
        pos[0] = 0.0

    bar_ret = close.pct_change().fillna(0.0).values
    gross_ret = pos * bar_ret

    # one side per signal change; entering on bar 0 also pays a side
    side_change = np.abs(np.diff(sig.astype(float), prepend=0.0))
    px = close.values
    px = np.where(px > 0, px, np.nan)
    cost = side_change * (spread_oz_rt / 2.0 / px + commission_side)
    cost = np.nan_to_num(cost, nan=0.0)

    net_ret = gross_ret - cost
    equity = np.cumprod(1.0 + net_ret)

    # per-trade net returns: compound bar returns over each holding block
    trades = []
    held = pos > 0
    if held.any():
        d = np.diff(held.astype(int))
        entries = np.where(d == 1)[0] + 1
        exits = np.where(d == -1)[0]
        if held[0]:
            entries = np.r_[0, entries]
        if held[-1]:
            exits = np.r_[exits, n - 1]
        for en, ex in zip(entries, exits):
            trades.append(float(np.prod(1.0 + net_ret[en:ex + 1]) - 1.0))
    trades = np.array(trades, dtype=float)

    return {
        "net_ret": net_ret,          # per-bar net returns
        "equity": equity,            # equity curve (starts at 1.0)
        "trades": trades,            # per-trade net returns
        "n_trades": len(trades),
    }
