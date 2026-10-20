"""Run the full strategy grid, compute honest stats, save results.csv."""
import json
import os
import time

import numpy as np
import pandas as pd

from common import load_ohlc
from backtest import backtest
import strategies
from stats import strategy_stats, deflated_sharpe_ratio, deflated_threshold, MIN_TRADES

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    df = load_ohlc(os.path.join(HERE, "data", "GC_F_1h.csv"))
    close = df["Close"]
    idx = df.index
    print(f"data: {len(df)} hourly bars, {idx.min()} -> {idx.max()}")
    print(f"expected configs: {strategies.expected_count()}")

    rows = []
    sharpes = []
    count = 0
    for family, params, signal in strategies.iter_strategies(df):
        bt = backtest(close, signal)
        st = strategy_stats(idx, bt)
        rows.append({
            "family": family,
            "params": json.dumps(params, sort_keys=True),
            "n_trades": st["n_trades"],
            "total_return": st["total_return"],
            "cagr": st["cagr"],
            "sharpe": st["sharpe"],
            "max_drawdown": st["max_drawdown"],
            "win_rate": st["win_rate"],
            "profit_factor": st["profit_factor"],
            "avg_trade": st["avg_trade"],
            "daily_T": st["daily_T"],
            "skew": st["skew"],
            "kurt": st["kurt"],
        })
        sharpes.append(st["sharpe"])
        count += 1
        if count % 2000 == 0:
            print(f"  ... {count} configs done ({time.time()-t0:.1f}s)")

    sharpes = np.array(sharpes, dtype=float)
    K = len(rows)
    sr0 = deflated_threshold(sharpes, K=K)
    print(f"deflated bar (SR_0): {sr0:.3f}")

    for r, sr in zip(rows, sharpes):
        st_skew = r["skew"]
        r["deflated_sharpe"] = deflated_sharpe_ratio(
            sr, r["daily_T"], K, st_skew, r["kurt"], sharpes
        ) if r["n_trades"] >= MIN_TRADES else np.nan

    out = pd.DataFrame(rows)
    out_path = os.path.join(HERE, "results.csv")
    out.to_csv(out_path, index=False)

    total_trades = int(out["n_trades"].sum())
    survivors = int(((out["deflated_sharpe"] > 0.95) & (out["n_trades"] >= MIN_TRADES)).sum())
    print(f"configs tested: {K} | total trades simulated: {total_trades}")
    print(f"strategies with Deflated Sharpe > 0.95: {survivors}")
    print(f"saved: {out_path} ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
