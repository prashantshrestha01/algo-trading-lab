"""Performance statistics.

Includes the Deflated Sharpe Ratio (DSR) of Bailey & Lopez de Prado (2014),
" The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
Overfitting and Non-Normality". DSR estimates the probability that a
strategy's true Sharpe ratio exceeds the expected maximum Sharpe under the
null hypothesis of no skill, given that K strategies were tried.

  DSR = Phi( (SR_hat - SR_0) * sqrt(T-1) / sqrt(1 - g3*SR_hat + (g4-1)/4 * SR_hat^2) )

where SR_0 = sqrt(V) * ((1-gamma)*Phi^-1(1-1/K) + gamma*Phi^-1(1-1/(K*e))),
V = variance of the K trial Sharpe ratios, gamma = Euler-Mascheroni,
g3/g4 = skewness/kurtosis of the strategy returns, T = # observations.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

ND = NormalDist()
EULER_GAMMA = 0.5772156649015329
MIN_TRADES = 10  # below this, per-strategy stats are too noisy to rank


def _moments(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.0, 3.0
    m = x.mean()
    d = x - m
    m2 = float((d ** 2).mean())
    if m2 <= 0:
        return 0.0, 3.0
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    return m3 / m2 ** 1.5, m4 / m2 ** 2


def annualized_sharpe(daily_ret):
    daily_ret = np.asarray(daily_ret, dtype=float)
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    if len(daily_ret) < 20:
        return np.nan
    sd = daily_ret.std(ddof=1)
    if sd <= 0:
        return np.nan
    return float(daily_ret.mean() / sd * math.sqrt(252))


def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, eq / peak - 1.0, 0.0)
    return float(dd.min())


def deflated_threshold(trial_sharpes, K=None):
    """Expected max Sharpe under the null (SR_0). Used as the 'deflated bar'."""
    ts = np.asarray(trial_sharpes, dtype=float)
    ts = ts[np.isfinite(ts)]
    if K is None:
        K = len(ts)
    if K < 2 or len(ts) < 2:
        return np.nan
    V = float(np.var(ts, ddof=1))
    if V <= 0:
        return np.nan
    return float(
        math.sqrt(V)
        * (
            (1 - EULER_GAMMA) * ND.inv_cdf(1 - 1.0 / K)
            + EULER_GAMMA * ND.inv_cdf(1 - 1.0 / (K * math.e))
        )
    )


def deflated_sharpe_ratio(sr, T, K, skew, kurt, trial_sharpes):
    """P(true Sharpe > 0 | selection bias). Returns NaN when not computable."""
    if not np.isfinite(sr) or T is None or T <= 1:
        return np.nan
    sr0 = deflated_threshold(trial_sharpes, K=K)
    if not np.isfinite(sr0):
        return np.nan
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom_sq <= 0:
        return np.nan
    return float(ND.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(denom_sq)))


def strategy_stats(close_idx, bt):
    """Full stat dict for one backtest result. close_idx: DatetimeIndex."""
    net_ret = np.asarray(bt["net_ret"], dtype=float)
    equity = np.asarray(bt["equity"], dtype=float)
    trades = np.asarray(bt["trades"], dtype=float)
    n = bt["n_trades"]

    s = pd.Series(net_ret, index=pd.DatetimeIndex(close_idx))
    daily = s.resample("1D").apply(lambda x: float(np.prod(1.0 + x.values) - 1.0))
    daily_vals = daily.values
    T = int(len(daily_vals))

    total_return = float(equity[-1] - 1.0) if len(equity) else np.nan
    years = T / 365.25
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if len(equity) and years > 0 else np.nan
    sr = annualized_sharpe(daily_vals)
    skew, kurt = _moments(daily_vals)
    mdd = max_drawdown(equity)

    if n > 0:
        win_rate = float((trades > 0).mean())
        gp = float(trades[trades > 0].sum())
        gl = float(-trades[trades < 0].sum())
        if gl > 0:
            profit_factor = gp / gl
        elif gp > 0:
            profit_factor = np.inf
        else:
            profit_factor = np.nan
        avg_trade = float(trades.mean())
    else:
        win_rate, profit_factor, avg_trade = np.nan, np.nan, np.nan

    return {
        "n_trades": n,
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sr,
        "max_drawdown": mdd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "daily_T": T,
        "skew": skew,
        "kurt": kurt,
    }
