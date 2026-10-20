"""Strategy zoo: 5 families, ~11,000 configurations, long-only.

Each generator yields (family, params_dict, signal) where signal is a boolean
numpy array: True = be long during that bar.

Grid plan (total = 10,996 configurations):
  sma_ema   : fast 5..60 x slow 80..320 step 10 x {SMA,EMA}        = 2,800
  rsi       : period 2..30 x oversold {20,25,30} x overbought {70,75,80} = 261
  donchian  : entry lookback 10..150 x exit lookback 3..30          = 3,948
  macd      : fast 6..16 x slow 20..50 x signal 6..14               = 3,069
  bollinger : period 10..60 x width 1.0..3.0 x exit {mid,band}     =   918
"""
import numpy as np
import pandas as pd


def hold_from_events(enter, exit_):
    """Vectorized enter/exit state machine: position = last event (1=enter, 0=exit).

    Exit wins if both fire on the same bar. Exits while flat are harmless no-ops.
    """
    n = len(enter)
    ev = np.full(n, np.nan)
    em = np.asarray(enter, dtype=bool)
    xm = np.asarray(exit_, dtype=bool)
    ev[em & ~xm] = 1.0
    ev[xm] = 0.0
    return pd.Series(ev).ffill().fillna(0.0).to_numpy() > 0


def _rsi(close, period):
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    ru = up.ewm(alpha=1.0 / period, adjust=False).mean()
    rd = dn.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = ru / rd.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def gen_sma_ema(close):
    fast_ps = list(range(5, 61))          # 56
    slow_ps = list(range(80, 321, 10))     # 25
    mas = {}
    for p in fast_ps + slow_ps:
        mas[("sma", p)] = close.rolling(p, min_periods=p).mean().to_numpy()
        mas[("ema", p)] = close.ewm(span=p, adjust=False).mean().to_numpy()
    for kind in ("sma", "ema"):
        for fp in fast_ps:
            fm = mas[(kind, fp)]
            for sp in slow_ps:
                sig = (fm > mas[(kind, sp)])
                sig = np.where(np.isnan(fm) | np.isnan(mas[(kind, sp)]), False, sig)
                yield ("sma_ema", {"kind": kind, "fast": fp, "slow": sp}, sig)


def gen_rsi(close):
    cv = close.to_numpy()
    for p in range(2, 31):
        r = _rsi(close, p).to_numpy()
        for os_ in (20, 25, 30):
            enter = r < os_
            for ob in (70, 75, 80):
                exit_ = r > ob
                yield ("rsi", {"period": p, "oversold": os_, "overbought": ob},
                       hold_from_events(enter, exit_))


def gen_donchian(df):
    c = df["Close"].to_numpy()
    hi_cache, lo_cache = {}, {}
    for elb in range(10, 151):
        hi_cache[elb] = df["High"].rolling(elb, min_periods=elb).max().shift(1).to_numpy()
    for xlb in range(3, 31):
        lo_cache[xlb] = df["Low"].rolling(xlb, min_periods=xlb).min().shift(1).to_numpy()
    for elb in range(10, 151):
        enter = c > hi_cache[elb]
        for xlb in range(3, 31):
            exit_ = c < lo_cache[xlb]
            yield ("donchian", {"entry_lb": elb, "exit_lb": xlb},
                   hold_from_events(enter, exit_))


def gen_macd(close):
    cv = close.to_numpy()
    fast_ps = list(range(6, 17))    # 11
    slow_ps = list(range(20, 51))   # 31
    sig_ps = list(range(6, 15))    # 9
    emas = {p: close.ewm(span=p, adjust=False).mean().to_numpy()
            for p in fast_ps + slow_ps}
    for fp in fast_ps:
        for sp in slow_ps:
            macd_line = emas[fp] - emas[sp]
            for sgp in sig_ps:
                sig_line = pd.Series(macd_line).ewm(span=sgp, adjust=False).mean().to_numpy()
                sig = macd_line > sig_line
                sig = np.where(np.isnan(macd_line), False, sig)
                yield ("macd", {"fast": fp, "slow": sp, "signal": sgp}, sig)


def gen_bollinger(close):
    cv = close.to_numpy()
    for p in range(10, 61):
        mid = close.rolling(p, min_periods=p).mean()
        sd = close.rolling(p, min_periods=p).std(ddof=1)
        midv, sdv = mid.to_numpy(), sd.to_numpy()
        for k in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0):
            lower = midv - k * sdv
            enter = cv < lower
            for mode in ("mid", "band"):
                exit_ = cv > midv if mode == "mid" else cv > lower
                yield ("bollinger", {"period": p, "width": k, "exit": mode},
                       hold_from_events(enter, exit_))


def iter_strategies(df):
    """Yield (family, params, signal) for all ~10,996 configurations."""
    close = df["Close"]
    yield from gen_sma_ema(close)
    yield from gen_rsi(close)
    yield from gen_donchian(df)
    yield from gen_macd(close)
    yield from gen_bollinger(close)


def expected_count():
    return 56 * 25 * 2 + 29 * 3 * 3 + 141 * 28 + 11 * 31 * 9 + 51 * 9 * 2
