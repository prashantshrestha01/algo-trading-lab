"""Shared helpers: load cached OHLC data."""
import pandas as pd


def load_ohlc(csv_path):
    """Load a yfinance-cached CSV (two-row MultiIndex header) -> clean OHLC DataFrame.

    Returns tz-naive DatetimeIndex, columns Open/High/Low/Close, sorted, dupes dropped.
    """
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)  # drop ticker level
    keep = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    df[["Open", "High", "Low"]] = df[["Open", "High", "Low"]].ffill()
    return df
