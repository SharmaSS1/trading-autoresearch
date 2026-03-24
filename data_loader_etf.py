#!/usr/bin/env python3
"""
ETF Data Loader — fetches SPY/QQQ/IWM daily OHLCV via yfinance.
Replaces the BTC 4H exchange API loader from DTG747.
Saves to data/etf_daily.csv with columns: timestamp, open, high, low, close, volume, symbol
"""

import os
import yfinance as yf
import pandas as pd

DATA_DIR = "data"
OUTPUT_CSV = os.path.join(DATA_DIR, "etf_daily.csv")
TICKERS = ["SPY", "QQQ", "IWM"]
PERIOD = "10y"      # 10 years of daily data (~2500 bars per ticker)
INTERVAL = "1d"


def fetch_etf_data(ticker: str) -> pd.DataFrame:
    print(f"Fetching {ticker} {INTERVAL} data ({PERIOD})...")
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
    df = df.reset_index()

    # Handle MultiIndex columns (flatten if needed)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    df = df.rename(columns={"date": "timestamp"})
    df["symbol"] = ticker
    df = df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]
    df = df[df["volume"] > 0].reset_index(drop=True)
    print(f"  Got {len(df)} bars ({df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()})")
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = [fetch_etf_data(t) for t in TICKERS]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} total rows to {OUTPUT_CSV}")
    for t in TICKERS:
        n = len(df[df["symbol"] == t])
        print(f"  {t}: {n} bars")


if __name__ == "__main__":
    main()
