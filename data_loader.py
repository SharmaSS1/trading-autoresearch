#!/usr/bin/env python3
"""
Data Loader — Fetch BTC/USDT 4H OHLCV from multiple sources.
Tries: mtoi.db -> Binance -> Binance US -> Bybit (fallback chain).
"""

import os
import requests
import pandas as pd

DATA_DIR = "data"
OUTPUT_CSV = os.path.join(DATA_DIR, "btc_4h.csv")
MTOI_DB_PATH = "/Users/dtg/.openclaw/workspace/data/mtoi.db"
LIMIT = 1000


def fetch_binance():
    """Fetch OHLCV from Binance (global)."""
    print("Fetching BTCUSDT 4h from Binance (global)...")
    resp = requests.get("https://api.binance.com/api/v3/klines", params={
        "symbol": "BTCUSDT", "interval": "4h", "limit": LIMIT,
    }, timeout=30)
    resp.raise_for_status()
    return _parse_binance_klines(resp.json())


def fetch_binance_us():
    """Fetch OHLCV from Binance US."""
    print("Fetching BTCUSDT 4h from Binance US...")
    resp = requests.get("https://api.binance.us/api/v3/klines", params={
        "symbol": "BTCUSDT", "interval": "4h", "limit": LIMIT,
    }, timeout=30)
    resp.raise_for_status()
    return _parse_binance_klines(resp.json())


def fetch_bybit():
    """Fetch OHLCV from Bybit public API."""
    print("Fetching BTCUSDT 4h from Bybit...")
    resp = requests.get("https://api.bybit.com/v5/market/kline", params={
        "category": "spot", "symbol": "BTCUSDT", "interval": "240", "limit": LIMIT,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for k in data["result"]["list"]:
        rows.append({
            "timestamp": pd.to_datetime(int(k[0]), unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return pd.DataFrame(rows)


def _parse_binance_klines(data):
    rows = []
    for k in data:
        rows.append({
            "timestamp": pd.to_datetime(k[0], unit="ms"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        })
    return pd.DataFrame(rows)


def load_mtoi_db():
    """Try loading from local mtoi.db SQLite database."""
    if not os.path.exists(MTOI_DB_PATH):
        print(f"mtoi.db not found at {MTOI_DB_PATH}, skipping.")
        return None

    try:
        import sqlite3
        print("Loading from mtoi.db...")
        conn = sqlite3.connect(MTOI_DB_PATH)
        # Schema: id, symbol, price, recorded_at
        query = """
            SELECT recorded_at, price
            FROM price_history
            WHERE symbol = 'BTC'
            ORDER BY recorded_at
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if len(df) < 10:
            print(f"  Only {len(df)} rows in mtoi.db, not enough for OHLCV")
            return None
        # Resample spot prices into 4H OHLCV candles
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
        df = df.set_index("recorded_at")
        ohlcv = df["price"].resample("4h").agg(
            open="first", high="max", low="min", close="last"
        ).dropna()
        ohlcv["volume"] = 0.0  # no volume data from spot prices
        ohlcv = ohlcv.reset_index().rename(columns={"recorded_at": "timestamp"})
        print(f"  Resampled {len(ohlcv)} 4H candles from mtoi.db spot prices")
        return ohlcv if len(ohlcv) >= 50 else None
    except Exception as e:
        print(f"  Could not load mtoi.db: {e}")
        return None


def fetch_ohlcv():
    """Try multiple sources in order."""
    sources = [fetch_binance, fetch_binance_us, fetch_bybit]
    for source in sources:
        try:
            df = source()
            if df is not None and len(df) > 0:
                print(f"  Got {len(df)} candles")
                return df
        except Exception as e:
            print(f"  Failed: {e}")
    return None


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Try mtoi.db first
    df = load_mtoi_db()

    # If not enough data, fetch from exchanges
    if df is None or len(df) < 200:
        df_exchange = fetch_ohlcv()
        if df_exchange is None:
            print("ERROR: Could not fetch data from any source!")
            return
        if df is not None and len(df) > 0:
            df = pd.concat([df, df_exchange]).drop_duplicates(
                subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            print(f"  Merged: {len(df)} candles total")
        else:
            df = df_exchange

    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved {len(df)} candles to {OUTPUT_CSV}")
    print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")


if __name__ == "__main__":
    main()
