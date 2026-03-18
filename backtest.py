"""
BTC 4H Backtesting Engine — Trading AutoResearch
=================================================

## AGENT-MODIFIABLE SECTION (between the markers below)
You CAN change:
  - run_strategy() logic: indicators, entry/exit signals, thresholds
  - Helper functions for indicators or signal generation
  - Constants like EMA periods, RSI levels, stop-loss/take-profit %

## FIXED SECTION (do NOT modify)
  - load_data()
  - compute_metrics()
  - main block and SCORE output
"""

import pandas as pd
import numpy as np

# ============================================================
# FIXED — Data Loading
# ============================================================

def load_data(path="data/btc_4h.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ============================================================
# FIXED — Metric Computation
# ============================================================

def compute_metrics(trades, df):
    """
    Compute composite_score = sharpe_ratio - 0.5 * max_drawdown_pct.

    trades: list of dicts with keys:
        entry_idx, exit_idx, entry_price, exit_price, direction ('long'/'short'), size
    """
    if not trades:
        return {"sharpe_ratio": 0.0, "max_drawdown_pct": 100.0, "composite_score": -50.0,
                "total_return_pct": 0.0, "num_trades": 0, "win_rate": 0.0}

    # Build equity curve from trades
    initial_capital = 10000.0
    equity = initial_capital
    equity_curve = [initial_capital]
    returns = []

    for t in trades:
        if t["direction"] == "long":
            pnl = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * t["size"]
        else:
            pnl = (t["entry_price"] - t["exit_price"]) / t["entry_price"] * t["size"]
        equity += pnl
        equity_curve.append(equity)
        returns.append(pnl / (equity - pnl) if (equity - pnl) > 0 else 0.0)

    # Sharpe ratio (annualized, 4H bars => 6 per day => ~2190 per year)
    if len(returns) < 2:
        sharpe_ratio = 0.0
    else:
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r == 0:
            sharpe_ratio = 0.0
        else:
            # Annualize: assume average holding ~1 bar, scale by sqrt(bars/year)
            sharpe_ratio = (mean_r / std_r) * np.sqrt(2190 / max(len(trades), 1) * len(trades))

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd
    max_drawdown_pct = max_dd * 100.0

    # Win rate
    wins = sum(1 for r in returns if r > 0)
    win_rate = wins / len(returns) * 100.0 if returns else 0.0

    total_return_pct = (equity - initial_capital) / initial_capital * 100.0
    composite_score = sharpe_ratio - 0.5 * max_drawdown_pct

    return {
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "composite_score": round(composite_score, 4),
        "total_return_pct": round(total_return_pct, 2),
        "num_trades": len(trades),
        "win_rate": round(win_rate, 2),
    }


# ============================================================
# AGENT-MODIFIABLE — Strategy
# ============================================================

def run_strategy(df):
    """
    EMA crossover + RSI momentum filter + ATR trailing stop.

    Changes from iter 32:
    - Remove EMA crossover exit: let trailing stop manage all exits
      instead of cutting winners short on EMA cross-back
    """
    # --- Parameters ---
    fast_ema = 9
    slow_ema = 21
    trend_ema = 50
    rsi_period = 14
    rsi_upper = 60
    atr_period = 14
    atr_trail_multiplier = 5.6
    breakeven_atr_mult = 99.0  # Disabled: Move stop to breakeven after this many ATR in profit
    take_profit_atr_mult = 99.0  # Disabled for now
    position_size = 1000.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_period = 20
    vol_mult = 1.2
    adx_period = 14
    adx_threshold = 22
    adx_upper = 50  # Raised cap: allow entries in strong trends

    # --- Indicators ---
    df = df.copy()

    # EMA crossover signals
    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()
    df["ema_trend"] = df["close"].ewm(span=trend_ema, adjust=False).mean()

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # MACD for entry confirmation
    ema_macd_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_macd_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["macd_line"] = ema_macd_fast - ema_macd_slow
    df["macd_signal"] = df["macd_line"].ewm(span=macd_signal, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # ATR for trailing stop
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(span=atr_period, adjust=False).mean()

    # Volume filter
    df["vol_avg"] = df["volume"].rolling(window=vol_period).mean()

    # ADX (Average Directional Index) for trend strength
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_for_adx = true_range.ewm(span=adx_period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=adx_period, adjust=False).mean() / atr_for_adx.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=adx_period, adjust=False).mean() / atr_for_adx.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(span=adx_period, adjust=False).mean()
    df["adx"] = df["adx"].fillna(0)

    # --- Signal generation (no look-ahead) ---
    trades = []
    position = None
    entry_idx = None
    entry_price = None
    stop_price = None
    highest_close = None
    lowest_close = None
    rsi_lower = 35  # RSI floor for short entries (avoid oversold)
    last_exit_idx = -999  # Track last exit for cooldown
    cooldown_bars = 6  # Wait this many bars after a losing trade

    warmup = trend_ema + 2

    for i in range(warmup, len(df)):
        price = df["close"].iloc[i]
        low = df["low"].iloc[i]
        atr = df["atr"].iloc[i]
        ema_f = df["ema_fast"].iloc[i]
        ema_s = df["ema_slow"].iloc[i]
        ema_t = df["ema_trend"].iloc[i]
        rsi = df["rsi"].iloc[i]
        prev_ema_f = df["ema_fast"].iloc[i - 1]
        prev_ema_s = df["ema_slow"].iloc[i - 1]
        macd_hist = df["macd_hist"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_avg = df["vol_avg"].iloc[i]
        vol_ok = vol > vol_mult * vol_avg if pd.notna(vol_avg) else False
        adx_val = df["adx"].iloc[i]

        if position is None:
            # Cooldown: skip entries for a few bars after a losing trade
            in_cooldown = (i - last_exit_idx) < cooldown_bars
            # Long entry: EMA crossover + trend + RSI filter + ADX + MACD confirmation
            # Require price to be at least 0.3% above/below trend EMA
            trend_margin = 0.003
            if (not in_cooldown
                    and prev_ema_f <= prev_ema_s and ema_f > ema_s
                    and price > ema_t * (1 + trend_margin)
                    and rsi < rsi_upper
                    and atr > 0
                    and adx_val > adx_threshold
                    and adx_val < adx_upper
                    and macd_hist > 0):
                position = "long"
                entry_idx = i
                entry_price = price
                highest_close = price
                stop_price = price - atr_trail_multiplier * atr

            # Short entry: EMA crosses down + below trend + RSI not oversold + ADX + MACD confirmation
            elif (not in_cooldown
                    and prev_ema_f >= prev_ema_s and ema_f < ema_s
                    and price < ema_t * (1 - trend_margin)
                    and rsi > rsi_lower
                    and atr > 0
                    and adx_val > adx_threshold
                    and macd_hist < 0):
                position = "short"
                entry_idx = i
                entry_price = price
                lowest_close = price
                stop_price = price + atr_trail_multiplier * atr

        elif position == "long":
            # Move to breakeven once price exceeds entry by breakeven_atr_mult * ATR
            if price >= entry_price + breakeven_atr_mult * atr and stop_price < entry_price:
                stop_price = entry_price

            # Take profit at fixed ATR multiple
            entry_atr = df["atr"].iloc[entry_idx]
            tp_price = entry_price + take_profit_atr_mult * entry_atr
            hit_tp = df["high"].iloc[i] >= tp_price

            # Update trailing stop
            if price > highest_close:
                highest_close = price
                new_stop = highest_close - atr_trail_multiplier * atr
                if new_stop > stop_price:
                    stop_price = new_stop

            # Exit on take-profit or trailing stop
            if hit_tp or low <= stop_price:
                if hit_tp:
                    exit_price = tp_price
                elif low <= stop_price:
                    exit_price = stop_price
                else:
                    exit_price = price
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": "long",
                    "size": position_size,
                })
                position = None
                entry_idx = None
                entry_price = None
                stop_price = None
                highest_close = None
                lowest_close = None

        elif position == "short":
            # Move to breakeven once price drops below entry by breakeven_atr_mult * ATR
            if price <= entry_price - breakeven_atr_mult * atr and stop_price > entry_price:
                stop_price = entry_price

            # Take profit at fixed ATR multiple (for shorts, price moves down)
            entry_atr = df["atr"].iloc[entry_idx]
            tp_price = entry_price - take_profit_atr_mult * entry_atr
            hit_tp = df["low"].iloc[i] <= tp_price

            # Update trailing stop (for shorts, stop moves down)
            if price < lowest_close:
                lowest_close = price
                new_stop = lowest_close + atr_trail_multiplier * atr
                if new_stop < stop_price:
                    stop_price = new_stop

            high = df["high"].iloc[i]
            # Exit on take-profit or trailing stop
            if hit_tp or high >= stop_price:
                if hit_tp:
                    exit_price = tp_price
                elif high >= stop_price:
                    exit_price = stop_price
                else:
                    exit_price = price
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": "short",
                    "size": position_size,
                })
                position = None
                entry_idx = None
                entry_price = None
                stop_price = None
                highest_close = None
                lowest_close = None

    # Close any open position at end
    if position is not None:
        trades.append({
            "entry_idx": entry_idx,
            "exit_idx": len(df) - 1,
            "entry_price": entry_price,
            "exit_price": df["close"].iloc[-1],
            "direction": position,
            "size": position_size,
        })

    return trades


# ============================================================
# FIXED — Main
# ============================================================

if __name__ == "__main__":
    df = load_data()
    trades = run_strategy(df)
    metrics = compute_metrics(trades, df)

    print(f"Strategy results on {len(df)} candles:")
    print(f"  Trades:      {metrics['num_trades']}")
    print(f"  Win rate:    {metrics['win_rate']}%")
    print(f"  Return:      {metrics['total_return_pct']}%")
    print(f"  Sharpe:      {metrics['sharpe_ratio']}")
    print(f"  Max DD:      {metrics['max_drawdown_pct']}%")
    print(f"SCORE: {metrics['composite_score']}")
