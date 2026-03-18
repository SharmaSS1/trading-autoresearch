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
    # Remove zero/NaN volume candles (data quality gate)
    df = df[df["volume"] > 0].reset_index(drop=True)
    return df


# ============================================================
# FIXED — Metric Computation
# ============================================================

def compute_metrics(trades, df):
    """
    Compute composite_score = capped_sharpe - 0.5 * max_drawdown_pct.

    trades: list of dicts with keys:
        entry_idx, exit_idx, entry_price, exit_price, direction ('long'/'short'), size

    Sharpe is CAPPED at 20.0 to prevent degenerate zero-variance solutions.
    Minimum return and trade count gates are enforced in the main block.
    """
    if not trades:
        return {
            "sharpe_ratio": 0.0, "max_drawdown_pct": 100.0, "composite_score": -50.0,
            "total_return_pct": 0.0, "num_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
        }

    # Build equity curve from trades
    initial_capital = 10000.0
    equity = initial_capital
    equity_curve = [initial_capital]
    returns = []
    gross_profit = 0.0
    gross_loss = 0.0

    for t in trades:
        if t["direction"] == "long":
            pnl = (t["exit_price"] - t["entry_price"]) / t["entry_price"] * t["size"]
        else:
            pnl = (t["entry_price"] - t["exit_price"]) / t["entry_price"] * t["size"]
        equity += pnl
        equity_curve.append(equity)
        ret = pnl / (equity - pnl) if (equity - pnl) > 0 else 0.0
        returns.append(ret)
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

    # Profit factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    # Sharpe ratio (annualized, 4H = 2190 bars/year)
    if len(returns) < 2:
        sharpe_ratio = 0.0
    else:
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r < 1e-10:
            # Near-zero variance: degenerate — treat as 0, not infinity
            sharpe_ratio = 0.0
        else:
            sharpe_ratio = (mean_r / std_r) * np.sqrt(2190)

    # HARD CAP: prevents gaming via near-zero variance solutions
    sharpe_ratio = min(sharpe_ratio, 20.0)

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
        "profit_factor": round(profit_factor, 4),
    }


# ============================================================
# AGENT-MODIFIABLE — Strategy
# ============================================================

def run_strategy(df):
    """
    EMA 20/50 crossover + ADX trend filter + ATR-based stops and take-profit.

    Improvements over baseline:
      - Wider EMAs (20/50) to reduce whipsaw in choppy markets
      - ADX filter: only enter when ADX > 20 (confirmed trend)
      - ATR-based stop loss (1.5x ATR) and take profit (3x ATR)
      - RSI extreme filter: avoid buying overbought / selling oversold
    """
    # --- Parameters ---
    fast_ema = 20
    slow_ema = 50
    rsi_period = 14
    atr_period = 14
    adx_period = 14
    adx_threshold = 20
    atr_sl_mult = 1.5    # stop loss = 1.5x ATR
    atr_tp_mult = 3.0    # take profit = 3x ATR
    position_size = 1000.0

    # --- Indicators ---
    df = df.copy()

    # EMAs
    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()

    # EMA crossover signal
    df["cross"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    df["cross_prev"] = df["cross"].shift(1)

    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # ATR
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=atr_period, adjust=False).mean()

    # ADX
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    # Zero out when the other is larger
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0.0)
    minus_dm_arr = np.where(minus_dm > pd.Series(np.where(plus_dm > 0, 0, minus_dm.values), index=df.index), minus_dm, 0.0)
    # Recalculate properly
    raw_plus = (high - high.shift(1))
    raw_minus = (low.shift(1) - low)
    plus_dm = np.where((raw_plus > raw_minus) & (raw_plus > 0), raw_plus, 0.0)
    minus_dm = np.where((raw_minus > raw_plus) & (raw_minus > 0), raw_minus, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(span=adx_period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(span=adx_period, adjust=False).mean()

    plus_di = 100 * plus_dm_s / df["atr"].replace(0, np.nan)
    minus_di = 100 * minus_dm_s / df["atr"].replace(0, np.nan)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan) * 100).fillna(0)
    df["adx"] = dx.ewm(span=adx_period, adjust=False).mean()

    # --- Trade loop ---
    trades = []
    position = None
    entry_price = None
    entry_idx = None
    stop_price = None
    tp_price = None
    warmup = slow_ema + adx_period + 5

    for i in range(warmup, len(df) - 1):
        close = df["close"].iloc[i]
        cross = df["cross"].iloc[i]
        cross_prev = df["cross_prev"].iloc[i]
        rsi = df["rsi"].iloc[i]
        atr = df["atr"].iloc[i]
        adx = df["adx"].iloc[i]

        bullish_cross = (cross == 1) and (cross_prev == -1)
        bearish_cross = (cross == -1) and (cross_prev == 1)

        # Exit logic
        if position == "long":
            hit_stop = close <= stop_price
            hit_tp = close >= tp_price
            if bearish_cross or hit_stop or hit_tp:
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": close,
                    "direction": "long", "size": position_size,
                })
                position = None

        elif position == "short":
            hit_stop = close >= stop_price
            hit_tp = close <= tp_price
            if bullish_cross or hit_stop or hit_tp:
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": close,
                    "direction": "short", "size": position_size,
                })
                position = None

        # Entry logic
        if position is None:
            if bullish_cross and rsi > 45 and rsi < 75 and adx > adx_threshold:
                position = "long"
                entry_price = close
                entry_idx = i
                stop_price = close - atr_sl_mult * atr
                tp_price = close + atr_tp_mult * atr
            elif bearish_cross and rsi < 55 and rsi > 25 and adx > adx_threshold:
                position = "short"
                entry_price = close
                entry_idx = i
                stop_price = close + atr_sl_mult * atr
                tp_price = close - atr_tp_mult * atr

    # Close open position at end
    if position is not None:
        trades.append({
            "entry_idx": entry_idx, "exit_idx": len(df) - 1,
            "entry_price": entry_price, "exit_price": df["close"].iloc[-1],
            "direction": position, "size": position_size,
        })

    return trades


# ============================================================
# FIXED — Main (train/test split + minimum gates)
# ============================================================

if __name__ == "__main__":
    df = load_data()

    # Train/test split: 75% train, 25% holdout
    # Agent optimizes on TRAIN only — holdout is never seen during iteration
    split = int(len(df) * 0.75)
    train_df = df.iloc[:split].reset_index(drop=True)
    holdout_df = df.iloc[split:].reset_index(drop=True)

    # Evaluate on training data
    train_trades = run_strategy(train_df)
    train_m = compute_metrics(train_trades, train_df)

    # Evaluate on holdout (same strategy, different data window)
    holdout_trades = run_strategy(holdout_df)
    holdout_m = compute_metrics(holdout_trades, holdout_df)

    # --- Minimum requirements gate ---
    FAIL_SCORE = -100.0
    if train_m["num_trades"] < 10:
        score = FAIL_SCORE
        reason = f"too few trades ({train_m['num_trades']} < 10 required)"
    elif train_m["total_return_pct"] < 1.0:
        score = FAIL_SCORE
        reason = f"return too low ({train_m['total_return_pct']:.2f}% < 1% required)"
    elif train_m["profit_factor"] < 1.0:
        score = FAIL_SCORE
        reason = f"profit factor below 1.0 ({train_m['profit_factor']:.4f})"
    else:
        # Score: 60% train, 40% holdout — strategy must generalize
        if holdout_m["composite_score"] < -20:
            # Heavy penalty for strategies that don't transfer to holdout
            score = train_m["composite_score"] - 40.0
            reason = f"overfitting penalty (holdout={holdout_m['composite_score']:.4f})"
        else:
            score = round(0.6 * train_m["composite_score"] + 0.4 * holdout_m["composite_score"], 4)
            reason = "ok"

    print(f"Strategy results on {len(train_df)} train / {len(holdout_df)} holdout candles:")
    print(f"  [TRAIN]   Trades: {train_m['num_trades']:3d} | WR: {train_m['win_rate']:5.1f}% | "
          f"Return: {train_m['total_return_pct']:6.2f}% | Sharpe: {train_m['sharpe_ratio']:6.4f} | "
          f"MaxDD: {train_m['max_drawdown_pct']:5.2f}% | PF: {train_m['profit_factor']:.4f}")
    print(f"  [HOLDOUT] Trades: {holdout_m['num_trades']:3d} | WR: {holdout_m['win_rate']:5.1f}% | "
          f"Return: {holdout_m['total_return_pct']:6.2f}% | Sharpe: {holdout_m['sharpe_ratio']:6.4f} | "
          f"MaxDD: {holdout_m['max_drawdown_pct']:5.2f}% | PF: {holdout_m['profit_factor']:.4f}")
    print(f"  Score: {score}  ({reason})")
    print(f"SCORE: {score}")
