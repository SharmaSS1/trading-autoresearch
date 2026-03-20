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
    Phase 4 composite score formula (reality-check layer added):

        composite_score = sharpe - 0.5 * max_drawdown_pct
                        + 0.5 * ln(min(profit_factor, 4.0))   ← PF capped at 4x
                        + 0.05 * min(total_return_pct, 100)
                        - win_rate_penalty                     ← NEW: penalizes unrealistic WR
                        - too_clean_penalty                    ← NEW: penalizes near-zero DD

    Reality-check penalties (Phase 4):
      - Win rate > 70%: penalty = (win_rate% - 70) * 0.5   (linear, -5pts at 80%, -15pts at 100%)
      - Win rate < 35%: penalty = (35 - win_rate%) * 0.3   (needs some edge)
      - Profit factor contribution capped at ln(4.0) ≈ 1.39 — rewards up to PF=4, not PF=10000
      - Max drawdown < 1%: penalty = (1.0 - max_dd%) * 5.0 (too clean = degenerate or overfit)

    Sharpe is CAPPED at 3.0 (realistic live ceiling for a good algo).
    Theoretical ceiling: ~15.0 for a genuinely good real-world strategy.

    Hard gates (score = -100 if violated):
      - Minimum 20 trades on training data (raised from 10)
      - Minimum 1% total return
      - Profit factor >= 1.0

    trades: list of dicts with keys:
        entry_idx, exit_idx, entry_price, exit_price, direction ('long'/'short'), size
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

    # HARD CAP: realistic live ceiling for a good algo
    # Phase 4: lowered from 50.0 → 3.0 (Sharpe >3 in live trading is exceptional, >10 is degenerate)
    sharpe_ratio = min(sharpe_ratio, 3.0)

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

    # Phase 4 composite score: reality-check layer applied
    # PF capped at 4.0 — rewards real edge, not degenerate cherry-picking
    pf_capped = min(profit_factor, 4.0)
    pf_term = 0.5 * np.log(max(pf_capped, 1.0))               # max contribution: 0.5*ln(4) ≈ 0.69
    return_term = 0.05 * min(total_return_pct, 100.0)          # rewards absolute returns, capped at 100%

    # Win rate reality check: real strategies win 40-70% of the time
    # >70% suggests overfitting or degenerate entry filtering
    if win_rate > 70.0:
        win_rate_penalty = (win_rate - 70.0) * 0.5             # -5 pts at 80%, -15 pts at 100%
    elif win_rate < 35.0:
        win_rate_penalty = (35.0 - win_rate) * 0.3             # needs some basic edge
    else:
        win_rate_penalty = 0.0

    # "Too clean" drawdown penalty: 0% DD over 14 months = degenerate
    # Real strategies experience drawdowns — this is healthy, not bad
    if max_drawdown_pct < 1.0:
        too_clean_penalty = (1.0 - max_drawdown_pct) * 5.0     # up to -5 pts for 0% DD
    else:
        too_clean_penalty = 0.0

    composite_score = (sharpe_ratio
                       - 0.5 * max_drawdown_pct
                       + pf_term
                       + return_term
                       - win_rate_penalty
                       - too_clean_penalty)

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
    Phase 4 long-only trend-following: EMA crossover + RSI pullback + EMA bounce.

    Iter 10: Remove fixed TP — let winners run via trailing stop + trend exit.
    Tighten trailing stop from 1.8 to 1.5 ATR to lock profits. Increase max hold
    from 40 to 60 bars to give trends more room. This should significantly boost
    total return while maintaining realistic drawdown.
    """
    # --- Parameters ---
    fast_ema = 21
    slow_ema = 45
    rsi_period = 14
    atr_period = 14
    adx_period = 14
    adx_threshold = 15
    atr_sl_mult = 2.0
    atr_trail_mult = 1.5
    risk_per_trade = 180.0
    max_hold_bars = 60
    cooldown_bars = 1

    # RSI pullback thresholds
    rsi_pullback_low = 46
    rsi_recover_low = 48

    max_pullback_age = 15

    vol_avg_period = 20

    # EMA bounce parameters
    ema_bounce_pct = 1.5

    # Breakeven stop: move stop to entry once trade is up by this many ATRs
    breakeven_atr_trigger = 1.2

    # Momentum breakout: lookback for new high
    breakout_lookback = 20

    # --- Indicators ---
    df = df.copy()

    # EMAs for trend
    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()

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

    # ADX (Average Directional Index)
    plus_dm = df["high"].diff()
    minus_dm = -df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_smooth = tr.ewm(span=adx_period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    df["adx"] = dx.ewm(span=adx_period, adjust=False).mean().fillna(0)

    # Volume moving average for confirmation
    df["vol_ma"] = df["volume"].rolling(window=vol_avg_period, min_periods=1).mean()

    # Rolling high for breakout detection
    df["high_roll"] = df["high"].rolling(window=breakout_lookback, min_periods=breakout_lookback).max()

    # --- Trade loop ---
    trades = []
    position = None
    entry_price = None
    entry_idx = None
    stop_price = None
    best_price = None
    position_size = 0.0
    entry_atr = 0.0
    breakeven_activated = False
    warmup = max(slow_ema + 20, breakout_lookback + 5)

    # RSI pullback tracking
    long_pullback_ready = False
    long_pullback_bar = 0
    last_trade_exit = -999

    # Track EMA crossover for fresh cross entry signal
    prev_ema_f = None
    prev_ema_s = None

    for i in range(warmup, len(df) - 1):
        close = df["close"].iloc[i]
        rsi = df["rsi"].iloc[i]
        atr = df["atr"].iloc[i]
        ema_f = df["ema_fast"].iloc[i]
        ema_s = df["ema_slow"].iloc[i]
        adx = df["adx"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_ma = df["vol_ma"].iloc[i]

        uptrend = ema_f > ema_s
        strong_trend = adx > adx_threshold

        # Detect fresh EMA crossover (fast crosses above slow)
        fresh_cross = False
        if prev_ema_f is not None and prev_ema_s is not None:
            if prev_ema_f <= prev_ema_s and ema_f > ema_s:
                fresh_cross = True
        prev_ema_f = ema_f
        prev_ema_s = ema_s

        # Track RSI pullback states
        if rsi < rsi_pullback_low:
            long_pullback_ready = True
            long_pullback_bar = i
        if long_pullback_ready and (i - long_pullback_bar) > max_pullback_age:
            long_pullback_ready = False
        if not uptrend:
            long_pullback_ready = False

        # === EXIT LOGIC ===
        if position == "long":
            # Breakeven stop: once trade is up by breakeven_atr_trigger * ATR,
            # move stop to entry price to eliminate downside risk
            if not breakeven_activated and close >= entry_price + breakeven_atr_trigger * entry_atr:
                breakeven_activated = True
                if entry_price > stop_price:
                    stop_price = entry_price

            if close > best_price:
                best_price = close
                trail_stop = best_price - atr_trail_mult * atr
                if trail_stop > stop_price:
                    stop_price = trail_stop

            hit_stop = close <= stop_price
            time_exit = (i - entry_idx) >= max_hold_bars
            trend_exit = ema_f < ema_s

            if hit_stop or time_exit or trend_exit:
                if hit_stop:
                    exit_px = stop_price
                else:
                    exit_px = close
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": exit_px,
                    "direction": "long", "size": position_size,
                })
                last_trade_exit = i
                position = None

        # === ENTRY LOGIC (long only) ===
        if position is None:
            if (i - last_trade_exit) < cooldown_bars:
                continue

            ema_dist_pct = abs(close - ema_s) / ema_s * 100.0 if ema_s > 0 else 0
            not_overextended = ema_dist_pct < 15.0
            vol_ok = vol >= vol_ma * 0.7

            # Signal 1: RSI pullback recovery in uptrend
            pullback_signal = (uptrend and strong_trend and long_pullback_ready
                               and rsi > rsi_recover_low and rsi < 70
                               and not_overextended and vol_ok)

            # Signal 2: Fresh EMA crossover (catch trend starts)
            cross_signal = (fresh_cross and strong_trend
                            and rsi > 40 and rsi < 70
                            and not_overextended and vol_ok)

            # Signal 3: EMA bounce — price pulls back near fast EMA in uptrend
            ema_f_dist_pct = (close - ema_f) / ema_f * 100.0 if ema_f > 0 else 999
            bounce_signal = (uptrend and strong_trend
                             and 0 <= ema_f_dist_pct <= ema_bounce_pct
                             and rsi > 44 and rsi < 58
                             and not_overextended and vol_ok
                             and (i - last_trade_exit) >= 4)

            # Signal 4: Momentum breakout — price makes new N-bar high in uptrend
            prev_high_roll = df["high_roll"].iloc[i - 1] if i > 0 else 0
            breakout_signal = (uptrend and strong_trend
                               and close > prev_high_roll
                               and rsi > 50 and rsi < 72
                               and not_overextended and vol_ok
                               and (i - last_trade_exit) >= 3)

            if pullback_signal or cross_signal or bounce_signal or breakout_signal:
                stop_dist = atr_sl_mult * atr
                position_size = risk_per_trade / (stop_dist / close) if stop_dist > 0 else 0
                if position_size <= 0:
                    continue
                position = "long"
                entry_price = close
                entry_idx = i
                entry_atr = atr
                stop_price = close - stop_dist
                best_price = close
                breakeven_activated = False
                long_pullback_ready = False

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
    if train_m["num_trades"] < 20:
        score = FAIL_SCORE
        reason = f"too few trades ({train_m['num_trades']} < 20 required)"
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
