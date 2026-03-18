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
    Trend-following with RSI pullback entries + ATR trailing stop + ADX filter.

    Strategy:
      - EMA 21/55 defines trend direction
      - ADX > 20 confirms trend strength (filters choppy markets)
      - Enter long on RSI pullback (RSI dips below 42 then recovers above 46) in uptrend
      - Enter short on RSI pullback (RSI rises above 58 then drops below 54) in downtrend
      - ATR-based stop loss with trailing mechanism
      - Tighter trailing stop (1.8x ATR) to lock in profits earlier
      - Max hold time of 30 bars to prevent extended drawdown periods
    """
    # --- Parameters ---
    fast_ema = 21
    slow_ema = 55
    rsi_period = 14
    atr_period = 14
    adx_period = 14
    adx_threshold = 25     # minimum ADX to confirm trend (stricter)
    atr_sl_mult = 1.1      # stop loss = 1.1x ATR
    atr_tp_mult = 4.0      # take profit = 4.0x ATR
    atr_trail_mult = 1.3   # trailing stop distance
    atr_trail_tight = 0.9  # tighter trail once trade is well in profit
    trail_tighten_threshold = 1.8  # tighten trail after price moves 1.8x ATR in favor
    position_size = 263.0
    max_hold_bars = 30      # max bars to hold a position
    breakeven_atr_mult = 0.52  # move stop to entry after price moves 0.52x ATR in favor
    vol_period = 20         # volume moving average period
    vol_mult = 0.5          # volume must be >= 0.5x average (allow more trades)
    # RSI thresholds for pullback detection
    rsi_pullback_low = 41
    rsi_recover_low = 46
    rsi_pullback_high = 58
    rsi_recover_high = 54

    # Bollinger Band for volatility regime filter
    bb_period = 20
    bb_std = 2.0


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

    # Volume moving average for confirmation
    df["vol_ma"] = df["volume"].rolling(window=vol_period).mean()

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
    df["plus_di"] = plus_di.fillna(0)
    df["minus_di"] = minus_di.fillna(0)

    # Bollinger Bands for volatility width
    df["bb_mid"] = df["close"].rolling(window=bb_period).mean()
    df["bb_std"] = df["close"].rolling(window=bb_period).std()
    df["bb_upper"] = df["bb_mid"] + bb_std * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - bb_std * df["bb_std"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
    df["bb_width_ma"] = df["bb_width"].rolling(window=bb_period).mean()


    # --- Trade loop ---
    trades = []
    position = None
    entry_price = None
    entry_idx = None
    stop_price = None
    tp_price = None
    best_price = None
    current_size = position_size
    warmup = slow_ema + 20

    # Track RSI pullback state
    long_pullback_ready = False
    short_pullback_ready = False
    last_trade_exit = -999
    last_trade_won = True
    cooldown_bars = 3  # bars to wait after a loss before re-entering

    for i in range(warmup, len(df) - 1):
        close = df["close"].iloc[i]
        rsi = df["rsi"].iloc[i]
        atr = df["atr"].iloc[i]
        ema_f = df["ema_fast"].iloc[i]
        ema_s = df["ema_slow"].iloc[i]
        adx = df["adx"].iloc[i]
        vol = df["volume"].iloc[i]
        vol_avg = df["vol_ma"].iloc[i]

        bb_width = df["bb_width"].iloc[i]
        bb_width_avg = df["bb_width_ma"].iloc[i]
        pdi = df["plus_di"].iloc[i]
        mdi = df["minus_di"].iloc[i]
        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s
        strong_trend = adx > adx_threshold
        volume_confirmed = vol >= vol_mult * vol_avg if pd.notna(vol_avg) else False
        # Volatility expanding = trending market (good for trend-following)
        vol_expanding = bb_width > bb_width_avg if pd.notna(bb_width_avg) and pd.notna(bb_width) else True
        # DI alignment: confirms directional momentum
        di_long = pdi > mdi
        di_short = mdi > pdi

        # Track RSI pullback states
        if rsi < rsi_pullback_low:
            long_pullback_ready = True
        if rsi > rsi_pullback_high:
            short_pullback_ready = True

        # Reset pullback flags if trend reverses
        if not uptrend:
            long_pullback_ready = False
        if not downtrend:
            short_pullback_ready = False

        # Exit logic with trailing stop + break-even + EMA cross exit
        if position == "long":
            if close > best_price:
                best_price = close
                # Use tighter trail once trade is well in profit
                if best_price >= entry_price + trail_tighten_threshold * atr:
                    trail_stop = best_price - atr_trail_tight * atr
                else:
                    trail_stop = best_price - atr_trail_mult * atr
                if trail_stop > stop_price:
                    stop_price = trail_stop

            # Break-even: once price moves 1.2x ATR in favor, move stop to entry
            if best_price >= entry_price + breakeven_atr_mult * atr:
                be_stop = entry_price
                if be_stop > stop_price:
                    stop_price = be_stop

            hit_stop = close <= stop_price
            hit_tp = close >= tp_price
            time_exit = (i - entry_idx) >= max_hold_bars
            # Exit if trend reverses (EMA cross)
            trend_exit = ema_f < ema_s and (i - entry_idx) >= 3
            if hit_stop or hit_tp or time_exit or trend_exit:
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": close,
                    "direction": "long", "size": current_size,
                })
                last_trade_won = close > entry_price
                last_trade_exit = i
                position = None

        elif position == "short":
            if close < best_price:
                best_price = close
                # Use tighter trail once trade is well in profit
                if best_price <= entry_price - trail_tighten_threshold * atr:
                    trail_stop = best_price + atr_trail_tight * atr
                else:
                    trail_stop = best_price + atr_trail_mult * atr
                if trail_stop < stop_price:
                    stop_price = trail_stop

            # Break-even: once price moves 1.2x ATR in favor, move stop to entry
            if best_price <= entry_price - breakeven_atr_mult * atr:
                be_stop = entry_price
                if be_stop < stop_price:
                    stop_price = be_stop

            hit_stop = close >= stop_price
            hit_tp = close <= tp_price
            time_exit = (i - entry_idx) >= max_hold_bars
            # Exit if trend reverses (EMA cross)
            trend_exit = ema_f > ema_s and (i - entry_idx) >= 3
            if hit_stop or hit_tp or time_exit or trend_exit:
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": close,
                    "direction": "short", "size": current_size,
                })
                last_trade_won = close < entry_price
                last_trade_exit = i
                position = None

        # Entry logic: RSI pullback within trend + volatility expansion
        if position is None:
            # Cooldown after a losing trade: wait before re-entering
            in_cooldown = (not last_trade_won) and (i - last_trade_exit < cooldown_bars)

            # Dynamic position sizing: scale with ADX strength
            # ADX 25 -> 0.7x size, ADX 40+ -> 1.0x size
            adx_scale = min(1.0, 0.7 + 0.3 * (adx - adx_threshold) / 15.0) if adx > adx_threshold else 0.7
            # Reduce size after a loss to limit consecutive-loss drawdown
            loss_scale = 0.7 if not last_trade_won and (i - last_trade_exit < 130) else 1.0
            trade_size = position_size * adx_scale * loss_scale

            if not in_cooldown and uptrend and strong_trend and volume_confirmed and long_pullback_ready and rsi > rsi_recover_low and rsi < 70:
                position = "long"
                entry_price = close
                entry_idx = i
                stop_price = close - atr_sl_mult * atr
                tp_price = close + atr_tp_mult * atr
                best_price = close
                long_pullback_ready = False
                current_size = trade_size

            elif not in_cooldown and downtrend and strong_trend and volume_confirmed and short_pullback_ready and rsi < rsi_recover_high and rsi > 30:
                position = "short"
                entry_price = close
                entry_idx = i
                stop_price = close + atr_sl_mult * atr
                tp_price = close - atr_tp_mult * atr
                best_price = close
                short_pullback_ready = False
                current_size = trade_size

    # Close open position at end
    if position is not None:
        trades.append({
            "entry_idx": entry_idx, "exit_idx": len(df) - 1,
            "entry_price": entry_price, "exit_price": df["close"].iloc[-1],
            "direction": position, "size": current_size,
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
