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
    Phase 4 long+short trend-following: EMA crossover + RSI pullback + EMA bounce + BB squeeze.

    Iter 67: Add time-based trailing stop decay — after a trade has been open for
    time_decay_start bars, progressively tighten the trail multiplier toward a minimum.
    This locks in profits on longer-running trades. Also tighten EMA floor buffer
    and slightly increase risk_pct since DD headroom exists (~2.7% vs 1% penalty).
    """
    # --- Parameters ---
    fast_ema = 21
    slow_ema = 45
    slow_ema2 = 80  # Higher timeframe trend filter
    rsi_period = 14
    atr_period = 14
    adx_period = 14
    adx_threshold = 15
    atr_sl_mult = 2.0
    atr_trail_mult = 1.6
    risk_pct = 0.016
    max_hold_bars = 80
    cooldown_bars = 0

    # RSI pullback thresholds (long) — widened to catch more setups
    rsi_pullback_low = 48
    rsi_recover_low = 50

    # RSI pullback thresholds (short — mirror)
    rsi_pullback_high = 52
    rsi_recover_high = 50

    max_pullback_age = 18

    vol_avg_period = 20

    # EMA bounce parameters
    ema_bounce_pct = 2.0

    # RSI-based profit-taking exit thresholds
    rsi_exit_long = 82  # Exit longs when very overbought (let winners run)
    rsi_exit_short = 18  # Exit shorts when very oversold (let winners run)

    # Breakeven stop: move stop to entry once trade is up by this many ATRs
    breakeven_atr_trigger = 1.3
    # Tighten trailing stop once trade is well in profit
    profit_accel_trigger = 2.5  # ATRs of profit to trigger tighter trail
    accel_trail_mult = 1.0  # Tighter trailing once in big profit

    # Momentum breakout: lookback for new high/low
    breakout_lookback = 20

    # ROC momentum confirmation
    roc_period = 10  # 10-bar rate of change
    roc_long_min = 0.0  # Price must be rising for longs
    roc_short_max = 0.0  # Price must be falling for shorts

    # Short-side risk is smaller (BTC has upward bias)
    short_risk_pct = 0.0072
    short_atr_sl_mult = 2.4
    short_atr_trail_mult = 1.9
    short_max_hold_bars = 40  # Allow shorts more room to develop
    short_adx_threshold = 17  # Slightly looser ADX for more short opportunities

    # Consecutive loss sizing reduction
    max_consec_losses = 3  # After N consecutive losses, reduce size
    loss_size_mult = 0.5  # Trade at 50% size after streak

    # EMA trailing stop floor: once trade is in profit, use fast EMA as stop floor
    ema_trail_floor_trigger = 0.8  # ATRs of profit before engaging EMA floor

    # Time-based trailing stop decay: after time_decay_start bars, tighten trail
    # multiplier linearly from current value toward time_decay_min_mult over
    # time_decay_duration bars
    time_decay_start = 20  # Bars before decay begins
    time_decay_duration = 25  # Bars over which to decay
    time_decay_min_mult = 0.7  # Minimum trail multiplier at full decay

    # --- Indicators ---
    df = df.copy()

    # EMAs for trend
    df["ema_fast"] = df["close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow_ema, adjust=False).mean()
    df["ema_slow2"] = df["close"].ewm(span=slow_ema2, adjust=False).mean()

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

    # Rolling high/low for breakout detection
    df["high_roll"] = df["high"].rolling(window=breakout_lookback, min_periods=breakout_lookback).max()
    df["low_roll"] = df["low"].rolling(window=breakout_lookback, min_periods=breakout_lookback).min()

    # Rate of Change (momentum confirmation)
    df["roc"] = df["close"].pct_change(periods=roc_period) * 100.0

    # EMA slope (rate of change of fast EMA over last 3 bars)
    df["ema_slope"] = df["ema_fast"].diff(3) / df["ema_fast"].shift(3) * 100.0
    df["ema_slope"] = df["ema_slope"].fillna(0.0)

    # ATR regime: ratio of current ATR to its moving average (for vol-adjusted sizing)
    atr_ma_period = 40
    df["atr_ma"] = df["atr"].rolling(window=atr_ma_period, min_periods=1).mean()
    df["atr_ratio"] = df["atr"] / df["atr_ma"].replace(0, np.nan)
    df["atr_ratio"] = df["atr_ratio"].fillna(1.0)

    # (EMA trail floor uses ema_fast computed above)

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
    short_pullback_ready = False
    short_pullback_bar = 0
    last_trade_exit = -999

    # Consecutive loss tracking for position sizing
    consec_losses = 0

    # Equity drawdown brake: track running equity to reduce size during drawdowns
    running_equity = 10000.0
    peak_equity = 10000.0
    dd_brake_threshold = 3.0  # Start reducing size at 3% drawdown
    dd_halt_threshold = 8.0  # Halt trading at 8% drawdown

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
        roc = df["roc"].iloc[i] if not np.isnan(df["roc"].iloc[i]) else 0.0
        ema_slope = df["ema_slope"].iloc[i] if not np.isnan(df["ema_slope"].iloc[i]) else 0.0

        ema_s2 = df["ema_slow2"].iloc[i]
        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s
        strong_trend = adx > adx_threshold
        # Higher timeframe trend filter
        macro_bullish = close > ema_s2
        macro_bearish = close < ema_s2

        # Detect fresh EMA crossovers
        fresh_cross_up = False
        fresh_cross_down = False
        if prev_ema_f is not None and prev_ema_s is not None:
            if prev_ema_f <= prev_ema_s and ema_f > ema_s:
                fresh_cross_up = True
            if prev_ema_f >= prev_ema_s and ema_f < ema_s:
                fresh_cross_down = True
        prev_ema_f = ema_f
        prev_ema_s = ema_s

        # Track RSI pullback states (long)
        if rsi < rsi_pullback_low:
            long_pullback_ready = True
            long_pullback_bar = i
        if long_pullback_ready and (i - long_pullback_bar) > max_pullback_age:
            long_pullback_ready = False
        if not uptrend:
            long_pullback_ready = False

        # Track RSI pullback states (short — RSI bounces high then comes back down)
        if rsi > rsi_pullback_high:
            short_pullback_ready = True
            short_pullback_bar = i
        if short_pullback_ready and (i - short_pullback_bar) > max_pullback_age:
            short_pullback_ready = False
        if not downtrend:
            short_pullback_ready = False

        # === EXIT LOGIC ===
        if position == "long":
            if not breakeven_activated and close >= entry_price + breakeven_atr_trigger * entry_atr:
                breakeven_activated = True
                if entry_price > stop_price:
                    stop_price = entry_price

            if close > best_price:
                best_price = close
            # Tighten trail once in significant profit
            profit_atrs = (best_price - entry_price) / entry_atr if entry_atr > 0 else 0
            curr_trail = accel_trail_mult if profit_atrs >= profit_accel_trigger else atr_trail_mult

            # Time-based decay: after time_decay_start bars, tighten trail
            bars_in_trade = i - entry_idx
            if bars_in_trade > time_decay_start:
                decay_progress = min(1.0, (bars_in_trade - time_decay_start) / time_decay_duration)
                curr_trail = curr_trail - (curr_trail - time_decay_min_mult) * decay_progress

            trail_stop = best_price - curr_trail * atr
            if trail_stop > stop_price:
                stop_price = trail_stop

            # EMA trail floor: once in profit, use fast EMA as stop floor
            profit_atrs_curr = (close - entry_price) / entry_atr if entry_atr > 0 else 0
            if profit_atrs_curr >= ema_trail_floor_trigger:
                ema_floor = ema_f - 0.15 * atr  # Tighter buffer below EMA
                if ema_floor > stop_price:
                    stop_price = ema_floor

            hit_stop = close <= stop_price
            time_exit = (i - entry_idx) >= max_hold_bars
            # Soft trend exit: only force exit on EMA cross if trade is losing AND been open ≥5 bars
            # Profitable trades are protected by trailing stop + EMA floor
            trend_exit = (ema_f < ema_s and close <= entry_price and (i - entry_idx) >= 5)

            if hit_stop or time_exit or trend_exit:
                exit_px = stop_price if hit_stop else close
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": exit_px,
                    "direction": "long", "size": position_size,
                })
                # Track consecutive losses for sizing
                pnl_long = (exit_px - entry_price) / entry_price * position_size
                running_equity += pnl_long
                if running_equity > peak_equity:
                    peak_equity = running_equity
                if exit_px < entry_price:
                    consec_losses += 1
                else:
                    consec_losses = 0
                last_trade_exit = i
                position = None

        elif position == "short":
            # Breakeven for shorts: price drops by breakeven_atr_trigger * ATR
            if not breakeven_activated and close <= entry_price - breakeven_atr_trigger * entry_atr:
                breakeven_activated = True
                if entry_price < stop_price:
                    stop_price = entry_price

            if close < best_price:
                best_price = close
            # Tighten trail once in significant profit
            profit_atrs = (entry_price - best_price) / entry_atr if entry_atr > 0 else 0
            curr_trail = accel_trail_mult if profit_atrs >= profit_accel_trigger else short_atr_trail_mult

            # Time-based decay for shorts
            bars_in_trade = i - entry_idx
            if bars_in_trade > time_decay_start:
                decay_progress = min(1.0, (bars_in_trade - time_decay_start) / time_decay_duration)
                curr_trail = curr_trail - (curr_trail - time_decay_min_mult) * decay_progress

            trail_stop = best_price + curr_trail * atr
            if trail_stop < stop_price:
                stop_price = trail_stop

            # EMA trail floor for shorts: use fast EMA as ceiling once in profit
            profit_atrs_curr = (entry_price - close) / entry_atr if entry_atr > 0 else 0
            if profit_atrs_curr >= ema_trail_floor_trigger:
                ema_ceil = ema_f + 0.15 * atr  # Tighter buffer above EMA
                if ema_ceil < stop_price:
                    stop_price = ema_ceil

            hit_stop = close >= stop_price
            time_exit = (i - entry_idx) >= short_max_hold_bars
            # Soft trend exit: only force exit on EMA cross if trade is losing AND been open ≥5 bars
            trend_exit = (ema_f > ema_s and close >= entry_price and (i - entry_idx) >= 5)

            if hit_stop or time_exit or trend_exit:
                exit_px = stop_price if hit_stop else close
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_price": entry_price, "exit_price": exit_px,
                    "direction": "short", "size": position_size,
                })
                # Track consecutive losses and equity for sizing
                pnl_short = (entry_price - exit_px) / entry_price * position_size
                running_equity += pnl_short
                if running_equity > peak_equity:
                    peak_equity = running_equity
                if exit_px > entry_price:
                    consec_losses += 1
                else:
                    consec_losses = 0
                last_trade_exit = i
                position = None

        # === ENTRY LOGIC ===
        if position is None:
            if (i - last_trade_exit) < cooldown_bars:
                continue

            # Equity drawdown brake: reduce/halt trading during drawdowns
            equity_dd_pct = (peak_equity - running_equity) / peak_equity * 100.0 if peak_equity > 0 else 0
            if equity_dd_pct >= dd_halt_threshold:
                continue  # Stop trading during severe drawdown
            dd_size_mult = 1.0
            if equity_dd_pct >= dd_brake_threshold:
                # Linear reduction: at dd_brake=2%, mult=1.0; at dd_halt=5%, mult=0.3
                dd_size_mult = max(0.3, 1.0 - 0.7 * (equity_dd_pct - dd_brake_threshold) / (dd_halt_threshold - dd_brake_threshold))

            ema_dist_pct = abs(close - ema_s) / ema_s * 100.0 if ema_s > 0 else 0
            not_overextended = ema_dist_pct < 15.0
            vol_ok = vol >= vol_ma * 0.7

            # Momentum confirmation via ROC
            roc_long_ok = roc >= roc_long_min
            roc_short_ok = roc <= roc_short_max

            # Size multiplier for consecutive losses
            size_mult = loss_size_mult if consec_losses >= max_consec_losses else 1.0

            # Volatility-adjusted sizing: always scale inversely with ATR ratio
            # Low vol → bigger size (more return), high vol → smaller size (less DD)
            atr_ratio = df["atr_ratio"].iloc[i] if not np.isnan(df["atr_ratio"].iloc[i]) else 1.0
            atr_ratio_clamped = max(0.6, min(atr_ratio, 2.0))  # Clamp to avoid extremes
            size_mult *= 1.0 / atr_ratio_clamped

            # Apply drawdown brake
            size_mult *= dd_size_mult

            # ---- LONG SIGNALS ----
            # Signal 1: RSI pullback recovery in uptrend
            pullback_signal = (uptrend and strong_trend and macro_bullish
                               and long_pullback_ready
                               and rsi > rsi_recover_low and rsi < 72
                               and not_overextended and vol_ok and roc_long_ok)

            # Signal 2: Fresh EMA crossover up
            cross_signal = (fresh_cross_up and strong_trend
                            and rsi > 38 and rsi < 72
                            and not_overextended and vol_ok)

            # Signal 3: EMA bounce — price pulls back near fast EMA in uptrend
            ema_f_dist_pct = (close - ema_f) / ema_f * 100.0 if ema_f > 0 else 999
            bounce_signal = (uptrend and strong_trend and macro_bullish
                             and 0 <= ema_f_dist_pct <= ema_bounce_pct
                             and rsi > 42 and rsi < 62
                             and not_overextended and vol_ok and roc_long_ok
                             and (i - last_trade_exit) >= 3)

            # Signal 4: Momentum breakout — price makes new N-bar high in uptrend
            prev_high_roll = df["high_roll"].iloc[i - 1] if i > 0 else 0
            breakout_signal = (uptrend and strong_trend and macro_bullish
                               and close > prev_high_roll
                               and rsi > 48 and rsi < 74
                               and not_overextended and vol_ok
                               and (i - last_trade_exit) >= 3)

            if pullback_signal or cross_signal or bounce_signal or breakout_signal:
                stop_dist = atr_sl_mult * atr
                curr_risk = running_equity * risk_pct
                position_size = curr_risk * size_mult / (stop_dist / close) if stop_dist > 0 else 0
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
                continue

            # ---- SHORT SIGNALS ----
            short_strong_trend = adx > short_adx_threshold
            # Signal S1: RSI pullback recovery in downtrend (RSI was high, came back)
            short_pullback_signal = (downtrend and short_strong_trend and macro_bearish
                                     and short_pullback_ready
                                     and rsi < rsi_recover_high and rsi > 28
                                     and not_overextended and vol_ok and roc_short_ok)

            # Signal S2: Fresh EMA crossover down
            short_cross_signal = (fresh_cross_down and short_strong_trend and macro_bearish
                                  and rsi < 62 and rsi > 28
                                  and not_overextended and vol_ok and roc_short_ok)

            # Signal S3: EMA bounce short — price rallies back near fast EMA in downtrend
            ema_f_dist_short = (ema_f - close) / ema_f * 100.0 if ema_f > 0 else 999
            short_bounce_signal = (downtrend and short_strong_trend and macro_bearish
                                   and 0 <= ema_f_dist_short <= ema_bounce_pct
                                   and rsi < 58 and rsi > 40
                                   and not_overextended and vol_ok and roc_short_ok
                                   and (i - last_trade_exit) >= 3)

            # Signal S4: Breakdown — price makes new N-bar low in downtrend
            prev_low_roll = df["low_roll"].iloc[i - 1] if i > 0 else float('inf')
            short_breakout_signal = (downtrend and short_strong_trend and macro_bearish
                                     and close < prev_low_roll
                                     and rsi < 52 and rsi > 26
                                     and not_overextended and vol_ok and roc_short_ok
                                     and (i - last_trade_exit) >= 3)

            if short_pullback_signal or short_cross_signal or short_bounce_signal or short_breakout_signal:
                stop_dist = short_atr_sl_mult * atr
                curr_risk = running_equity * short_risk_pct
                position_size = curr_risk * size_mult / (stop_dist / close) if stop_dist > 0 else 0
                if position_size <= 0:
                    continue
                position = "short"
                entry_price = close
                entry_idx = i
                entry_atr = atr
                stop_price = close + stop_dist
                best_price = close
                breakeven_activated = False
                short_pullback_ready = False

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
