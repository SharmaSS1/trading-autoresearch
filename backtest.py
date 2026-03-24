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

def load_data(path="data/etf_daily.csv", symbol="SPY"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Filter to one ticker
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol].drop(columns=["symbol"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Remove zero/NaN volume candles
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
            sharpe_ratio = (mean_r / std_r) * np.sqrt(252)  # 252 trading days/year for daily ETF data

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
    AGENT-MODIFIABLE: mean reversion starter on SPY daily bars.
    Entry: RSI(14) < 45 AND price > SMA(200) AND volume > 20-day avg volume.
    Exit: RSI > 65, trailing stop.
    Long-only. One position at a time.

    Change vs iter 18: Add volume confirmation filter.
    Requiring volume > 20-day average volume at entry filters out quiet drift-down
    days (low conviction) in favour of high-volume selloffs (panic/capitulation).
    Panic selling is the ideal mean-reversion setup: lots of forced sellers pushing
    price below fair value, followed by snap-back. Low-volume dips can continue
    indefinitely. This should improve win rate and profit factor without cutting
    trade count too severely (high-volume down days cluster with RSI dips).
    """
    trades = []
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # --- RSI(14) ---
    period = 14
    rsi = np.zeros(n)
    if n > period:
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, n - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rs = avg_gain / avg_loss if avg_loss > 1e-10 else 100.0
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    # --- SMA(200) trend filter ---
    sma_period = 200
    sma200 = np.zeros(n)
    for i in range(sma_period - 1, n):
        sma200[i] = np.mean(close[i - sma_period + 1 : i + 1])

    # --- 20-day average volume ---
    vol_period = 20
    vol_avg = np.zeros(n)
    for i in range(vol_period - 1, n):
        vol_avg[i] = np.mean(volume[i - vol_period + 1 : i + 1])

    # --- Strategy parameters ---
    rsi_entry = 45.0        # buy when RSI drops below this
    rsi_exit = 65.0         # sell when RSI recovers above this
    trail_pct = 0.02        # trailing stop: 2% below rolling intra-trade high
    cooldown_bars = 20      # wait 20 bars (~4 weeks) after a trailing-stop hit

    position = None  # None or dict with entry info
    cooldown_until = -1     # bar index after which we can re-enter
    trade_high = 0.0        # rolling high since entry (for trailing stop)

    for i in range(max(period + 1, sma_period), n):
        if position is None:
            # Entry: RSI oversold AND price above 200-day SMA AND volume spike AND not in cooldown
            in_uptrend = close[i] > sma200[i]
            high_volume = vol_avg[i] > 0 and volume[i] > vol_avg[i]
            if i > cooldown_until and rsi[i] > 0 and rsi[i] < rsi_entry and in_uptrend and high_volume:
                position = {
                    "entry_idx": i,
                    "entry_price": close[i],
                    "size": 10000.0,  # fixed dollar size
                }
                trade_high = close[i]
        else:
            # Update trailing high using today's high
            if high[i] > trade_high:
                trade_high = high[i]
            trailing_stop = trade_high * (1.0 - trail_pct)

            exit_price = None
            stopped_out = False
            # Exit 1: RSI recovered
            if rsi[i] > rsi_exit:
                exit_price = close[i]
            # Exit 2: trailing stop hit (use low of day)
            elif low[i] <= trailing_stop:
                exit_price = trailing_stop
                stopped_out = True

            if exit_price is not None:
                if stopped_out:
                    cooldown_until = i + cooldown_bars  # freeze entries after a stop-out

            if exit_price is not None:
                trades.append({
                    "entry_idx": position["entry_idx"],
                    "exit_idx": i,
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "direction": "long",
                    "size": position["size"],
                })
                position = None

    # Close any open position at end
    if position is not None:
        trades.append({
            "entry_idx": position["entry_idx"],
            "exit_idx": n - 1,
            "entry_price": position["entry_price"],
            "exit_price": close[n - 1],
            "direction": "long",
            "size": position["size"],
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
