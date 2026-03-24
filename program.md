# Trading AutoResearch — ETF Mean Reversion Program

## Objective
Find a mean-reversion strategy on liquid US ETFs (SPY, QQQ, IWM) using daily bars.
Target: Sharpe 1.5–2.5 on out-of-sample data.

WARNING: Crypto Sharpe targets (10–20+) are NOT the goal. ETF Sharpe > 3.0 is suspicious overfit.
Sharpe 1.5–2.5 is genuinely excellent for a daily ETF strategy.

## Strategy Type: Mean Reversion
Mean reversion = price has drifted too far from its average and will snap back.
Classic signals: RSI oversold, price below Bollinger Band, large gap down, etc.

Direction: long-only (we do not short ETFs — no margin, no complexity).
Use bracket orders (stop-loss + take-profit) where possible.

## Instruments
- SPY (S&P 500 ETF) — highest liquidity, tightest spread
- QQQ (Nasdaq 100 ETF) — higher volatility, more mean-reversion opportunities
- IWM (Russell 2000 ETF) — smaller caps, more volatility

The agent focuses on ONE ticker at a time (start with SPY).

## Data
- Daily OHLCV bars via yfinance (10 years history)
- File: data/etf_daily.csv, column `symbol` = SPY/QQQ/IWM
- Load SPY rows: `df[df['symbol'] == 'SPY']`
- Train: first 80% of bars. Holdout: last 20% (never used during optimization)

## Scoring Formula
score = 0.6 * train_composite + 0.4 * holdout_composite

composite_score = sharpe_ratio (capped at 3.0)
               - 0.5 * max_drawdown_pct
               + 0.5 * ln(min(profit_factor, 4.0))
               + 0.05 * min(total_return_pct, 100)
               - win_rate_penalty
               - too_clean_penalty

Reality-check penalties (same as existing compute_metrics — DO NOT change compute_metrics):
  - win_rate > 70%: penalty = (win_rate - 70) * 0.5
  - win_rate < 35%: penalty = (35 - win_rate) * 0.3
  - max_drawdown < 1%: penalty = (1.0 - max_dd) * 5.0

Hard gates (score = -100 if violated):
  - Minimum 20 trades on training data
  - Minimum 1% total return
  - Profit factor >= 1.0

## ETF-Calibrated Target Performance
- Sharpe 1.5–2.5 (excellent and realistic for daily ETF)
- Max drawdown 5–20% (some drawdown = strategy is actually trading)
- Win rate 45–65%
- Profit factor 1.3–2.5
- 30–150 trades per year (not too few = overfit, not too many = noise)
- Score target: 2.0–5.0 (ETF range, NOT BTC range of 8–15)

## Annualization
Daily data: annualization factor = 252 trading days/year (NOT 2190, which was for BTC 4H bars).
This is already set in the FIXED compute_metrics section.

## What the Agent CAN Change (run_strategy only)
- Entry and exit signal logic
- Technical indicators and their parameters
- Stop-loss and take-profit levels
- Position sizing (long-only, max 1 position at a time)
- Trade filters (volatility, trend confirmation, time filters)
- Which ETF ticker to use (SPY recommended to start)

## What the Agent Must NOT Change
- load_data() function
- compute_metrics() function
- The main block (train/test split + scoring logic)
- The SCORE: <float> output format (must be last line of stdout)
- Must not introduce look-ahead bias (no future data in signals)

## No Look-Ahead Bias
At candle index i, only use data from indices 0..i.
Never use shift(-1) on close prices or indicator values.

## Change ONE Variable at a Time (Karpathy Rule)
If you try multiple changes at once and the score drops, you won't know what caused it.
Change exactly one thing per iteration: one indicator, one threshold, one structural idea.

## Good Starting Ideas for Mean Reversion on SPY Daily
- RSI(14) < 30 → buy; RSI > 70 → sell
- Price closes below lower Bollinger Band → buy; price returns to middle band → sell
- 3-day consecutive decline of >1% each day → buy; 3% gain → sell
- ATR-based stop loss at entry_price - 2 * ATR(14)

## Bad Ideas to Avoid
- Shorting (we are long-only)
- Leverage or margin
- Intraday signals (we use daily close prices)
- Look-ahead bias (no future data)
- Rewriting the entire file (change ONE thing)

## Score History Interpretation
Score 0–1.5: strategy is not working yet (too few trades, or losing money)
Score 1.5–2.5: good! This is the target range. Paper-trade this config.
Score 2.5–3.5: excellent. Verify out-of-sample, then consider going live.
Score > 3.5: suspicious. Check for overfit. Walk-forward test before trusting.
