# Trading AutoResearch — Program Objectives

## Goal
Maximize `composite_score` on BTC 4H OHLCV data.

```
composite_score = sharpe_ratio - 0.5 * max_drawdown_pct
```

Higher is better. Sharpe ratio rewards consistent returns; the drawdown penalty discourages reckless risk.

## What the Agent CAN Change (in backtest.py)
- Entry and exit logic (signals, conditions, timing)
- Technical indicators and their parameters (EMA periods, RSI thresholds, etc.)
- Adding new indicators (Bollinger Bands, MACD, ATR, etc.)
- Stop-loss and take-profit levels
- Position sizing rules
- Trade filters (volume, volatility, time-of-day, etc.)
- Combining multiple signals or adding confirmation logic

## What the Agent Must NOT Change
- Data loading section (reads from `data/btc_4h.csv`)
- The `compute_metrics(trades, df)` function
- The `SCORE: <float>` output format (must be the last line of stdout)
- The CSV column format: timestamp, open, high, low, close, volume
- Must not introduce look-ahead bias (no peeking at future candles)

## No Look-Ahead Bias Rule
At any candle index `i`, the strategy may only use data from indices `0..i` (inclusive).
Do NOT use future close prices, future indicator values, or shift signals backward in time.

## Agent Workflow
1. Read this file (`program.md`) and current `backtest.py`
2. Analyze the current strategy and recent score
3. Propose ONE focused change (not a full rewrite)
4. Edit `backtest.py` in place
5. The runner will execute and evaluate the change

## Example Good Changes
- "Widen RSI filter from 30/70 to 25/75 to catch stronger momentum"
- "Add ATR-based stop loss instead of fixed percentage"
- "Add volume confirmation: only enter when volume > 1.5x 20-period average"
- "Switch from EMA 12/26 to EMA 9/21 for faster signals"

## Example Bad Changes
- Rewriting the entire file from scratch
- Changing the metric formula
- Using `df['close'].shift(-1)` (look-ahead bias)
- Removing the SCORE output line
- Adding external data sources or API calls
