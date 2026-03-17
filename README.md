# Trading AutoResearch

Automated trading strategy research, modeled on [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

**Core idea:** Human edits `program.md` (objectives), AI agent iterates on `backtest.py` (strategy code), `runner.py` orchestrates the loop — running backtest, measuring metrics, keeping improvements via git commits, discarding failures, repeating.

## Quick Start

```bash
pip install -r requirements.txt
python data_loader.py          # fetch BTC 4H data
python runner.py --dry-run     # verify pipeline works
python runner.py --max-iters 20  # run the full agent loop
```

## How It Works

1. `runner.py` calls Claude CLI with `program.md` context + current `backtest.py`
2. Agent proposes one focused change to the strategy
3. Runner executes `backtest.py`, parses the `SCORE:` output
4. If score improved: `git commit` the change
5. If score worse: `git checkout backtest.py` (revert)
6. Repeat

## Files

| File | Modified by | Purpose |
|------|------------|---------|
| `program.md` | Human | Objectives and constraints for the agent |
| `backtest.py` | Agent | BTC 4H backtesting strategy |
| `runner.py` | Nobody | Orchestration loop (fixed) |
| `data_loader.py` | Nobody | Fetches OHLCV data |

## Customization

Edit `program.md` to change what the agent optimizes for, what it can/can't modify, and any strategy constraints.

## Credits

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).
