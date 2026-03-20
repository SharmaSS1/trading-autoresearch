#!/usr/bin/env python3
"""
Trading AutoResearch — Runner (Orchestration Loop)
===================================================
Modeled on Karpathy's autoresearch runner.
DO NOT modify this file — the agent only modifies backtest.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import re

STATE_FILE = "state.json"
BACKTEST_SCRIPT = "backtest.py"
PROGRAM_FILE = "program.md"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"best_score": float("-inf"), "iteration": 0, "run_history": []}


def save_state(state):
    # Convert -inf/inf for JSON
    s = state.copy()
    if s["best_score"] == float("-inf"):
        s["best_score"] = None
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


def run_backtest(timeout=120):
    """Run backtest.py and return (score, stdout, success)."""
    try:
        result = subprocess.run(
            [sys.executable, BACKTEST_SCRIPT],
            capture_output=True, text=True, timeout=timeout
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            print(f"  backtest.py failed (rc={result.returncode})")
            if stderr:
                print(f"  stderr: {stderr[:500]}")
            return None, stdout, False

        # Parse SCORE from last line
        lines = stdout.strip().split("\n")
        last_line = lines[-1] if lines else ""
        match = re.match(r"SCORE:\s*([-\d.eE+]+)", last_line)
        if not match:
            print(f"  Could not parse SCORE from last line: {last_line!r}")
            return None, stdout, False

        score = float(match.group(1))
        return score, stdout, True

    except subprocess.TimeoutExpired:
        print("  backtest.py timed out (>120s)")
        return None, "", False
    except Exception as e:
        print(f"  Error running backtest: {e}")
        return None, "", False


def git_commit_improvement(old_score, new_score, iteration):
    """Stage backtest.py + state.json, commit, and push so MTOI dashboard stays live."""
    subprocess.run(["git", "add", BACKTEST_SCRIPT, STATE_FILE], check=True,
                    capture_output=True)
    old_str = f"{old_score:.4f}" if old_score is not None and old_score != float("-inf") else "none"
    msg = f"improvement: score {old_str} -> {new_score:.4f} [iter {iteration}]"
    subprocess.run(["git", "commit", "-m", msg], check=True,
                    capture_output=True)
    # Push to GitHub so MTOI dashboard can read live state
    try:
        subprocess.run(["git", "push"], check=True, capture_output=True, timeout=30)
        print("  -> pushed to GitHub")
    except Exception as e:
        print(f"  -> push failed (non-fatal): {e}")
    return msg


def git_revert_backtest():
    """Discard changes to backtest.py."""
    subprocess.run(["git", "checkout", BACKTEST_SCRIPT], check=True,
                    capture_output=True)


def call_agent(dry_run=False):
    """Call Claude CLI to propose the next change to backtest.py."""
    if dry_run:
        print("  [dry-run] Skipping agent call")
        return True

    # Read current files for context
    with open(PROGRAM_FILE) as f:
        program = f.read()
    with open(BACKTEST_SCRIPT) as f:
        backtest = f.read()

    state = load_state()
    history_summary = ""
    if state["run_history"]:
        recent = state["run_history"][-5:]
        history_summary = "\n".join(
            f"  iter {h['iteration']}: score={h['score']} ({h['result']})"
            for h in recent
        )

    prompt = f"""You are an AI trading strategy researcher. Your task is to improve the strategy in backtest.py.

## Objectives (from program.md):
{program}

## Current backtest.py:
```python
{backtest}
```

## Current best score: {state['best_score']}
## Iteration: {state['iteration'] + 1}
## Recent history:
{history_summary if history_summary else "  (no history yet)"}

## Instructions:
1. Analyze the current strategy
2. Propose ONE focused improvement (not a full rewrite)
3. Edit backtest.py in place with your change
4. Only modify the AGENT-MODIFIABLE section
5. Do NOT change load_data(), compute_metrics(), or the SCORE output
6. Do NOT introduce look-ahead bias

Make your change now by editing backtest.py directly."""

    # Ensure node is in PATH (required for claude CLI)
    import copy
    env = copy.copy(os.environ)
    node_bin = "/opt/homebrew/opt/node@22/bin"
    if node_bin not in env.get("PATH", ""):
        env["PATH"] = node_bin + ":" + env.get("PATH", "")

    try:
        result = subprocess.run(
            ["claude", "--print", "-p", prompt, "--allowedTools",
             "Read", "Edit", "Write", "Bash"],
            capture_output=True, text=True, timeout=200, env=env
        )
        if result.returncode != 0:
            print(f"  Agent call failed (rc={result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:300]}")
            return False
        return True
    except FileNotFoundError:
        print("  'claude' CLI not found — install Claude Code to enable agent loop")
        return False
    except subprocess.TimeoutExpired:
        print("  Agent call timed out (>120s)")
        return False
    except Exception as e:
        print(f"  Agent error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Trading AutoResearch Runner")
    parser.add_argument("--max-iters", type=int, default=10,
                        help="Maximum iterations (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run backtest once without agent, don't commit")
    args = parser.parse_args()

    state = load_state()
    best_score = state["best_score"]
    if best_score is None:
        best_score = float("-inf")
    start_iter = state["iteration"]

    print(f"=== Trading AutoResearch Runner ===")
    print(f"Starting at iteration {start_iter}, best_score={best_score}")
    print(f"Max iterations: {args.max_iters}")
    if args.dry_run:
        print("Mode: DRY RUN (single evaluation, no agent, no commits)\n")
    else:
        print()

    for i in range(args.max_iters):
        iteration = start_iter + i + 1
        # Pull latest program.md (allows remote edits from MTOI dashboard)
        try:
            subprocess.run(["git", "pull", "--quiet"], capture_output=True, timeout=15)
        except Exception:
            pass  # non-fatal, continue with local version
        print(f"[iter {iteration}]", end=" ")

        if not args.dry_run:
            # Call agent to modify backtest.py
            print("Calling agent...", end=" ")
            if not call_agent(dry_run=False):
                print("Agent failed, skipping iteration")
                time.sleep(2)
                continue

        # Run backtest
        score, stdout, success = run_backtest()

        if not success or score is None:
            print(f"score=FAIL best={best_score}")
            if not args.dry_run:
                git_revert_backtest()
                print("  -> reverted")
            state["run_history"].append({
                "iteration": iteration, "score": None, "result": "fail"
            })
            state["iteration"] = iteration
            save_state(state)
            if args.dry_run:
                break
            time.sleep(2)
            continue

        improved = score > best_score
        result_str = "kept" if improved else "reverted"
        print(f"score={score:.4f} best={best_score:.4f} ({result_str})")

        if stdout:
            for line in stdout.split("\n"):
                if not line.startswith("SCORE:"):
                    print(f"  {line}")

        if improved and not args.dry_run:
            old_best = best_score
            best_score = score
            commit_msg = git_commit_improvement(old_best, score, iteration)
            print(f"  -> committed: {commit_msg}")
        elif improved and args.dry_run:
            best_score = score
            print("  -> [dry-run] would commit")
        elif not improved and not args.dry_run:
            git_revert_backtest()
            print("  -> reverted backtest.py")

        state["best_score"] = best_score
        state["iteration"] = iteration
        state["run_history"].append({
            "iteration": iteration,
            "score": round(score, 4),
            "result": "kept" if improved else "reverted",
        })
        save_state(state)

        if args.dry_run:
            break

        time.sleep(2)

    print(f"\n=== Done. Best score: {best_score} after {state['iteration']} iterations ===")


if __name__ == "__main__":
    main()
