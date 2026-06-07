"""
Dixon-Coles backtest: train on 2018-2024, test on 2025-2026.
Reports accuracy, log-loss, Brier score, and calibration buckets.
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from db import get_conn, setup_db
from model_builder import build_and_train_model
from predict import predict_match

TRAIN_CUTOFF = "2024-12-31"
TEST_START   = "2025-01-01"


def _log_loss(p: float) -> float:
    return -math.log(max(p, 1e-9))


def run_backtest():
    setup_db()
    conn = get_conn()

    all_rows = conn.execute("""
        SELECT match_date, home_team, away_team, home_score, away_score, tournament, neutral
        FROM matches
        WHERE result NOT IN ('TBD', '')
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY match_date
    """).fetchall()
    conn.close()

    train_rows = [r for r in all_rows if r["match_date"] <= TRAIN_CUTOFF]
    test_rows  = [r for r in all_rows if r["match_date"] >= TEST_START]

    print(f"Training on {len(train_rows)} matches (up to {TRAIN_CUTOFF})")
    print(f"Testing on  {len(test_rows)} matches ({TEST_START}+)\n")

    # Train model on cutoff data
    bundle = build_and_train_model(cutoff_date=TRAIN_CUTOFF)

    # Evaluate
    total        = 0
    correct      = 0
    total_ll     = 0.0
    total_brier  = 0.0
    # Calibration: bucket model prob by decile and track hit rate
    buckets = {i: {"hits": 0, "n": 0} for i in range(10)}  # 0-9 = 0-10%, 10-20%, ...

    results_by_outcome = {"home": {"correct": 0, "n": 0},
                          "draw": {"correct": 0, "n": 0},
                          "away": {"correct": 0, "n": 0}}

    for r in test_rows:
        actual_h = int(r["home_score"])
        actual_a = int(r["away_score"])
        if actual_h > actual_a:
            actual_outcome = "home"
        elif actual_h == actual_a:
            actual_outcome = "draw"
        else:
            actual_outcome = "away"

        pred = predict_match(r["home_team"], r["away_team"], bundle,
                             neutral=bool(r["neutral"]))

        ph = pred["home_prob"]
        pd = pred["draw_prob"]
        pa = pred["away_prob"]

        # Accuracy: did top prediction match?
        probs = {"home": ph, "draw": pd, "away": pa}
        predicted = max(probs, key=probs.__getitem__)
        if predicted == actual_outcome:
            correct += 1

        results_by_outcome[actual_outcome]["n"] += 1
        if predicted == actual_outcome:
            results_by_outcome[actual_outcome]["correct"] += 1

        # Log-loss on correct outcome
        p_actual = probs[actual_outcome]
        total_ll    += _log_loss(p_actual)
        total_brier += (1 - p_actual) ** 2 + ph**2 + pd**2 + pa**2 - p_actual**2

        # Calibration: bucket by the winning outcome's model prob
        bucket_idx = min(int(p_actual * 10), 9)
        buckets[bucket_idx]["n"] += 1
        buckets[bucket_idx]["hits"] += 1  # by definition the actual outcome is always a hit

        total += 1

    if total == 0:
        print("No test matches found.")
        return

    acc      = correct / total
    avg_ll   = total_ll / total
    avg_brier = total_brier / total

    # Naive baseline: predict average (33.3% each) — gives log-loss = log(3) ≈ 1.099
    naive_ll = math.log(3)

    print("=" * 60)
    print(f"  Dixon-Coles Backtest  |  test window: {TEST_START} → 2026")
    print("=" * 60)
    print(f"\n  Matches tested:   {total}")
    print(f"  Accuracy:         {acc*100:.1f}%  (top-1 outcome correct)")
    print(f"  Log-loss:         {avg_ll:.4f}  (naive: {naive_ll:.4f})")
    print(f"  Brier score:      {avg_brier:.4f}  (lower is better)")
    print(f"  Log-loss gain:    {(naive_ll - avg_ll):.4f} vs naive\n")

    print("  By outcome:")
    for outcome, s in results_by_outcome.items():
        if s["n"] > 0:
            pct = s["correct"] / s["n"] * 100
            print(f"    {outcome.capitalize():6s} — predicted {s['correct']}/{s['n']} correctly ({pct:.1f}%)")

    # Calibration: bucket the assigned probability for the actual outcome
    print("\n  Calibration (assigned prob for actual outcome):")
    print(f"  {'Prob range':>12}  {'N':>5}  {'Avg assigned':>13}")
    for i in range(10):
        b = buckets[i]
        if b["n"] > 0:
            lo = i * 10
            hi = lo + 10
            # average assigned prob in this bucket — we need to track that separately
            print(f"  {lo:3d}–{hi:3d}%      {b['n']:>5}")

    # Better calibration: bucket by max(home, draw, away) prob decile
    print("\n  EV calibration: max-prob deciles")
    decile_buckets = {i: {"hits": 0, "n": 0, "sum_p": 0.0} for i in range(10)}
    for r in test_rows:
        actual_h = int(r["home_score"])
        actual_a = int(r["away_score"])
        if actual_h > actual_a:
            actual_outcome = "home"
        elif actual_h == actual_a:
            actual_outcome = "draw"
        else:
            actual_outcome = "away"
        pred = predict_match(r["home_team"], r["away_team"], bundle, neutral=bool(r["neutral"]))
        probs = {"home": pred["home_prob"], "draw": pred["draw_prob"], "away": pred["away_prob"]}
        top_outcome = max(probs, key=probs.__getitem__)
        top_p       = probs[top_outcome]
        idx = min(int(top_p * 10), 9)
        decile_buckets[idx]["n"]     += 1
        decile_buckets[idx]["sum_p"] += top_p
        if top_outcome == actual_outcome:
            decile_buckets[idx]["hits"] += 1

    print(f"  {'Prob range':>12}  {'N':>5}  {'Avg model%':>10}  {'Actual%':>8}")
    for i in range(10):
        b = decile_buckets[i]
        if b["n"] > 0:
            lo = i * 10
            hi = lo + 10
            avg_p  = b["sum_p"] / b["n"] * 100
            actual = b["hits"] / b["n"] * 100
            print(f"  {lo:3d}–{hi:3d}%      {b['n']:>5}  {avg_p:>9.1f}%  {actual:>7.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_backtest()
