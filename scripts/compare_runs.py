"""Compare two runs on the same questions with a paired test and a paired bootstrap."""

import argparse
import json
import math
import random
from collections import Counter

OUTCOME_ORDER = ["correct", "wrong_result", "submitted_sql_error", "turn_limit"]


def load(path: str) -> tuple[dict, Counter, int]:
    """Map (db_id, question, gold_sql) -> correct?, plus outcome counts and duplicate keys."""
    correct = {}
    outcomes = Counter()
    duplicates = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            key = (record["db_id"], record["question"], record["gold_sql"])
            duplicates += key in correct
            correct[key] = record["reward"] == 1.0
            outcomes[record.get("outcome", "unknown")] += 1
    return correct, outcomes, duplicates


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value: among b + c discordant pairs, is b ~ Binomial(b + c, 1/2)?"""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2**n)


def bootstrap_ci(diffs: list[int], iters: int, seed: int) -> tuple[float, float]:
    """95% percentile interval for the mean of per-question differences (B minus A)."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="JSONL from run_baseline.py (the reference run)")
    parser.add_argument("run_b", help="JSONL from run_baseline.py (the run being tested)")
    parser.add_argument("--bootstrap", type=int, default=10_000, help="Bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    a, outcomes_a, dup_a = load(args.run_a)
    b, outcomes_b, dup_b = load(args.run_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("No questions in common; were both runs on the same split and --seed?")
    if len(keys) != len(a) or len(keys) != len(b) or dup_a or dup_b:
        print(
            f"Warning: A has {len(a)} questions, B has {len(b)}, {len(keys)} in common; "
            f"duplicate keys: A {dup_a}, B {dup_b}. Comparing the common ones only.\n"
        )

    both = sum(a[k] and b[k] for k in keys)
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    neither = len(keys) - both - only_a - only_b
    n = len(keys)
    diffs = [int(b[k]) - int(a[k]) for k in keys]
    lo, hi = bootstrap_ci(diffs, args.bootstrap, args.seed)

    print(f"Questions compared: {n}")
    print(f"Accuracy A: {100 * (both + only_a) / n:.1f}%   B: {100 * (both + only_b) / n:.1f}%")
    print(f"Difference (B minus A): {100 * (only_b - only_a) / n:+.1f} points")
    print(f"Paired bootstrap 95% CI: [{100 * lo:+.1f}, {100 * hi:+.1f}] points\n")
    print("                  B correct   B wrong")
    print(f"  A correct       {both:9d}   {only_a:7d}")
    print(f"  A wrong         {only_b:9d}   {neither:7d}\n")
    print(f"Discordant questions: {only_a + only_b} (right only in A: {only_a}, only in B: {only_b})")
    print(f"McNemar exact two-sided p-value: {mcnemar_exact(only_b, only_a):.4f}\n")

    print("Outcome counts (all episodes in each file):")
    for name in OUTCOME_ORDER:
        print(f"  {name:<22} A {outcomes_a[name]:5d}   B {outcomes_b[name]:5d}")


if __name__ == "__main__":
    main()