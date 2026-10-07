"""Run a served model through the SQL environment and report execution accuracy."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from sqlagent.agent import Trajectory, parse_action
from sqlagent.data import load_split
from sqlagent.reward import execute_sql
from sqlagent.rollout import make_generate, run_many, trajectory_to_dict

OUTCOMES = [
    ("correct", "Correct"),
    ("wrong_result", "Submitted a query that ran but returned the wrong result"),
    ("submitted_sql_error", "Submitted a query that failed to execute"),
    ("turn_limit", "Never submitted (hit the turn limit)"),
]


def classify(traj: Trajectory) -> str:
    """Assign each episode to exactly one outcome category."""
    if traj.reward == 1.0:
        return "correct"
    if traj.truncated:
        return "turn_limit"
    # The episode ended by submission, so the last assistant message holds the submitted query.
    action = parse_action(traj.messages[-1]["content"]) or {}
    args = action.get("args")
    query = args.get("query") if isinstance(args, dict) else None
    if isinstance(query, str) and not execute_sql(traj.example.db_path, query).ok:
        return "submitted_sql_error"
    return "wrong_result"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def report(results: list[tuple[Trajectory, str]]) -> None:
    n = len(results)
    counts = Counter(outcome for _, outcome in results)
    correct = counts["correct"]
    lo, hi = wilson_interval(correct, n)

    turns_all = sum(t.turns for t, _ in results) / n
    correct_turns = [t.turns for t, o in results if o == "correct"]
    turns_correct = sum(correct_turns) / len(correct_turns) if correct_turns else float("nan")
    with_invalid = sum(t.invalid_actions > 0 for t, _ in results)
    with_sql_error = sum(t.sql_errors > 0 for t, _ in results)

    print(f"\nEpisodes: {n}")
    print(f"Execution accuracy: {100 * correct / n:.1f}%  (95% CI {100 * lo:.1f}-{100 * hi:.1f})")
    print(f"Mean turns: {turns_all:.2f} overall, {turns_correct:.2f} on correct episodes")
    print(f"Episodes with at least one invalid action: {100 * with_invalid / n:.1f}%")
    print(f"Episodes with at least one SQL error: {100 * with_sql_error / n:.1f}%")
    print("\nOutcomes:")
    for key, label in OUTCOMES:
        print(f"  {counts[key]:5d}  {100 * counts[key] / n:5.1f}%  {label}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model name exactly as served by vLLM")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--split", default="dev", choices=["dev", "train"])
    p.add_argument("--limit", type=int, default=None, help="Only run the first N examples")
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--workers", type=int, default=32, help="Concurrent episodes")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512, help="Max tokens per turn")
    p.add_argument("--out", default="outputs/baseline.jsonl", help="Where to save trajectories")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_split(args.split)
    if args.limit:
        examples = examples[: args.limit]

    generate = make_generate(
        args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    # Write each trajectory as it finishes, so a crash keeps the partial results.
    with open(out_path, "w", encoding="utf-8") as f:
        episodes = run_many(examples, generate, args.max_turns, args.workers)
        for traj in tqdm(episodes, total=len(examples)):
            outcome = classify(traj)
            results.append((traj, outcome))
            record = trajectory_to_dict(traj) | {"outcome": outcome}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nSaved {len(results)} trajectories to {out_path}")
    report(results)


if __name__ == "__main__":
    main()