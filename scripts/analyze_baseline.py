"""Break down a baseline run: tool-use patterns, failed submissions, and invalid actions."""

import argparse
import json
from collections import Counter

from sqlagent.agent import parse_action
from sqlagent.data import DEFAULT_SPIDER_DIR
from sqlagent.reward import execute_sql

SCHEMA_TOOLS = {"list_tables", "describe_table"}
ERROR_KINDS = [
    ("no such table", "unknown table"),
    ("no such column", "unknown column"),
    ("syntax error", "syntax error"),
    ("ambiguous column", "ambiguous column name"),
]


def tool_sequence(record: dict) -> list:
    """Tool names the model called, in order (None where the output couldn't be parsed)."""
    seq = []
    for message in record["messages"]:
        if message["role"] == "assistant":
            action = parse_action(message["content"])
            seq.append(action.get("tool") if action else None)
    return seq


def submitted_query(record: dict):
    """The SQL passed to submit_answer, or None if the episode never submitted."""
    last = next(m for m in reversed(record["messages"]) if m["role"] == "assistant")
    action = parse_action(last["content"]) or {}
    args = action.get("args")
    if action.get("tool") == "submit_answer" and isinstance(args, dict):
        return args.get("query")
    return None


def error_kind(error: str) -> str:
    low = error.lower()
    for needle, label in ERROR_KINDS:
        if needle in low:
            return label
    return "other"


def print_groups(title: str, groups: dict) -> None:
    print(f"\n{title}")
    for label, flags in groups.items():
        if flags:
            rate = 100 * sum(flags) / len(flags)
            print(f"  {label:<52} n={len(flags):5d}  accuracy={rate:5.1f}%")


def split_by(records: list, title: str, labels: tuple, predicate) -> None:
    """Print accuracy for episodes where predicate is true vs. false."""
    groups = {labels[0]: [], labels[1]: []}
    for r in records:
        groups[labels[0] if predicate(r) else labels[1]].append(r["outcome"] == "correct")
    print_groups(title, groups)


def normalize_sql(sql: str) -> str:
    """Collapse whitespace and drop a trailing semicolon.

    Letter case is left alone on purpose: string literals are case-sensitive in SQLite,
    so lowercasing could treat two different queries as the same.
    """
    return " ".join(sql.split()).rstrip(";").strip()


def run_sql_attempts(record: dict) -> list[tuple[str, bool]]:
    """(normalized query, ran without error) for each run_sql call, in order."""
    msgs = record["messages"]
    attempts = []
    for i, message in enumerate(msgs):
        if message["role"] != "assistant":
            continue
        action = parse_action(message["content"]) or {}
        args = action.get("args")
        if action.get("tool") != "run_sql" or not isinstance(args, dict):
            continue
        query = args.get("query")
        if not isinstance(query, str):
            continue
        observation = msgs[i + 1]["content"] if i + 1 < len(msgs) else ""
        failed = observation.startswith(("Observation:\nSQL error", "Observation:\nError:"))
        attempts.append((normalize_sql(query), not failed))
    return attempts


TESTING_LABELS = [
    "submitted a query it had already run successfully",
    "submitted a query that had already failed",
    "ran other queries, but submitted an untested one",
    "never ran run_sql",
]


def submission_testing(records: list) -> None:
    """Was the submitted query identical to one the model had already tried?"""
    groups = {label: [] for label in TESTING_LABELS}
    for r in records:
        query = submitted_query(r)
        if not isinstance(query, str):
            continue  # the episode never submitted
        submitted = normalize_sql(query)
        attempts = run_sql_attempts(r)
        if any(q == submitted and ok for q, ok in attempts):
            label = TESTING_LABELS[0]
        elif any(q == submitted for q, _ in attempts):
            label = TESTING_LABELS[1]
        elif attempts:
            label = TESTING_LABELS[2]
        else:
            label = TESTING_LABELS[3]
        groups[label].append(r["outcome"] == "correct")
    print_groups("Was the submitted query tested first? (episodes that submitted)", groups)


def failed_submission_causes(records: list) -> None:
    kinds = Counter()
    for r in records:
        if r["outcome"] != "submitted_sql_error":
            continue
        query = submitted_query(r)
        if query is None:
            kinds["no query found"] += 1
            continue
        db_path = DEFAULT_SPIDER_DIR / "database" / r["db_id"] / f"{r['db_id']}.sqlite"
        result = execute_sql(db_path, query, max_rows=1_000_000)
        kinds[error_kind(result.error) if not result.ok else "ran fine on re-check"] += 1
    print("\nWhy submitted queries failed to execute")
    for kind, count in kinds.most_common():
        print(f"  {count:5d}  {kind}")


def invalid_action_breakdown(records: list, n_samples: int) -> None:
    messages_seen = Counter()
    samples = []
    for r in records:
        msgs = r["messages"]
        for i, m in enumerate(msgs):
            if m["role"] == "user" and m["content"].startswith("Observation:\nError:"):
                messages_seen[m["content"].split("\n")[1][:80]] += 1
                if len(samples) < n_samples:
                    samples.append(msgs[i - 1]["content"][:300])
    print("\nError messages shown to the model after invalid actions")
    for line, count in messages_seen.most_common(8):
        print(f"  {count:5d}  {line}")
    print("\nRaw model outputs that triggered them")
    for sample in samples:
        print(" ", repr(sample))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="JSONL file written by run_baseline.py")
    parser.add_argument("--samples", type=int, default=5, help="Raw invalid outputs to print")
    args = parser.parse_args()

    with open(args.path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"{len(records)} episodes")

    split_by(
        records,
        "Schema exploration (list_tables / describe_table)",
        ("explored the schema", "did not explore the schema"),
        lambda r: bool(SCHEMA_TOOLS & set(tool_sequence(r))),
    )
    split_by(
        records,
        "Testing queries before submitting",
        ("ran run_sql at least once", "never ran run_sql"),
        lambda r: "run_sql" in tool_sequence(r),
    )
    split_by(
        records,
        "First action",
        ("submitted immediately", "did something else first"),
        lambda r: tool_sequence(r)[:1] == ["submit_answer"],
    )
    submission_testing(records)
    failed_submission_causes(records)
    invalid_action_breakdown(records, args.samples)


if __name__ == "__main__":
    main()