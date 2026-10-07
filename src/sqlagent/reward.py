import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import sqlglot


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: list | None = None
    error: str | None = None
    columns: list | None = None


def execute_sql(db_path, sql: str, timeout_s: float = 5.0, max_rows: int = 10_000) -> ExecResult:
    """Run one SQL statement on a read-only connection with a time limit."""
    # Open database in read-only mode.
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = None
    try:
        conn = sqlite3.connect(uri, uri=True)

        # Some Spider databases contain non-UTF-8 text.
        # Replace non-UTF-8 text with invalid bytes.
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        
        # SQLite calls this every 10,000 VM steps.
        # Returning nonzero aborts the query.
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)

        # Cap the result size at max_rows.
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        if len(rows) > max_rows:
            return ExecResult(ok=False, error=f"Result exceeds {max_rows} rows")
        return ExecResult(ok=True, rows=rows, columns=columns)
    
    except Exception as e: # noqa: BLE001 - model-written SQL can raise anything; score as failure
        return ExecResult(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        if conn is not None:
            conn.close()


def _norm_value(v):
    # Round to 6 decimal places to avoid floating-point errors.
    return round(v, 6) if isinstance(v, float) else v


def _norm_rows(rows):
    return [tuple(_norm_value(v) for v in row) for row in rows]


def has_order_by(sql: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return "order by" in sql.lower()  # fall back to the crude check
    return tree.args.get("order") is not None


def results_match(pred_rows, gold_rows, ordered: bool) -> bool:
    # Compare two SQL results.
    pred, gold = _norm_rows(pred_rows), _norm_rows(gold_rows)
    if ordered:
        return pred == gold
    return Counter(pred) == Counter(gold)


def execution_reward(pred_sql: str, gold_sql: str, db_path, timeout_s: float = 5.0) -> float:
    """1.0 if the prediction's result matches the gold result, else 0.0."""
    gold = execute_sql(db_path, gold_sql, timeout_s)
    if not gold.ok:
        raise ValueError(f"Gold query failed: {gold.error}")
    pred = execute_sql(db_path, pred_sql, timeout_s)
    if not pred.ok:
        return 0.0
    return 1.0 if results_match(pred.rows, gold.rows, has_order_by(gold_sql)) else 0.0