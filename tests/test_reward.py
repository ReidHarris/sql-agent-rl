import sqlite3
import time

import pytest

from sqlagent.reward import execute_sql, execution_reward


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE employees (id INTEGER, name TEXT, dept TEXT, salary REAL);
        INSERT INTO employees VALUES
            (1, 'Ana', 'eng', 90000),
            (2, 'Bo',  'eng', 70000),
            (3, 'Cy',  'ops', 50000),
            (4, 'Di',  'ops', 40000);
        """
    )
    conn.commit()
    conn.close()
    return path


def reward(db, pred, gold, **kw):
    return execution_reward(pred, gold, db, **kw)


# --- should match ---

def test_identical_query(db):
    q = "SELECT name FROM employees"
    assert reward(db, q, q) == 1.0


def test_different_route_same_rows(db):
    gold = "SELECT name FROM employees WHERE salary > 60000"
    pred = "SELECT name FROM employees WHERE NOT salary <= 60000"
    assert reward(db, pred, gold) == 1.0


def test_row_order_ignored_without_order_by(db):
    gold = "SELECT name FROM employees"
    pred = "SELECT name FROM employees ORDER BY name DESC"
    assert reward(db, pred, gold) == 1.0


def test_float_rounding(db):
    assert reward(db, "SELECT 0.1 + 0.2", "SELECT 0.3") == 1.0


# --- should not match ---

def test_order_matters_with_order_by(db):
    gold = "SELECT name FROM employees ORDER BY salary DESC"
    pred = "SELECT name FROM employees ORDER BY salary ASC"
    assert reward(db, pred, gold) == 0.0


def test_duplicates_matter(db):
    gold = "SELECT dept FROM employees"
    pred = "SELECT DISTINCT dept FROM employees"
    assert reward(db, pred, gold) == 0.0


def test_column_order_matters(db):
    gold = "SELECT dept, name FROM employees"
    pred = "SELECT name, dept FROM employees"
    assert reward(db, pred, gold) == 0.0


def test_wrong_rows(db):
    gold = "SELECT name FROM employees WHERE dept = 'eng'"
    pred = "SELECT name FROM employees WHERE dept = 'ops'"
    assert reward(db, pred, gold) == 0.0


# --- failures score zero ---

def test_syntax_error(db):
    assert reward(db, "SELEC name FROM employees", "SELECT name FROM employees") == 0.0


def test_missing_table(db):
    assert reward(db, "SELECT * FROM nope", "SELECT name FROM employees") == 0.0


def test_multiple_statements_rejected(db):
    pred = "SELECT name FROM employees; SELECT 1"
    assert reward(db, pred, "SELECT name FROM employees") == 0.0


def test_write_is_blocked_and_db_unchanged(db):
    gold = "SELECT name FROM employees"
    assert reward(db, "DELETE FROM employees", gold) == 0.0
    assert len(execute_sql(db, gold).rows) == 4


def test_timeout(db):
    runaway = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) "
        "SELECT count(*) FROM r"
    )
    start = time.monotonic()
    assert reward(db, runaway, "SELECT 1", timeout_s=0.3) == 0.0
    assert time.monotonic() - start < 5


# --- gold problems and known loopholes ---

def test_bad_gold_raises(db):
    with pytest.raises(ValueError):
        reward(db, "SELECT 1", "SELECT * FROM nope")


def test_empty_matches_empty_known_loophole(db):
    # Documents current behavior: an empty prediction "solves" any empty-gold question.
    gold = "SELECT name FROM employees WHERE salary > 1000000"
    pred = "SELECT name FROM employees WHERE 1 = 0"
    assert reward(db, pred, gold) == 1.0

def test_large_results_are_compared(db):
    q = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < 20000) "
        "SELECT x FROM r"
    )
    assert reward(db, q, q) == 1.0