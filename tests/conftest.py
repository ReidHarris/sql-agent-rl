import sqlite3

import pytest

from sqlagent.data import Example


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, dept TEXT, salary REAL);
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


@pytest.fixture
def example(db):
    return Example(
        question="Who earns more than 60000?",
        db_id="test",
        gold_sql="SELECT name FROM employees WHERE salary > 60000",
        db_path=db,
    )