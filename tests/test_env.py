from pathlib import Path

import pytest

from sqlagent.data import DEFAULT_SPIDER_DIR, load_split
from sqlagent.env import SQLEnv


@pytest.fixture
def env(example):
    e = SQLEnv(max_turns=4)
    e.reset(example)
    return e


def act(tool, **args):
    return {"tool": tool, "args": args}


def test_reset_returns_question(example):
    assert "Who earns more than 60000?" in SQLEnv().reset(example)


def test_list_tables(env):
    result = env.step(act("list_tables"))
    assert "employees" in result.observation
    assert not result.done


def test_describe_table_shows_columns_and_samples(env):
    result = env.step(act("describe_table", name="employees"))
    assert "salary" in result.observation
    assert "Ana" in result.observation


def test_describe_table_is_case_insensitive(env):
    assert "salary" in env.step(act("describe_table", name="EMPLOYEES")).observation


def test_describe_unknown_table(env):
    result = env.step(act("describe_table", name="nope"))
    assert result.observation.startswith("Error")
    assert not result.done


def test_run_sql_shows_header_and_rows(env):
    result = env.step(act("run_sql", query="SELECT name FROM employees WHERE dept = 'ops'"))
    assert "name" in result.observation
    assert "Cy" in result.observation


def test_run_sql_error_is_an_observation(env):
    result = env.step(act("run_sql", query="SELEC 1"))
    assert "SQL error" in result.observation
    assert not result.done
    assert result.reward == 0.0


def test_run_sql_cannot_modify_data(env):
    env.step(act("run_sql", query="DELETE FROM employees"))
    result = env.step(act("run_sql", query="SELECT count(*) FROM employees"))
    assert "4" in result.observation


def test_long_results_are_truncated(env):
    query = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < 100) "
        "SELECT x FROM r"
    )
    result = env.step(act("run_sql", query=query))
    assert "100 rows total" in result.observation


def test_submit_correct_answer(env):
    result = env.step(act("submit_answer", query="SELECT name FROM employees WHERE salary > 60000"))
    assert result.done
    assert result.reward == 1.0


def test_submit_wrong_answer(env):
    result = env.step(act("submit_answer", query="SELECT name FROM employees"))
    assert result.done
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "bad_action",
    [
        "list_tables",
        {},
        {"tool": "nope"},
        {"tool": "run_sql", "args": {}},
        {"tool": "run_sql", "args": "SELECT 1"},
        {"tool": "submit_answer", "args": {"query": ""}},
    ],
)
def test_malformed_actions_do_not_crash(env, bad_action):
    result = env.step(bad_action)
    assert result.observation.startswith("Error")
    assert not result.done
    assert result.reward == 0.0


def test_turn_limit_ends_episode(env):
    for _ in range(3):
        assert not env.step(act("list_tables")).done
    result = env.step(act("list_tables"))
    assert result.done
    assert result.reward == 0.0
    assert result.info["truncated"]


def test_step_after_done_raises(env):
    env.step(act("submit_answer", query="SELECT 1"))
    with pytest.raises(RuntimeError):
        env.step(act("list_tables"))


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        SQLEnv().step(act("list_tables"))


@pytest.mark.skipif(not Path(DEFAULT_SPIDER_DIR).exists(), reason="Spider data not downloaded")
def test_scripted_agent_solves_spider_dev_sample():
    env = SQLEnv()
    for ex in load_split("dev")[:50]:
        env.reset(ex)
        env.step(act("list_tables"))
        result = env.step(act("submit_answer", query=ex.gold_sql))
        assert result.reward == 1.0