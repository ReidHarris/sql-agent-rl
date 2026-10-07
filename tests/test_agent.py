from sqlagent.agent import parse_action, run_episode
from sqlagent.env import SQLEnv


def call(tool, **args):
    return f'<tool_call>{{"tool": "{tool}", "args": {args!r}}}</tool_call>'.replace("'", '"')


def scripted(outputs):
    it = iter(outputs)
    return lambda messages: next(it)


# --- parser ---

def test_parse_valid_call():
    text = 'Let me look.\n<tool_call>{"tool": "list_tables", "args": {}}</tool_call>'
    assert parse_action(text) == {"tool": "list_tables", "args": {}}


def test_parse_missing_closing_tag():
    text = '<tool_call>{"tool": "list_tables", "args": {}}'
    assert parse_action(text) == {"tool": "list_tables", "args": {}}


def test_parse_newline_inside_sql_string():
    text = '<tool_call>{"tool": "run_sql", "args": {"query": "SELECT 1\nFROM t"}}</tool_call>'
    assert parse_action(text)["args"]["query"] == "SELECT 1\nFROM t"


def test_parse_uses_first_call_only():
    text = (
        '<tool_call>{"tool": "list_tables", "args": {}}</tool_call>'
        '<tool_call>{"tool": "submit_answer", "args": {"query": "SELECT 1"}}</tool_call>'
    )
    assert parse_action(text)["tool"] == "list_tables"


def test_parse_failures_return_none():
    assert parse_action("I think the answer is 4.") is None
    assert parse_action("<tool_call>not json</tool_call>") is None
    assert parse_action("<tool_call>[1, 2, 3]</tool_call>") is None


# --- episode loop ---

def test_successful_episode(example):
    outputs = [
        call("list_tables"),
        call("submit_answer", query="SELECT name FROM employees WHERE salary > 60000"),
    ]
    traj = run_episode(SQLEnv(), example, scripted(outputs))
    assert traj.reward == 1.0
    assert traj.turns == 2
    assert not traj.truncated
    assert traj.invalid_actions == 0
    roles = [m["role"] for m in traj.messages]
    assert roles == ["system", "user", "assistant", "user", "assistant"]


def test_invalid_output_is_counted_and_episode_continues(example):
    outputs = [
        "I think the answer is Ana.",
        call("submit_answer", query="SELECT name FROM employees WHERE salary > 60000"),
    ]
    traj = run_episode(SQLEnv(), example, scripted(outputs))
    assert traj.invalid_actions == 1
    assert traj.reward == 1.0


def test_sql_errors_are_counted(example):
    outputs = [
        call("run_sql", query="SELEC 1"),
        call("submit_answer", query="SELECT name FROM employees WHERE salary > 60000"),
    ]
    traj = run_episode(SQLEnv(), example, scripted(outputs))
    assert traj.sql_errors == 1


def test_turn_limit_gives_truncated_zero_reward(example):
    env = SQLEnv(max_turns=3)
    traj = run_episode(env, example, lambda messages: call("list_tables"))
    assert traj.truncated
    assert traj.reward == 0.0
    assert traj.turns == 3


def test_missing_closing_tag_is_restored_in_history(example):
    outputs = [
        '<tool_call>{"tool": "submit_answer", "args": {"query": "SELECT 1"}}',
    ]
    traj = run_episode(SQLEnv(), example, scripted(outputs))
    assert traj.messages[-1]["content"].endswith("</tool_call>")