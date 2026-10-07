from sqlagent.rollout import run_many, trajectory_to_dict

CORRECT = (
    '<tool_call>{"tool": "submit_answer", '
    '"args": {"query": "SELECT name FROM employees WHERE salary > 60000"}}</tool_call>'
)


def test_run_many_yields_one_trajectory_per_example(example):
    examples = [example] * 5
    trajs = list(run_many(examples, lambda messages: CORRECT, workers=4))
    assert len(trajs) == 5
    assert all(t.reward == 1.0 for t in trajs)


def test_trajectory_record_is_json_serializable(example):
    import json

    traj = next(run_many([example], lambda messages: CORRECT))
    record = trajectory_to_dict(traj)
    assert json.loads(json.dumps(record))["reward"] == 1.0