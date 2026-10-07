from sqlagent.data import load_split
from sqlagent.env import SQLEnv


def show(action, result):
    print(f">>> {action}")
    print(result.observation)
    print()


ex = load_split("dev")[0]
env = SQLEnv()
print(env.reset(ex), "\n")

action = {"tool": "list_tables", "args": {}}
result = env.step(action)
show(action, result)
table = result.observation.splitlines()[0]

for action in [
    {"tool": "describe_table", "args": {"name": table}},
    {"tool": "run_sql", "args": {"query": ex.gold_sql}},
    {"tool": "submit_answer", "args": {"query": ex.gold_sql}},
]:
    result = env.step(action)
    show(action, result)

print(f"reward={result.reward} done={result.done}")