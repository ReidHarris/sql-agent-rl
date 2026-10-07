from dataclasses import dataclass, field

from sqlagent.data import Example
from sqlagent.reward import execute_sql, execution_reward

MAX_OBS_CHARS = 2000
MAX_DISPLAY_ROWS = 20
TOOLS = ("list_tables", "describe_table", "run_sql", "submit_answer")

@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict = field(default_factory=dict)

def _truncate(text: str) -> str:
    if len(text) <= MAX_OBS_CHARS:
        return text
    return text[:MAX_OBS_CHARS] + "\n... (output trunctated)"

def _format_result(columns, rows) -> str:
    header = " | ".join(columns)
    if not rows:
        return f"{header}\n(no rows)"
    lines = [" | ".join(str(v) for v in row) for row in rows[:MAX_DISPLAY_ROWS]]
    if len(rows) > MAX_DISPLAY_ROWS:
        lines.append(f"... ({len(rows)} rows total; showing first {MAX_DISPLAY_ROWS})")
    return "\n".join([header, *lines])

class SQLEnv:
    def __init__(self, max_turns: int = 8, timeout_s: float = 5.0):
        self.max_turns = max_turns
        self.timeout_s = timeout_s
        self.example: Example | None = None
        self.turn = 0
        self.done = True

    def reset(self, example: Example) -> str: 
        self.example = example 
        self.turn = 0 
        self.done = False 
        return f"Question: {example.question}"

    def step(self, action) -> StepResult:
        if self.done:
            raise RuntimeError("Episode is over (or never started); call reset() first.")
        self.turn += 1
        result = self._dispatch(action)
        if not result.done and self.turn >= self.max_turns:
            result = StepResult(
                observation = result.observation,
                reward=0.0,
                done=True,
                info={"truncated": True},
            )
        self.done = result.done
        return result

    # --- helpers ---

    def _obs(self, text: str) -> StepResult:
        return StepResult(observation=_truncate(text), reward=0.0, done=False)

    def _error(self, message: str) -> StepResult:
        result = self._obs(f"Error: {message}")
        result.info["invalid_action"] = True
        return result

    def _dispatch(self, action) -> StepResult:
        if not isinstance(action, dict) or "tool" not in action:
            return self._error('Action must be an object like {"tool": ..., "args": {...}}.')
        tool = action["tool"]
        args = action.get("args") or {}
        if not isinstance(args, dict):
            return self._error("'args' must be an object.")
        if tool == "list_tables":
            return self._list_tables()
        if tool == "describe_table":
            return self._describe_table(args.get("name"))
        if tool == "run_sql":
            return self._run_sql(args.get("query"))
        if tool == "submit_answer":
            return self._submit(args.get("query"))
        return self._error(f"Unknown tool {tool!r}. Available tools: {', '.join(TOOLS)}.")

     # --- tools ---

    def _table_names(self) -> list[str]:
        res = execute_sql(
            self.example.db_path,
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
            self.timeout_s,
        )
        return [row[0] for row in res.rows] if res.ok else []

    def _list_tables(self) -> StepResult:
        names = self._table_names()
        return self._obs("\n".join(names) if names else "(no tables)")

    def _describe_table(self, name) -> StepResult:
        if not isinstance(name, str):
            return self._error("describe_table requires a string argument 'name'.")
        lookup = {t.lower(): t for t in self._table_names()}
        real = lookup.get(name.lower())
        if real is None:
            return self._error(f"No table named {name!r}. Use list_tables to see the tables.")
        quoted = '"' + real.replace('"', '""') + '"'
        info = execute_sql(self.example.db_path, f"PRAGMA table_info({quoted})", self.timeout_s)
        if not info.ok:
            return self._error(f"Could not describe {real}: {info.error}")
        lines = [f"Table {real}:"]
        for _cid, col, col_type, _notnull, _default, pk in info.rows:
            lines.append(f"  {col} {col_type}" + (" PRIMARY KEY" if pk else ""))
        sample = execute_sql(
            self.example.db_path, f"SELECT * FROM {quoted} LIMIT 3", self.timeout_s
        )
        if sample.ok and sample.rows:
            lines.append("Sample rows:")
            lines.append(_format_result(sample.columns, sample.rows))
        return self._obs("\n".join(lines))

    def _run_sql(self, query) -> StepResult:
        if not isinstance(query, str) or not query.strip():
            return self._error("run_sql requires a non-empty string argument 'query'.")
        res = execute_sql(self.example.db_path, query, self.timeout_s)
        if not res.ok:
            return self._obs(f"SQL error: {res.error}")
        return self._obs(_format_result(res.columns, res.rows))

    def _submit(self, query) -> StepResult:
        if not isinstance(query, str) or not query.strip():
            return self._error("submit_answer requires a non-empty string argument 'query'.")
        reward = execution_reward(
            query, self.example.gold_sql, self.example.db_path, self.timeout_s
        )
        return StepResult(
            observation="Answer submitted.",
            reward=reward,
            done=True,
            info={"submitted_sql": query},
        )