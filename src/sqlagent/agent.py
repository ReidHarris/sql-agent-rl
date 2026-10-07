import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from sqlagent.data import Example
from sqlagent.env import SQLEnv
from sqlagent.prompt import build_system_prompt

STOP = "</tool_call>"
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)

def parse_action(text: str):
    """Return an action dict from the first tool call in text, or None if there isn't one."""
    match = TOOL_CALL_RE.search(text)
    if match is None:
        return None
    try:
        action = json.loads(match.group(1).strip(), strict=False)
    except json.JSONDecodeError:
        return None
    return action if isinstance(action, dict) else None

@dataclass
class Trajectory:
    example: Example
    messages: list[dict]
    reward: float
    turns: int
    truncated: bool
    invalid_actions: int
    sql_errors: int

def run_episode(
    env: SQLEnv, example: Example, generate: Callable[[list[dict]], str]
) -> Trajectory:
    question = env.reset(example)
    messages = [
        {"role": "system", "content": build_system_prompt(env.max_turns)},
        {"role": "user", "content": question},
    ]
    invalid_actions = 0
    sql_errors = 0
    while True:
        text = generate(messages)
        # Generation stops at the closing tag and drops it. Put it back.
        if "<tool_call>" in text and STOP not in text:
            text += STOP
        messages.append({"role": "assistant", "content": text})

        result = env.step(parse_action(text))
        invalid_actions += bool(result.info.get("invalid_action"))
        sql_errors += result.observation.startswith("SQL error")
        if result.done:
            break
        messages.append({"role": "user", "content": f"Observation:\n{result.observation}"})

    return Trajectory(
        example=example,
        messages=messages,
        reward=result.reward,
        turns=env.turn,
        truncated=bool(result.info.get("truncated")),
        invalid_actions=invalid_actions,
        sql_errors=sql_errors,
    )