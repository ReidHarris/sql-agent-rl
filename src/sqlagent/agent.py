import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from sqlagent.data import Example
from sqlagent.env import SQLEnv
from sqlagent.prompt import build_system_prompt

STOP = "</tool_call>"
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)

KEY_ALIASES = {
    "name": "tool",
    "tool_name": "tool",
    "tools": "tool",
    "arguments": "args",
    "parameters": "args",
}

JSON_DECODER = json.JSONDecoder(strict=False)  # strict=False allows raw newlines in strings


def normalize_action(action: dict) -> dict:
    return {KEY_ALIASES.get(k, k): v for k, v in action.items()}


def _first_action_object(text: str):
    """Return the first JSON object in text that has a 'tool' key, or None."""
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = JSON_DECODER.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            obj = normalize_action(obj)
            if "tool" in obj:
                return obj
        start = text.find("{", start + 1)
    return None


def parse_action(text: str):
    """Extract a tool call from model output, or return None if there isn't one.

    Prefers the contents of <tool_call> tags, but falls back to any JSON object in
    the text, which covers markdown fences and bare JSON.
    """
    match = TOOL_CALL_RE.search(text)
    return _first_action_object(match.group(1) if match else text)

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