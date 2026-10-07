from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from sqlagent.agent import Trajectory, run_episode
from sqlagent.data import Example
from sqlagent.env import SQLEnv


def make_generate(
    model: str,
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> Callable[[list[dict]], str]:
    """Build a generate(messages) -> str function backed by an OpenAI-compatible server."""
    client = OpenAI(base_url=base_url, api_key="not-needed", timeout=timeout, max_retries=3)

    def generate(messages: list[dict]) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["</tool_call>"],
        )
        return response.choices[0].message.content or ""

    return generate


def run_many(
    examples: list[Example],
    generate: Callable[[list[dict]], str],
    max_turns: int = 8,
    workers: int = 32,
) -> Iterator[Trajectory]:
    """Run episodes concurrently, yielding each trajectory as it finishes.

    Results arrive in completion order, not dataset order. Each episode gets its own
    SQLEnv because an environment holds per-episode state.
    """

    def one(example: Example) -> Trajectory:
        return run_episode(SQLEnv(max_turns=max_turns), example, generate)

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(one, ex) for ex in examples]
        for future in as_completed(futures):
            yield future.result()


def trajectory_to_dict(traj: Trajectory) -> dict:
    """Flatten a trajectory into a JSON-serializable record."""
    return {
        "question": traj.example.question,
        "db_id": traj.example.db_id,
        "gold_sql": traj.example.gold_sql,
        "reward": traj.reward,
        "turns": traj.turns,
        "truncated": traj.truncated,
        "invalid_actions": traj.invalid_actions,
        "sql_errors": traj.sql_errors,
        "messages": traj.messages,
    }