from __future__ import annotations

from typing import Dict

from .agent import QLearningAgent, summarize_training
from .environment import BusTransitEnv


DEFAULT_DEMAND = [120.0, 80.0, 100.0, 60.0]
DEFAULT_TARGETS = [8, 6, 7, 4]


def run_training(episodes: int = 30) -> Dict[str, object]:
    env = BusTransitEnv(
        demand=DEFAULT_DEMAND,
        target_frequencies=DEFAULT_TARGETS,
        initial_frequencies=[5, 5, 5, 5],
        max_steps=20,
    )
    agent = QLearningAgent(env)
    result = agent.train(episodes=episodes)
    return summarize_training(result)


if __name__ == "__main__":
    summary = run_training()
    print("Training summary:", summary)
