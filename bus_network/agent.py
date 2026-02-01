from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .environment import BusTransitEnv


@dataclass
class TrainingResult:
    best_state: Tuple[int, ...]
    best_reward: float
    reward_history: List[float]


class GreedyTransitAgent:
    """Simple baseline agent that greedily improves immediate reward."""

    def __init__(self, env: BusTransitEnv) -> None:
        self.env = env

    def train(self, episodes: int = 20) -> TrainingResult:
        best_state = self.env.reset()
        best_reward = float("-inf")
        reward_history: List[float] = []
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, reward = self._best_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            reward_history.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward
                best_state = state
        return TrainingResult(best_state=best_state, best_reward=best_reward, reward_history=reward_history)

    def _best_action(self, state: Tuple[int, ...]) -> Tuple[int, float]:
        current_reward = self.env.describe()["reward"]
        best_action = 0
        best_reward = current_reward
        for action in range(self.env.action_size):
            snapshot = list(state)
            route_index = action // 2
            delta = 1 if action % 2 == 0 else -1
            updated = snapshot[route_index] + delta
            updated = max(self.env.min_frequency, min(self.env.max_frequency, updated))
            snapshot[route_index] = updated
            self.env._frequencies = snapshot  # local evaluation
            candidate_reward = self.env.describe()["reward"]
            if candidate_reward > best_reward:
                best_reward = candidate_reward
                best_action = action
        self.env._frequencies = list(state)
        return best_action, best_reward


def summarize_training(result: TrainingResult) -> Dict[str, Iterable[float]]:
    return {
        "best_reward": result.best_reward,
        "best_state": result.best_state,
        "reward_history": result.reward_history,
    }
