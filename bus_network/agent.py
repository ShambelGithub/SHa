from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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
                action = self._best_action(state)
                state, step_reward, done, _ = self.env.step(action)
                total_reward += step_reward
            reward_history.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward
                best_state = state
        return TrainingResult(best_state=best_state, best_reward=best_reward, reward_history=reward_history)

    def _best_action(self, state: Tuple[int, ...]) -> int:
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
            candidate_reward = self.env.evaluate_frequencies(snapshot).reward
            if candidate_reward > best_reward:
                best_reward = candidate_reward
                best_action = action
        return best_action


def summarize_training(result: TrainingResult) -> Dict[str, object]:
    return {
        "best_reward": result.best_reward,
        "best_state": result.best_state,
        "reward_history": result.reward_history,
    }
