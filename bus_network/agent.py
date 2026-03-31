from __future__ import annotations

import random
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


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration.

    Uses the Bellman equation to update Q-values and an epsilon that decays
    from ``epsilon_start`` to ``epsilon_end`` over training, ensuring the
    agent explores broadly at first and exploits learned values later.
    This addresses the convergence problem of the pure greedy agent, which
    can get stuck in local optima because it never explores suboptimal actions.
    """

    def __init__(
        self,
        env: BusTransitEnv,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table: Dict[Tuple[int, ...], List[float]] = {}

    def _get_q_values(self, state: Tuple[int, ...]) -> List[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.env.action_size
        return self.q_table[state]

    def _select_action(self, state: Tuple[int, ...], rng: random.Random) -> int:
        if rng.random() < self.epsilon:
            return rng.randrange(self.env.action_size)
        q_values = self._get_q_values(state)
        return q_values.index(max(q_values))

    def train(self, episodes: int = 20, seed: int | None = None) -> TrainingResult:
        rng = random.Random(seed)
        best_state = self.env.reset()
        best_reward = float("-inf")
        reward_history: List[float] = []

        for _ in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self._select_action(state, rng)
                next_state, step_reward, done, _ = self.env.step(action)

                # Bellman update: Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') - Q(s,a)]
                q_values = self._get_q_values(state)
                next_q_max = max(self._get_q_values(next_state))
                td_target = step_reward + self.discount_factor * next_q_max
                q_values[action] += self.learning_rate * (td_target - q_values[action])

                state = next_state
                total_reward += step_reward

            reward_history.append(total_reward)
            if total_reward > best_reward:
                best_reward = total_reward
                best_state = state

            # Decay epsilon after each episode for exploitation–exploration trade-off
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return TrainingResult(best_state=best_state, best_reward=best_reward, reward_history=reward_history)


def summarize_training(result: TrainingResult) -> Dict[str, object]:
    return {
        "best_reward": result.best_reward,
        "best_state": result.best_state,
        "reward_history": result.reward_history,
    }
