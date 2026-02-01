from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


def _variance(values: List[int]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


@dataclass
class StepMetrics:
    coverage: float
    unmet_demand: float
    cost: float
    balance_penalty: float
    reward: float


class BusTransitEnv:
    """Minimal bus transit network design environment for RL optimization."""

    def __init__(
        self,
        demand: List[float],
        target_frequencies: List[int],
        initial_frequencies: List[int] | None = None,
        min_frequency: int = 1,
        max_frequency: int = 10,
        max_steps: int = 30,
        coverage_weight: float = 1.0,
        unmet_penalty: float = 2.0,
        cost_per_bus: float = 1.0,
        balance_weight: float = 0.1,
    ) -> None:
        if len(demand) != len(target_frequencies):
            raise ValueError("Demand and target frequencies must match in length.")
        if initial_frequencies and len(initial_frequencies) != len(demand):
            raise ValueError("Initial frequencies must match demand length.")
        self.demand = list(demand)
        self.target_frequencies = list(target_frequencies)
        self.initial_frequencies = list(initial_frequencies or target_frequencies)
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.max_steps = max_steps
        self.coverage_weight = coverage_weight
        self.unmet_penalty = unmet_penalty
        self.cost_per_bus = cost_per_bus
        self.balance_weight = balance_weight
        self._frequencies = list(self.initial_frequencies)
        self._steps_taken = 0

    @property
    def action_size(self) -> int:
        return len(self._frequencies) * 2

    def reset(self) -> Tuple[int, ...]:
        self._frequencies = list(self.initial_frequencies)
        self._steps_taken = 0
        return self.state

    @property
    def state(self) -> Tuple[int, ...]:
        return tuple(self._frequencies)

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, float]]:
        if action < 0 or action >= self.action_size:
            raise ValueError("Action out of bounds.")
        route_index = action // 2
        delta = 1 if action % 2 == 0 else -1
        updated = self._frequencies[route_index] + delta
        updated = max(self.min_frequency, min(self.max_frequency, updated))
        self._frequencies[route_index] = updated
        metrics = self._evaluate()
        self._steps_taken += 1
        done = self._steps_taken >= self.max_steps
        return self.state, metrics.reward, done, metrics.__dict__

    def _evaluate(self) -> StepMetrics:
        coverage = 0.0
        unmet = 0.0
        for demand, frequency, target in zip(
            self.demand, self._frequencies, self.target_frequencies
        ):
            served_fraction = min(frequency / target, 1.0) if target > 0 else 0.0
            coverage += demand * served_fraction
            unmet += demand * (1.0 - served_fraction)
        cost = sum(self._frequencies) * self.cost_per_bus
        balance_penalty = self.balance_weight * _variance(self._frequencies)
        reward = (
            self.coverage_weight * coverage
            - self.unmet_penalty * unmet
            - cost
            - balance_penalty
        )
        return StepMetrics(
            coverage=coverage,
            unmet_demand=unmet,
            cost=cost,
            balance_penalty=balance_penalty,
            reward=reward,
        )

    def describe(self) -> Dict[str, float]:
        metrics = self._evaluate()
        return metrics.__dict__.copy()
