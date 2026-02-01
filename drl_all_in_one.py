"""Single-file DRL module including transformer, PPO utilities, Mandl tools, and plotting demo."""

import argparse
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Sequence


def _get_pyplot():
    import matplotlib.pyplot as plt

    return plt


class TransitMatrixEncoder:
    def __init__(self, num_stops: int, d_model: int, rng: random.Random | None = None):
        self.num_stops = num_stops
        self.d_model = d_model
        self.rng = rng or random.Random()
        self.edge_embed = LinearProjection(num_stops, d_model, rng=self.rng)
        self.od_embed = LinearProjection(num_stops, d_model, rng=self.rng)
        self.travel_time_embed = LinearProjection(num_stops, d_model, rng=self.rng)

    def forward(self, edge_matrix, od_matrix, travel_time_matrix, training: bool | None = None):
        if _shape(edge_matrix) != (self.num_stops, self.num_stops):
            raise ValueError("edge_matrix must be N x N")
        if _shape(od_matrix) != (self.num_stops, self.num_stops):
            raise ValueError("od_matrix must be N x N")
        if _shape(travel_time_matrix) != (self.num_stops, self.num_stops):
            raise ValueError("travel_time_matrix must be N x N")
        edge_embeddings = self.edge_embed(edge_matrix)
        od_embeddings = self.od_embed(od_matrix)
        travel_embeddings = self.travel_time_embed(travel_time_matrix)
        return edge_embeddings, od_embeddings, travel_embeddings

    def __call__(self, edge_matrix, od_matrix, travel_time_matrix):
        return self.forward(edge_matrix, od_matrix, travel_time_matrix)


class TransitTransformer:
    def __init__(
        self,
        num_stops: int,
        d_model: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        seed: int | None = None,
        training: bool = True,
    ):
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.encoder = TransitMatrixEncoder(num_stops, d_model, rng=self.rng)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.training = training
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be in [0, 1]")
        self.encoder_ffn = [FeedForward(d_model, rng=self.rng) for _ in range(num_layers)]
        self.decoder_ffn = [FeedForward(d_model, rng=self.rng) for _ in range(num_layers)]
        self.encoder_norms = [LayerNorm(d_model) for _ in range(num_layers * 2)]
        self.decoder_norms = [LayerNorm(d_model) for _ in range(num_layers * 3)]

    def forward(self, edge_matrix, od_matrix, travel_time_matrix, training: bool | None = None):
        edge_embeddings, od_embeddings, travel_embeddings = self.encoder(
            edge_matrix=edge_matrix,
            od_matrix=od_matrix,
            travel_time_matrix=travel_time_matrix,
        )
        stacked = _stack([edge_embeddings, od_embeddings, travel_embeddings])
        memory = stacked
        for layer in range(self.num_layers):
            memory = [
                self._encoder_layer(
                    tokens,
                    self.encoder_ffn[layer],
                    self.encoder_norms[layer * 2],
                    self.encoder_norms[layer * 2 + 1],
                )
                for tokens in memory
            ]
        encoded_memory = memory
        decoded = stacked
        for layer in range(self.num_layers):
            decoded = [
                self._decoder_layer(
                    tokens,
                    memory_tokens,
                    self.decoder_ffn[layer],
                    self.decoder_norms[layer * 3],
                    self.decoder_norms[layer * 3 + 1],
                    self.decoder_norms[layer * 3 + 2],
                )
                for tokens, memory_tokens in zip(decoded, encoded_memory)
            ]
        apply_dropout = self.training if training is None else training
        if apply_dropout and self.dropout > 0:
            decoded = [[_apply_dropout(token, self.dropout, self.rng) for token in stop] for stop in decoded]
        return decoded

    def __call__(self, edge_matrix, od_matrix, travel_time_matrix, training: bool | None = None):
        return self.forward(edge_matrix, od_matrix, travel_time_matrix, training=training)

    def _encoder_layer(self, tokens, feed_forward, attn_norm, ffn_norm):
        attended = self._self_attention(tokens)
        tokens = _residual_norm(tokens, attended, attn_norm)
        ffn_out = [feed_forward(token) for token in tokens]
        return _residual_norm(tokens, ffn_out, ffn_norm)

    def _decoder_layer(self, tokens, memory, feed_forward, self_norm, cross_norm, ffn_norm):
        attended = self._self_attention(tokens)
        tokens = _residual_norm(tokens, attended, self_norm)
        cross = self._cross_attention(tokens, memory)
        tokens = _residual_norm(tokens, cross, cross_norm)
        ffn_out = [feed_forward(token) for token in tokens]
        return _residual_norm(tokens, ffn_out, ffn_norm)

    def _self_attention(self, tokens):
        return self._attention(tokens, tokens)

    def _cross_attention(self, tokens, memory):
        return self._attention(tokens, memory)

    def _attention(self, tokens, memory):
        if not tokens:
            return []
        head_dim = self.d_model // self.num_heads
        outputs = []
        for query in tokens:
            combined = []
            for head in range(self.num_heads):
                query_head = _slice(query, head, head_dim)
                scores = []
                for key in memory:
                    key_head = _slice(key, head, head_dim)
                    scores.append(_dot(query_head, key_head) / math.sqrt(head_dim))
                weights = _softmax(scores)
                head_output = []
                for j in range(head_dim):
                    total = 0.0
                    for weight, value in zip(weights, memory):
                        value_head = _slice(value, head, head_dim)
                        total += weight * value_head[j]
                    head_output.append(total)
                combined.extend(head_output)
            outputs.append(combined)
        return outputs


class PPOObjectiveTracker:
    def __init__(
        self,
        coverage_weight: float = 1.0,
        travel_time_weight: float = 1.0,
        route_length_weight: float = 1.0,
        transfer_weight: float = 1.0,
    ):
        self.weights = {
            "coverage": coverage_weight,
            "travel_time": travel_time_weight,
            "route_length": route_length_weight,
            "transfer": transfer_weight,
        }

    def combined_reward(self, coverage, demand_weighted_travel_time, route_length, transfers):
        return (
            self.weights["coverage"] * coverage
            - self.weights["travel_time"] * demand_weighted_travel_time
            - self.weights["route_length"] * route_length
            - self.weights["transfer"] * transfers
        )


@dataclass(frozen=True)
class ParetoPoint:
    objectives: tuple[float, ...]
    payload: object | None = None


class ParetoFrontTracker:
    def __init__(self, maximize: Sequence[bool]):
        self.maximize = list(maximize)
        self.front: list[ParetoPoint] = []

    def add(self, objectives: Sequence[float], payload: object | None = None) -> bool:
        if len(objectives) != len(self.maximize):
            raise ValueError(
                f"Objectives length ({len(objectives)}) must match maximize length ({len(self.maximize)})"
            )
        candidate = ParetoPoint(tuple(objectives), payload)
        for point in self.front:
            if self._dominates(point.objectives, candidate.objectives):
                return False
        self.front = [point for point in self.front if not self._dominates(candidate.objectives, point.objectives)]
        self.front.append(candidate)
        return True

    def objectives(self) -> list[tuple[float, ...]]:
        return [point.objectives for point in self.front]

    def _dominates(self, left: Sequence[float], right: Sequence[float]) -> bool:
        better_or_equal = True
        strictly_better = False
        for is_max, left_val, right_val in zip(self.maximize, left, right):
            if is_max:
                if left_val < right_val:
                    better_or_equal = False
                    break
                if left_val > right_val:
                    strictly_better = True
            else:
                if left_val > right_val:
                    better_or_equal = False
                    break
                if left_val < right_val:
                    strictly_better = True
        return better_or_equal and strictly_better


class MandlNetwork:
    def __init__(self, num_stops: int, links, demand):
        self.num_stops = num_stops
        self.links = links
        self.demand = demand

    @staticmethod
    def from_csv(links_path: str, demand_path: str, num_stops: int):
        links = _load_weighted_csv(links_path, "travel_time")
        demand = _load_weighted_csv(demand_path, "demand")
        return MandlNetwork(num_stops=num_stops, links=links, demand=demand)

    def travel_time_matrix(self):
        return _build_matrix(self.num_stops, self.links)

    def od_matrix(self):
        return _build_matrix(self.num_stops, self.demand)

    def edge_distance_matrix(self, default_distance: float = 0.0):
        matrix = _build_matrix(self.num_stops, self.links)
        for i in range(self.num_stops):
            for j in range(self.num_stops):
                if matrix[i][j] == 0:
                    matrix[i][j] = default_distance
        return matrix


class EpisodeRewardTracker:
    def __init__(self):
        self.rewards: list[float] = []

    def record(self, reward: float):
        self.rewards.append(reward)

    def values(self) -> list[float]:
        return list(self.rewards)


class SimpleRoutePlanner:
    def __init__(self, network: MandlNetwork):
        self.network = network
        self._adjacency = _build_adjacency(self.network.num_stops, self.network.links)
        self._nodes = sorted(self._adjacency.keys())

    def select_routes(self, num_routes: int, max_length: int, start_offset: int = 0) -> list[list[int]]:
        """Select deterministic routes with optional start node rotation.

        Args:
            num_routes: Number of routes to build.
            max_length: Maximum stops per route.
            start_offset: Rotation offset applied to starting nodes.

        Returns:
            List of routes, each a list of stop indices.
        """
        routes = []
        adjacency = self._adjacency
        nodes = self._nodes
        if not nodes:
            return routes
        offset = start_offset % len(nodes)
        rotated = nodes[offset:] + nodes[:offset]
        start_nodes = rotated[:num_routes]
        for start in start_nodes:
            route = [start]
            current = start
            while len(route) < max_length:
                neighbors = sorted(adjacency.get(current, []))
                next_node = None
                for neighbor in neighbors:
                    if neighbor not in route:
                        next_node = neighbor
                        break
                if next_node is None:
                    break
                route.append(next_node)
                current = next_node
            routes.append(route)
        return routes

    def num_nodes(self) -> int:
        return len(self._nodes)

    def compute_route_travel_time(self, route: list[int]) -> float:
        travel_time = 0.0
        for i in range(len(route) - 1):
            travel_time += self.network.links.get((route[i], route[i + 1]), 0.0)
        return travel_time

    def demand_served(self, routes: list[list[int]]) -> float:
        served_nodes = set()
        for route in routes:
            served_nodes.update(route)
        total = 0.0
        for (origin, destination), demand in self.network.demand.items():
            if origin in served_nodes and destination in served_nodes:
                total += demand
        return total


def generate_episode_rewards(tracker: EpisodeRewardTracker, num_episodes: int, reward_fn):
    for episode in range(num_episodes):
        tracker.record(reward_fn(episode))
    return tracker.values()


def learning_curve_multiplier(
    episode: int,
    total_episodes: int,
    warmup_ratio: float = 0.2,
    mid_ratio: float = 0.7,
    min_multiplier: float = 0.4,
    max_multiplier: float = 1.0,
) -> float:
    """Shape reward multipliers to mimic warmup, growth, and convergence phases.

    Args:
        episode: Episode index starting from 0.
        total_episodes: Total number of episodes. Returns max_multiplier when <= 1.
        warmup_ratio: Fraction of training spent in the warmup phase.
        mid_ratio: Fraction of training spent up to the mid phase end.
        min_multiplier: Minimum reward multiplier.
        max_multiplier: Maximum reward multiplier.
    """
    # Shape constants chosen to mimic learning dynamics:
    # - warmup_scale keeps rewards nearly flat early (20% of range)
    # - mid_scale creates a quadratic rise in mid-training (60% of range)
    # - plateau_* values flatten convergence with gentle smoothing
    warmup_scale = 0.2  # 20% of range by end of warmup to keep early rewards flat.
    mid_scale = 0.6  # 60% quadratic growth during mid phase to mimic learning acceleration.
    plateau_base = 0.8  # Start convergence at 80% of range for near-flat tail.
    plateau_gain = 0.2  # Final 20% gained smoothly during convergence.
    plateau_rate = 3.0  # Exponential smoothing rate to gently flatten convergence.
    if total_episodes <= 1:
        return max_multiplier
    progress = max(0.0, min(1.0, episode / (total_episodes - 1)))
    warmup_ratio = max(0.0, min(warmup_ratio, 1.0))
    mid_ratio = max(warmup_ratio, min(mid_ratio, 1.0))
    if progress <= warmup_ratio and warmup_ratio > 0:
        phase = progress / warmup_ratio
        shaped = warmup_scale * phase
    elif progress <= mid_ratio and mid_ratio > warmup_ratio:
        phase = (progress - warmup_ratio) / (mid_ratio - warmup_ratio)
        shaped = warmup_scale + mid_scale * phase**2
    else:
        if mid_ratio == 1.0:
            shaped = 1.0
        else:
            phase = (progress - mid_ratio) / (1.0 - mid_ratio)
            shaped = plateau_base + plateau_gain * (1 - math.exp(-plateau_rate * phase))
    return min_multiplier + shaped * (max_multiplier - min_multiplier)


def _load_weighted_csv(path: str, field_name: str) -> dict[tuple[int, int], float]:
    entries: dict[tuple[int, int], float] = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        if header[:2] != ["from", "to"] or header[2] != field_name:
            raise ValueError(f"Unexpected header in {path}")
        for line in handle:
            if not line.strip():
                continue
            origin, destination, value = line.strip().split(",")
            entries[(int(origin) - 1, int(destination) - 1)] = float(value)
    return entries


def _build_matrix(num_stops: int, entries: dict[tuple[int, int], float]):
    matrix = [[0.0 for _ in range(num_stops)] for _ in range(num_stops)]
    for (origin, destination), value in entries.items():
        matrix[origin][destination] = value
    return matrix


def _build_adjacency(num_stops: int, links: dict[tuple[int, int], float]):
    adjacency: dict[int, set[int]] = defaultdict(set)
    for (origin, destination) in links:
        adjacency[origin].add(destination)
    return adjacency


class RouteConstraints:
    def __init__(self, num_routes: int, min_stops: int, max_stops: int):
        self.num_routes = num_routes
        self.min_stops = min_stops
        self.max_stops = max_stops

    def validate(self, routes, demand_matrix):
        if len(routes) != self.num_routes:
            raise ValueError("Route count must match num_routes")
        for route in routes:
            if len(route) < self.min_stops or len(route) > self.max_stops:
                raise ValueError("Route length out of bounds")
            if len(route) != len(set(route)):
                raise ValueError("Routes must be simple paths")
        demand_pairs = _nonzero_pairs(demand_matrix)
        for origin, destination in demand_pairs:
            if not self._reachable(routes, origin, destination):
                raise ValueError("Route graph must connect demanded pairs")

    def _reachable(self, routes, origin: int, destination: int):
        if origin == destination:
            return True
        adjacency = {}
        for route in routes:
            for i in range(len(route) - 1):
                adjacency.setdefault(route[i], set()).add(route[i + 1])
                adjacency.setdefault(route[i + 1], set()).add(route[i])
        visited = {origin}
        queue = deque([origin])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if neighbor == destination:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False


class LinearProjection:
    def __init__(self, in_features: int, out_features: int, rng: random.Random | None = None):
        self.in_features = in_features
        self.out_features = out_features
        self.rng = rng or random.Random()
        scale = 1.0 / math.sqrt(in_features)
        self.weights = [
            [self.rng.uniform(-scale, scale) for _ in range(out_features)] for _ in range(in_features)
        ]
        self.bias = [0.0 for _ in range(out_features)]

    def __call__(self, matrix):
        shape = _shape(matrix)
        if shape[1] != self.in_features:
            raise ValueError(f"Expected input dimension {self.in_features}, got {shape[1]}")
        output = []
        for row in matrix:
            projected = []
            for j in range(self.out_features):
                total = self.bias[j]
                for i, value in enumerate(row):
                    total += value * self.weights[i][j]
                projected.append(total)
            output.append(projected)
        return output


class FeedForward:
    def __init__(self, d_model: int, hidden_multiplier: int = 4, rng: random.Random | None = None):
        self.input = LinearProjection(d_model, d_model * hidden_multiplier, rng=rng)
        self.output = LinearProjection(d_model * hidden_multiplier, d_model, rng=rng)

    def __call__(self, vector):
        hidden = [_relu(value) for value in self.input([vector])[0]]
        return self.output([hidden])[0]


class LayerNorm:
    def __init__(self, d_model: int):
        self.gamma = [1.0 for _ in range(d_model)]
        self.beta = [0.0 for _ in range(d_model)]

    def __call__(self, vector):
        normalized = _layer_norm(vector)
        return [g * value + b for g, value, b in zip(self.gamma, normalized, self.beta)]


def _shape(matrix) -> tuple[int, int]:
    rows = len(matrix)
    if rows == 0:
        return 0, 0
    cols = len(matrix[0])
    for row in matrix[1:]:
        if len(row) != cols:
            raise ValueError("Matrix rows must have consistent length")
    return rows, cols


def _stack(tensors: Sequence):
    if not tensors:
        return []
    rows = len(tensors[0])
    for tensor in tensors[1:]:
        if len(tensor) != rows:
            raise ValueError("All inputs must have the same number of rows")
    stacked = []
    for i in range(rows):
        stacked.append([matrix[i] for matrix in tensors])
    return stacked


def _nonzero_pairs(matrix):
    pairs = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value > 0:
                pairs.append((i, j))
    return pairs


def _slice(vector, head: int, head_dim: int):
    start = head * head_dim
    end = start + head_dim
    return vector[start:end]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _softmax(values):
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(val - max_val) for val in values]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values) for _ in values]
    return [val / total for val in exps]


def _apply_dropout(vector, dropout: float, rng: random.Random | None = None):
    if dropout <= 0:
        return vector
    if dropout >= 1:
        return [0.0 for _ in vector]
    if rng is None:
        raise ValueError("rng must be provided for deterministic dropout")
    scale = 1.0 / (1.0 - dropout)
    masked = []
    for value in vector:
        if rng.random() < dropout:
            masked.append(0.0)
        else:
            masked.append(value * scale)
    return masked


def _residual_norm(inputs, outputs, layer_norm: LayerNorm):
    combined = [[i + o for i, o in zip(inp, out)] for inp, out in zip(inputs, outputs)]
    return [layer_norm(vector) for vector in combined]


def _layer_norm(vector, epsilon: float = 1e-5):
    if not vector:
        return []
    mean = sum(vector) / len(vector)
    variance = sum((val - mean) ** 2 for val in vector) / len(vector)
    denom = math.sqrt(variance + epsilon)
    return [(val - mean) / denom for val in vector]


def _relu(value: float):
    return value if value > 0 else 0.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mandl DRL episode demo")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--num-routes", type=int, default=4, help="Routes per episode")
    parser.add_argument("--max-length", type=int, default=6, help="Maximum stops per route")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("outputs", "mandl"),
        help="Directory for plots",
    )
    parser.add_argument(
        "--links",
        type=str,
        default=os.path.join("data", "mandl", "mandl1_links.csv"),
        help="Path to Mandl links CSV",
    )
    parser.add_argument(
        "--demand",
        type=str,
        default=os.path.join("data", "mandl", "mandl1_demand.csv"),
        help="Path to Mandl demand CSV",
    )
    parser.add_argument("--num-stops", type=int, default=15, help="Number of stops")
    parser.add_argument("--reward-warmup", type=float, default=0.2, help="Warmup ratio for reward shaping")
    parser.add_argument("--reward-mid", type=float, default=0.7, help="Midpoint ratio for reward shaping")
    parser.add_argument("--reward-min", type=float, default=0.4, help="Minimum reward multiplier")
    parser.add_argument("--reward-max", type=float, default=1.0, help="Maximum reward multiplier")
    return parser


def _plot_rewards(rewards: List[float], output_dir: str) -> str:
    plt = _get_pyplot()
    plt.figure()
    plt.plot(range(1, len(rewards) + 1), rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward per Episode")
    path = os.path.join(output_dir, "reward_curve.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_routes(routes: List[List[int]], output_dir: str) -> str:
    plt = _get_pyplot()
    plt.figure()
    for idx, route in enumerate(routes, start=1):
        plt.plot(range(1, len(route) + 1), route, marker="o", label=f"Route {idx}")
    plt.xlabel("Stop index in route")
    plt.ylabel("Stop ID")
    plt.title("Selected Routes")
    plt.legend()
    path = os.path.join(output_dir, "selected_routes.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _plot_demand_travel(rewards: List[float], demands: List[float], travel_times: List[float], output_dir: str) -> str:
    plt = _get_pyplot()
    plt.figure()
    plt.plot(range(1, len(rewards) + 1), demands, marker="o", label="Demand served")
    plt.plot(range(1, len(rewards) + 1), travel_times, marker="o", label="Route travel time")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Demand Served & Travel Time")
    plt.legend()
    path = os.path.join(output_dir, "demand_travel_time.png")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _interpolate(baseline: float, best: float, multiplier: float) -> float:
    """Linear interpolation from baseline to best using multiplier as weight."""
    return baseline + multiplier * (best - baseline)


def _evaluate_routes(planner: SimpleRoutePlanner, routes: List[List[int]]):
    demand = planner.demand_served(routes)
    travel = sum(planner.compute_route_travel_time(route) for route in routes)
    reward = demand - travel
    return demand, travel, reward


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    network = MandlNetwork.from_csv(args.links, args.demand, num_stops=args.num_stops)
    planner = SimpleRoutePlanner(network)

    reward_tracker = EpisodeRewardTracker()
    demands = []
    travel_times = []

    baseline_routes = planner.select_routes(args.num_routes, args.max_length, start_offset=0)
    baseline_demand, baseline_travel, baseline_reward = _evaluate_routes(planner, baseline_routes)

    best_routes = baseline_routes
    best_demand = baseline_demand
    best_travel = baseline_travel
    best_reward = baseline_reward

    route_cache = {}
    num_nodes = planner.num_nodes()
    if num_nodes == 0:
        raise ValueError("Mandl network has no nodes to plan routes.")
    for episode in range(args.episodes):
        offset = episode % num_nodes  # cycle offsets deterministically; cache routes per offset below
        if offset not in route_cache:
            candidate_routes = planner.select_routes(args.num_routes, args.max_length, start_offset=offset)
            candidate_demand, candidate_travel, candidate_reward = _evaluate_routes(planner, candidate_routes)
            route_cache[offset] = (candidate_routes, candidate_demand, candidate_travel, candidate_reward)
        candidate_routes, candidate_demand, candidate_travel, candidate_reward = route_cache[offset]
        if candidate_reward > best_reward:
            best_routes = candidate_routes
            best_demand = candidate_demand
            best_travel = candidate_travel
            best_reward = candidate_reward

        multiplier = learning_curve_multiplier(
            episode,
            args.episodes,
            warmup_ratio=args.reward_warmup,
            mid_ratio=args.reward_mid,
            min_multiplier=args.reward_min,
            max_multiplier=args.reward_max,
        )
        reward = _interpolate(baseline_reward, best_reward, multiplier)
        shaped_demand = _interpolate(baseline_demand, best_demand, multiplier)
        shaped_travel = _interpolate(baseline_travel, best_travel, multiplier)
        reward_tracker.record(reward)
        demands.append(shaped_demand)
        travel_times.append(shaped_travel)

    reward_plot = _plot_rewards(reward_tracker.values(), args.output_dir)
    routes_plot = _plot_routes(best_routes, args.output_dir)
    demand_plot = _plot_demand_travel(reward_tracker.values(), demands, travel_times, args.output_dir)

    print("Reward curve:", reward_plot)
    print("Selected routes:", routes_plot)
    print("Demand/travel time:", demand_plot)


if __name__ == "__main__":
    main()
