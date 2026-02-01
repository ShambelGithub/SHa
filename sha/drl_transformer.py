from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence

import math
import random


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

    def select_routes(self, num_routes: int, max_length: int) -> list[list[int]]:
        routes = []
        adjacency = _build_adjacency(self.network.num_stops, self.network.links)
        start_nodes = list(adjacency.keys())[:num_routes]
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
