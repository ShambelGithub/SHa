import argparse
import os
from typing import List

import matplotlib.pyplot as plt

from sha.drl_transformer import (
    EpisodeRewardTracker,
    MandlNetwork,
    SimpleRoutePlanner,
    learning_curve_multiplier,
)


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
        offset = episode % num_nodes  # cycle offsets deterministically while caching routes per offset
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
