import argparse
import os
from typing import List

import matplotlib.pyplot as plt

from sha.drl_transformer import EpisodeRewardTracker, MandlNetwork, SimpleRoutePlanner


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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    network = MandlNetwork.from_csv(args.links, args.demand, num_stops=args.num_stops)
    planner = SimpleRoutePlanner(network)

    reward_tracker = EpisodeRewardTracker()
    demands = []
    travel_times = []
    selected_routes = []

    for episode in range(args.episodes):
        routes = planner.select_routes(args.num_routes, args.max_length)
        selected_routes = routes
        demand_served = planner.demand_served(routes)
        travel_time = sum(planner.compute_route_travel_time(route) for route in routes)
        reward = demand_served - travel_time
        reward_tracker.record(reward)
        demands.append(demand_served)
        travel_times.append(travel_time)

    reward_plot = _plot_rewards(reward_tracker.values(), args.output_dir)
    routes_plot = _plot_routes(selected_routes, args.output_dir)
    demand_plot = _plot_demand_travel(reward_tracker.values(), demands, travel_times, args.output_dir)

    print("Reward curve:", reward_plot)
    print("Selected routes:", routes_plot)
    print("Demand/travel time:", demand_plot)


if __name__ == "__main__":
    main()
