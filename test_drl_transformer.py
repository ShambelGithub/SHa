import importlib.util
from pathlib import Path

import pytest

# Load drl_all_in_one as a module (standalone single-file bundle)
_MODULE_PATH = Path(__file__).resolve().parent / "drl_all_in_one.py"
_spec = importlib.util.spec_from_file_location("drl_all_in_one", str(_MODULE_PATH))
_drl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_drl)

TransitMatrixEncoder = _drl.TransitMatrixEncoder
TransitTransformer = _drl.TransitTransformer
PPOObjectiveTracker = _drl.PPOObjectiveTracker
ParetoFrontTracker = _drl.ParetoFrontTracker
RouteConstraints = _drl.RouteConstraints
MandlNetwork = _drl.MandlNetwork
SimpleRoutePlanner = _drl.SimpleRoutePlanner
EpisodeRewardTracker = _drl.EpisodeRewardTracker
TabularQLearningAgent = _drl.TabularQLearningAgent
learning_curve_multiplier = _drl.learning_curve_multiplier


def test_transit_matrix_encoder_shapes():
    num_stops = 4
    d_model = 8
    encoder = TransitMatrixEncoder(num_stops=num_stops, d_model=d_model)
    edge = [[1.0 for _ in range(num_stops)] for _ in range(num_stops)]
    od = [[1.0 if i == j else 0.0 for j in range(num_stops)] for i in range(num_stops)]
    travel = [[0.0 for _ in range(num_stops)] for _ in range(num_stops)]
    edge_emb, od_emb, travel_emb = encoder(edge, od, travel)
    assert len(edge_emb) == num_stops and len(edge_emb[0]) == d_model
    assert len(od_emb) == num_stops and len(od_emb[0]) == d_model
    assert len(travel_emb) == num_stops and len(travel_emb[0]) == d_model


def test_transit_transformer_forward():
    num_stops = 3
    d_model = 6
    expected_num_matrices = 3
    model = TransitTransformer(num_stops=num_stops, d_model=d_model, num_heads=2, num_layers=1)
    edge = [[0.1 for _ in range(num_stops)] for _ in range(num_stops)]
    od = [[0.2 for _ in range(num_stops)] for _ in range(num_stops)]
    travel = [[0.3 for _ in range(num_stops)] for _ in range(num_stops)]
    output = model(edge, od, travel)
    assert len(output) == num_stops
    assert len(output[0]) == expected_num_matrices
    assert len(output[0][0]) == d_model


def test_ppo_objective_tracker_combines_terms():
    coverage_weight = 2.0
    travel_time_weight = 1.0
    route_length_weight = 0.5
    transfer_weight = 0.25
    tracker = PPOObjectiveTracker(
        coverage_weight=coverage_weight,
        travel_time_weight=travel_time_weight,
        route_length_weight=route_length_weight,
        transfer_weight=transfer_weight,
    )
    coverage = 10.0
    travel_time = 4.0
    route_length = 6.0
    transfers = 8.0
    expected = (
        coverage * coverage_weight
        - travel_time_weight * travel_time
        - route_length_weight * route_length
        - transfer_weight * transfers
    )
    reward = tracker.combined_reward(
        coverage=coverage,
        demand_weighted_travel_time=travel_time,
        route_length=route_length,
        transfers=transfers,
    )
    assert reward == expected


def test_route_constraints_validation():
    routes = [[0, 1, 2], [2, 3]]
    demand = [
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    constraints = RouteConstraints(num_routes=2, min_stops=2, max_stops=4)
    constraints.validate(routes, demand)


def test_pareto_front_tracker_keeps_non_dominated():
    tracker = ParetoFrontTracker(maximize=[True, False])
    assert tracker.add([10.0, 5.0]) is True
    assert tracker.add([8.0, 6.0]) is False
    assert tracker.add([12.0, 7.0]) is True
    assert tracker.add([11.0, 4.0]) is True
    objectives = tracker.objectives()
    assert len(objectives) == 2
    assert (10.0, 5.0) not in objectives
    assert (8.0, 6.0) not in objectives
    assert (12.0, 7.0) in objectives
    assert (11.0, 4.0) in objectives


def test_learning_curve_multiplier_progression():
    total = 10
    start = learning_curve_multiplier(0, total)
    mid = learning_curve_multiplier(5, total)
    end = learning_curve_multiplier(9, total)
    assert start < mid < end


def test_learning_curve_multiplier_plateaus():
    total = 30
    min_multiplier = 0.4
    max_multiplier = 1.0
    multiplier_range = max_multiplier - min_multiplier
    plateau_fraction = 0.25
    plateau_tolerance = multiplier_range * plateau_fraction
    mid = learning_curve_multiplier(
        20,
        total,
        warmup_ratio=0.2,
        mid_ratio=0.7,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    end = learning_curve_multiplier(
        29,
        total,
        warmup_ratio=0.2,
        mid_ratio=0.7,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )
    assert end - mid < plateau_tolerance


def test_tabular_qlearning_agent_explores_and_learns():
    """Agent with high epsilon explores different actions; Q-values update."""
    agent = TabularQLearningAgent(num_actions=5, epsilon_start=1.0, epsilon_end=0.05, seed=42)
    actions_taken = set()
    for _ in range(30):
        action = agent.select_action(0)
        agent.update(state=0, action=action, reward=float(action), next_state=0)
        actions_taken.add(action)
    agent.decay_epsilon()  # decay once per episode, not per step
    # After exploration, Q-values for higher-reward actions should be larger
    q_vals = agent.q_table[0]
    assert max(q_vals) > min(q_vals), "Q-values should diverge after updates"
    assert len(actions_taken) > 1, "Agent should explore multiple actions"


def test_tabular_qlearning_epsilon_decays():
    agent = TabularQLearningAgent(num_actions=3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.5, seed=0)
    for _ in range(20):
        agent.decay_epsilon()
    assert agent.epsilon < 0.2


def test_tabular_qlearning_exploits_best_action():
    """After training with ε→0, agent should consistently pick the best action."""
    agent = TabularQLearningAgent(num_actions=4, epsilon_start=0.0, epsilon_end=0.0, seed=7)
    # Manually set Q-values so action 2 is best
    agent.q_table[0] = [0.1, 0.5, 9.9, 0.3]
    selected = {agent.select_action(0) for _ in range(10)}
    assert selected == {2}


def test_single_file_module_imports():
    assert hasattr(_drl, "learning_curve_multiplier")
    assert hasattr(_drl, "TransitTransformer")
    assert hasattr(_drl, "PPOObjectiveTracker")
    assert hasattr(_drl, "MandlNetwork")
    assert hasattr(_drl, "SimpleRoutePlanner")
    assert hasattr(_drl, "TabularQLearningAgent")
