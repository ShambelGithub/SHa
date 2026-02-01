from sha.drl_transformer import (
    ParetoFrontTracker,
    PPOObjectiveTracker,
    RouteConstraints,
    TransitMatrixEncoder,
    TransitTransformer,
    MandlNetwork,
    SimpleRoutePlanner,
    EpisodeRewardTracker,
    learning_curve_multiplier,
)


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


def test_mandl_episode_reward_tracking():
    links_path = "/home/runner/work/SHa/SHa/data/mandl/mandl1_links.csv"
    demand_path = "/home/runner/work/SHa/SHa/data/mandl/mandl1_demand.csv"
    network = MandlNetwork.from_csv(links_path, demand_path, num_stops=15)
    planner = SimpleRoutePlanner(network)
    tracker = EpisodeRewardTracker()
    routes = planner.select_routes(num_routes=2, max_length=4)
    demand_served = planner.demand_served(routes)
    travel_time = sum(planner.compute_route_travel_time(route) for route in routes)
    tracker.record(demand_served - travel_time)
    assert len(tracker.values()) == 1


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
    plateau_tolerance = multiplier_range * plateau_fraction  # allow limited tail gain within final quarter of range
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
