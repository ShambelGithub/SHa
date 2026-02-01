# SHa

Thesis

## DRL Transformer Components

This repository provides core components for a DRL workflow that embeds transit network matrices and supports PPO-style objective aggregation.

### Transit Transformer

`sha.TransitMatrixEncoder` projects the three BTNDP matrices (edge distance, origin-destination demand, travel time) from `(N, N)` into `(N, d_model)` embeddings using `edge_embed`, `od_embed`, and `travel_time_embed` linear layers. `sha.TransitTransformer` stacks these embeddings and passes them through a Transformer encoder/decoder to produce a fused representation.

### PPO Objectives & Constraints

`sha.PPOObjectiveTracker` combines weighted objectives for maximizing coverage and minimizing demand-weighted travel time, route length, and transfers. `sha.ParetoFrontTracker` keeps a non-dominated set of multi-objective outcomes to support Pareto front analysis. `sha.RouteConstraints` validates the connectivity, route count, stop limits, and simple path requirements.

### Mandl Network Episode Tracking

`sha.MandlNetwork` loads Mandl CSV inputs, `sha.SimpleRoutePlanner` selects deterministic routes, and `sha.EpisodeRewardTracker` stores per-episode rewards for plotting. `sha.learning_curve_multiplier` shapes reward growth across episodes for a warmup, mid-training parabola, and convergence plateau. See `scripts/mandl_episode_demo.py` for generating reward curves, route travel time, and demand served plots.

### Single-File DRL Module

`drl_all_in_one.py` bundles the transformer, PPO utilities, Mandl helpers, and plotting demo into one file for easy download and execution.
