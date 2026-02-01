# SHa

Thesis

## DRL Transformer Components

This repository provides core components for a DRL workflow that embeds transit network matrices and supports PPO-style objective aggregation.

### Transit Transformer

`sha.TransitMatrixEncoder` projects the three BTNDP matrices (edge distance, origin-destination demand, travel time) from `(N, N)` into `(N, d_model)` embeddings using `edge_embed`, `od_embed`, and `travel_time_embed` linear layers. `sha.TransitTransformer` stacks these embeddings and passes them through a Transformer encoder/decoder to produce a fused representation.

### PPO Objectives & Constraints

`sha.PPOObjectiveTracker` combines weighted objectives for maximizing coverage and minimizing demand-weighted travel time, route length, and transfers. `sha.ParetoFrontTracker` keeps a non-dominated set of multi-objective outcomes to support Pareto front analysis. `sha.RouteConstraints` validates the connectivity, route count, stop limits, and simple path requirements.
