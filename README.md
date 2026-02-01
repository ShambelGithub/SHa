# SHa

Thesis: DRL developer for optimization of bus transit network design.

## Overview

This repository provides a minimal Deep Reinforcement Learning (DRL) scaffold for
bus transit network design. The current implementation includes:

- A lightweight environment that evaluates route coverage, unmet demand, cost,
  and balance penalties.
- A simple greedy agent that serves as a baseline for DRL optimization.
- A training entry point you can expand with advanced DRL algorithms.

## Quick start

```bash
python -m bus_network.training
```

## Testing

```bash
python -m unittest
```
