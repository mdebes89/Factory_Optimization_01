# Factory Resource Allocation Optimization using Reinforcement Learning

## Overview
RL-based optimization for factory resource allocation to maximize car production while minimizing costs.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Test environment
python scripts/test_env.py --render

# Train agent
python scripts/train.py --algorithm ppo --num-agents 10 --num-episodes 1000

# Evaluate model
python scripts/evaluate.py --model-path results/models/model.pth --render
```

## Project Structure
- `config/`: Configuration files
- `env/`: Factory environment components
- `models/`: Neural network models
- `algorithms/`: RL algorithms (PPO, A2C)
- `utils/`: Utilities and helpers
- `scripts/`: Training and evaluation scripts
- `tests/`: Unit tests
