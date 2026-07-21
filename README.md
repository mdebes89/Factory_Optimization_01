# Factory Resource Allocation Optimization using Reinforcement Learning

## Overview

This project implements a **Reinforcement Learning (RL)** solution to optimize resource allocation in a factory environment. The goal is to **maximize car production while minimizing operational costs** through intelligent agent assignment to various production stations.

The RL agent learns to make optimal decisions about:
- How many agents to assign to each station
- When to assign agents (considering battery constraints)
- How to balance production of individual components vs. complete cars
- When to use intra-day charging vs. overnight charging

## Problem Specification

### Factory Environment

The factory consists of the following production stations:

| Station | Quantity | Output Rate | Max Agents | Component Reward |
|---------|----------|-------------|------------|------------------|
| Wheels Creation | 2 stations | 1 wheel/hour/agent | 1 per station | 0.5 per wheel |
| Doors Creation | 2 stations | 1 door/hour/agent | 1 per station | 0.5 per door |
| Chassis Creation | 1 station | 1 chassis/2 hours/agent | 2 | 1 per chassis |
| Engine Creation | 1 station | 1 engine/2 hours/agent | 1 | 1 per engine |
| Car Assembly | 1 station | 1 car/4 hours/agent | 4 | 10 per car |

### Resource Requirements for Car Assembly
To produce one car, the assembly station requires:
- 4 wheels
- 4 doors
- 1 chassis
- 1 engine

All components must be available in the warehouse simultaneously for car assembly to occur.

### Agent Constraints

- **Working Hours**: Each day has **16 working hours**
- **Agent Battery Life**: 
  - Single agent can work **8 hours without interruption**
  - **Overnight charging**: Free (eco charging)
  - **Intra-day charging**: Takes **4 hours** and costs **-0.2 reward**
- **Agent Cost**: Each agent used in a day costs **-0.1 reward**
- **Unused Agents**: No cost (0 reward impact)

### Reward Structure

| Action | Reward |
|--------|--------|
| Produce Wheel | +0.5 |
| Produce Door | +0.5 |
| Produce Chassis | +1.0 |
| Produce Engine | +1.0 |
| Produce Car | +10.0 |
| Intra-day Charging | -0.2 |
| Use Agent for Day | -0.1 |

### Optimization Goals

1. **Primary Objective**: Maximize the number of cars produced
2. **Secondary Objective**: Minimize the total cost in agents used
3. **Tertiary Objective**: Efficiently balance component production to avoid bottlenecks

### Key Challenges

- **Bottleneck Management**: Car assembly requires all components in specific ratios
- **Agent Allocation**: Limited agents must be distributed across stations with different production rates
- **Battery Management**: Decide between overnight charging (free but requires planning) vs. intra-day charging (costly but flexible)
- **Production Timing**: Components must be available when needed for assembly
- **Cost Optimization**: More agents = faster production but higher costs

## Implementation Goals

The RL agent aims to learn:

1. **Optimal Agent Distribution**: How to allocate limited agents across stations to maximize car production
2. **Component Balance**: Maintain appropriate inventory levels of all components to avoid assembly bottlenecks
3. **Energy Efficiency**: Minimize intra-day charging costs through smart scheduling
4. **Cost-Effective Scaling**: Use the minimum number of agents necessary to achieve maximum car production
5. **Temporal Planning**: Schedule production activities considering the 16-hour workday constraint

## Project Structure

```
Factory_Optimization_01/
├── config/              # Configuration files for environment and training
├── env/                 # Factory environment implementation
│   ├── factory_env.py   # Main RL environment
│   └── ...
├── models/              # Neural network models
├── algorithms/          # RL algorithms (PPO, A2C, etc.)
├── utils/               # Utility functions and helpers
├── scripts/             # Training and evaluation scripts
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   └── test_env.py      # Environment testing
├── tests/               # Unit tests
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Test the Environment
```bash
python scripts/test_env.py --render
```

### Train the RL Agent
```bash
python scripts/train.py --algorithm ppo --num-agents 10 --num-episodes 1000
```

### Evaluate a Trained Model
```bash
python scripts/evaluate.py --model-path results/models/model.pth --render
```

## Mathematical Formulation

The problem can be formulated as a **Markov Decision Process (MDP)** where:

- **State (S)**: Current inventory levels, agent assignments, battery status, time of day
- **Action (A)**: Agent allocation decisions for each station
- **Reward (R)**: Net reward from production minus costs
- **Transition (P)**: State changes based on production rates and agent assignments

The RL agent learns a policy π(a|s) that maximizes the expected cumulative reward:

```
E[Σ (reward_production - reward_costs)]
```

## Success Metrics

1. **Cars Produced per Day**: Primary metric for success
2. **Cost per Car**: Total agent and charging costs divided by cars produced
3. **Component Utilization**: Percentage of produced components used in car assembly
4. **Agent Efficiency**: Cars produced per agent used

## Future Enhancements

- Multi-agent RL for coordinated agent behavior
- Hierarchical RL for strategic vs. tactical decisions
- Transfer learning for different factory configurations
- Real-time adaptation to changing demand patterns
