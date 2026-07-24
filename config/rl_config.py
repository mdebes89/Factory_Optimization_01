"""
Reinforcement Learning Configuration

Hyperparameter tuning presets for different scenarios:
- DEFAULT: Original configuration (baseline)
- STABLE: More stable training with conservative updates
- AGGRESSIVE: Faster learning with higher risk of instability
- BALANCED: Recommended starting point for this environment
- EXPLORATION: Focus on exploration and policy diversity
"""

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 16
SAVE_INTERVAL = 10
LOG_INTERVAL = 1
EVAL_INTERVAL = 5

# ============================================================================
# PPO Hyperparameter Configurations
# ============================================================================

# Original configuration (baseline)
PPO_CONFIG = {
    'gamma': 0.99,
    'lr': 3e-4,
    'clip_epsilon': 0.2,
    'ppo_epochs': 4,
    'batch_size': 64,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'gae_lambda': 0.95,
    'num_mini_batches': 4
}

# Stable training configuration
# - Lower learning rate for stability
# - Smaller clip epsilon
# - More epochs for better convergence
# - Higher value loss coefficient
PPO_STABLE_CONFIG = {
    'gamma': 0.99,
    'lr': 1e-4,
    'clip_epsilon': 0.1,
    'ppo_epochs': 8,
    'batch_size': 64,
    'value_loss_coef': 0.75,
    'entropy_coef': 0.005,
    'max_grad_norm': 0.5,
    'gae_lambda': 0.95,
    'num_mini_batches': 4
}

# Aggressive learning configuration
# - Higher learning rate for faster convergence
# - Larger clip epsilon
# - Lower value loss coefficient
# - Higher entropy for exploration
PPO_AGGRESSIVE_CONFIG = {
    'gamma': 0.99,
    'lr': 1e-3,
    'clip_epsilon': 0.3,
    'ppo_epochs': 4,
    'batch_size': 64,
    'value_loss_coef': 0.25,
    'entropy_coef': 0.05,
    'max_grad_norm': 1.0,
    'gae_lambda': 0.95,
    'num_mini_batches': 2
}

# Balanced configuration (RECOMMENDED for this environment)
# - Moderate learning rate
# - Balanced clip epsilon
# - Good mix of value and policy loss
# - Adaptive entropy coefficient
PPO_BALANCED_CONFIG = {
    'gamma': 0.99,
    'lr': 5e-4,
    'clip_epsilon': 0.2,
    'ppo_epochs': 6,
    'batch_size': 64,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.02,
    'max_grad_norm': 0.8,
    'gae_lambda': 0.95,
    'num_mini_batches': 4
}

# Exploration-focused configuration
# - Higher entropy coefficient to encourage exploration
# - Lower value loss to focus on policy improvement
# - Slightly higher learning rate
PPO_EXPLORATION_CONFIG = {
    'gamma': 0.99,
    'lr': 5e-4,
    'clip_epsilon': 0.25,
    'ppo_epochs': 4,
    'batch_size': 64,
    'value_loss_coef': 0.3,
    'entropy_coef': 0.1,
    'max_grad_norm': 0.5,
    'gae_lambda': 0.95,
    'num_mini_batches': 4
}

# Configuration for large action spaces (2640 actions in this environment)
# - Larger network capacity
# - More epochs for better learning
# - Higher entropy to explore the large action space
PPO_LARGE_ACTION_CONFIG = {
    'gamma': 0.99,
    'lr': 3e-4,
    'clip_epsilon': 0.2,
    'ppo_epochs': 8,
    'batch_size': 128,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.05,
    'max_grad_norm': 0.5,
    'gae_lambda': 0.95,
    'num_mini_batches': 8
}

# ============================================================================
# A2C Hyperparameter Configurations
# ============================================================================

A2C_CONFIG = {
    'gamma': 0.99,
    'lr': 3e-4,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5
}

A2C_STABLE_CONFIG = {
    'gamma': 0.99,
    'lr': 1e-4,
    'value_loss_coef': 0.75,
    'entropy_coef': 0.005,
    'max_grad_norm': 0.5
}

A2C_BALANCED_CONFIG = {
    'gamma': 0.99,
    'lr': 5e-4,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.02,
    'max_grad_norm': 0.8
}

# ============================================================================
# Network Architecture Configurations
# ============================================================================

# Original network configuration
NETWORK_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'activation': 'relu'
}

# Larger network for complex environments
NETWORK_LARGE_CONFIG = {
    'hidden_size': 256,
    'num_layers': 3,
    'activation': 'relu'
}

# Smaller, faster network
NETWORK_SMALL_CONFIG = {
    'hidden_size': 64,
    'num_layers': 2,
    'activation': 'relu'
}

# Network with different activation
NETWORK_TANH_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'activation': 'tanh'
}

# ============================================================================
# Training Presets
# ============================================================================

# Preset configurations combining algorithm, hyperparameters, and network
TRAINING_PRESETS = {
    'default': {
        'algorithm': 'ppo',
        'config': PPO_CONFIG,
        'network': NETWORK_CONFIG,
        'num_episodes': 1000
    },
    'stable': {
        'algorithm': 'ppo',
        'config': PPO_STABLE_CONFIG,
        'network': NETWORK_CONFIG,
        'num_episodes': 2000
    },
    'aggressive': {
        'algorithm': 'ppo',
        'config': PPO_AGGRESSIVE_CONFIG,
        'network': NETWORK_CONFIG,
        'num_episodes': 500
    },
    'balanced': {
        'algorithm': 'ppo',
        'config': PPO_BALANCED_CONFIG,
        'network': NETWORK_LARGE_CONFIG,
        'num_episodes': 1500
    },
    'exploration': {
        'algorithm': 'ppo',
        'config': PPO_EXPLORATION_CONFIG,
        'network': NETWORK_CONFIG,
        'num_episodes': 1000
    },
    'large_action': {
        'algorithm': 'ppo',
        'config': PPO_LARGE_ACTION_CONFIG,
        'network': NETWORK_LARGE_CONFIG,
        'num_episodes': 2000
    }
}

# ============================================================================
# Hyperparameter Tuning Guide
# ============================================================================

HYPERPARAMETER_GUIDE = """
PPO Hyperparameter Tuning Guide for Factory Optimization:

1. LEARNING RATE (lr):
   - Too high: Training unstable, reward oscillates
   - Too low: Slow learning, takes many episodes
   - Recommended range: 1e-4 to 1e-3
   - Start with: 3e-4 or 5e-4

2. CLIP EPSILON (clip_epsilon):
   - Controls how much policy can change per update
   - Too high: Large updates, potential instability
   - Too low: Very slow learning
   - Recommended range: 0.1 to 0.3
   - Start with: 0.2

3. PPO EPOCHS (ppo_epochs):
   - Number of optimization epochs per batch
   - More epochs: Better convergence but slower
   - Fewer epochs: Faster but may not converge as well
   - Recommended range: 4 to 10
   - Start with: 4-6

4. VALUE LOSS COEFFICIENT (value_loss_coef):
   - Balances policy loss vs value function loss
   - Higher: Better value estimates, may slow policy learning
   - Lower: Faster policy learning, value estimates may be less accurate
   - Recommended range: 0.25 to 1.0
   - Start with: 0.5

5. ENTROPY COEFFICIENT (entropy_coef):
   - Encourages exploration
   - Higher: More exploration, slower convergence
   - Lower: Less exploration, may get stuck in local optima
   - Recommended range: 0.001 to 0.1
   - Start with: 0.01-0.05

6. GAMMA (gamma):
   - Discount factor for future rewards
   - Higher: Agent considers long-term rewards more
   - Lower: Agent focuses on immediate rewards
   - Recommended: 0.99 (for 16-step episodes)

7. GAE LAMBDA (gae_lambda):
   - Controls bias-variance tradeoff in advantage estimation
   - Higher: Lower bias, higher variance
   - Lower: Higher bias, lower variance
   - Recommended: 0.95

8. MAX GRAD NORM (max_grad_norm):
   - Gradient clipping for stability
   - Recommended: 0.5 to 1.0

For this environment (16 steps, 2640 actions):
- Use PPO_BALANCED_CONFIG or PPO_LARGE_ACTION_CONFIG
- Network: NETWORK_LARGE_CONFIG (256 units, 3 layers)
- num_episodes: 1500-2000
"""
