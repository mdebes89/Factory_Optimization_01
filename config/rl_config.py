"""
Reinforcement Learning Configuration
"""

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 16
SAVE_INTERVAL = 10
LOG_INTERVAL = 1
EVAL_INTERVAL = 5

PPO_CONFIG = {
    'gamma': 0.99,
    'lr': 3e-4,
    'clip_epsilon': 0.2,
    'ppo_epochs': 4,
    'batch_size': 64,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5
}

A2C_CONFIG = {
    'gamma': 0.99,
    'lr': 3e-4,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5
}

NETWORK_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'activation': 'relu'
}
