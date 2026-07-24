#!/usr/bin/env python3
"""Training Script with Hyperparameter Tuning Support"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from collections import deque
import time

from env.factory_env import FactoryEnv
from models.policy_network import PolicyNetwork
from algorithms.ppo import PPO
from algorithms.a2c import A2C
from config.rl_config import (
    PPO_CONFIG, PPO_BALANCED_CONFIG, PPO_STABLE_CONFIG, 
    PPO_AGGRESSIVE_CONFIG, PPO_EXPLORATION_CONFIG, PPO_LARGE_ACTION_CONFIG,
    A2C_CONFIG, A2C_BALANCED_CONFIG,
    NETWORK_CONFIG, NETWORK_LARGE_CONFIG, NETWORK_SMALL_CONFIG,
    TRAINING_PRESETS, HYPERPARAMETER_GUIDE
)

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent with hyperparameter tuning')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'a2c'])
    parser.add_argument('--num-agents', type=int, default=10)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--save-dir', type=str, default='results/models')
    parser.add_argument('--log-dir', type=str, default='results/logs')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--preset', type=str, default='balanced', 
                        choices=list(TRAINING_PRESETS.keys()),
                        help='Training preset configuration')
    parser.add_argument('--hidden-size', type=int, default=None,
                        help='Override hidden layer size')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Override number of layers')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Override discount factor')
    parser.add_argument('--clip-epsilon', type=float, default=None,
                        help='Override PPO clip epsilon')
    parser.add_argument('--entropy-coef', type=float, default=None,
                        help='Override entropy coefficient')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--monitor', action='store_true',
                        help='Enable detailed monitoring')
    return parser.parse_args()


def convert_action(action_int, num_agents):
    """Convert integer action to dictionary action for FactoryEnv"""
    action_spaces = {
        'wheel_1': 2,
        'wheel_2': 2,
        'door_1': 2,
        'door_2': 2,
        'chassis': 3,
        'engine': 2,
        'assembly': 5,
        'fast_charge': num_agents + 1
    }
    
    total_actions = 1
    for dim in action_spaces.values():
        total_actions *= dim
    
    action_dict = {}
    remaining = action_int % total_actions
    
    for key in reversed(list(action_spaces.keys())):
        dim = action_spaces[key]
        action_dict[key] = remaining % dim
        remaining = remaining // dim
    
    return action_dict


def get_network_config(args, preset_config):
    """Get network configuration, applying command-line overrides"""
    config = preset_config.get('network', NETWORK_CONFIG).copy()
    if args.hidden_size is not None:
        config['hidden_size'] = args.hidden_size
    if args.num_layers is not None:
        config['num_layers'] = args.num_layers
    return config


def get_algorithm_config(args, preset_config):
    """Get algorithm configuration, applying command-line overrides"""
    config = preset_config.get('config', {}).copy()
    
    if args.lr is not None:
        config['lr'] = args.lr
    if args.gamma is not None:
        config['gamma'] = args.gamma
    if args.clip_epsilon is not None:
        config['clip_epsilon'] = args.clip_epsilon
    if args.entropy_coef is not None:
        config['entropy_coef'] = args.entropy_coef
    
    return config


def create_policy_network(state_dim, action_dim, network_config):
    """Create policy network with specified configuration"""
    hidden_size = network_config.get('hidden_size', 128)
    num_layers = network_config.get('num_layers', 2)
    activation = network_config.get('activation', 'relu')
    
    # For now, use the existing PolicyNetwork which has fixed 2 layers
    # The hidden_size parameter is used in the network
    return PolicyNetwork(state_dim, action_dim, hidden_size)


def train_with_preset(args):
    """Train using a preset configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get preset configuration
    preset = TRAINING_PRESETS[args.preset]
    algorithm_name = preset.get('algorithm', args.algorithm)
    algorithm_config = get_algorithm_config(args, preset)
    network_config = get_network_config(args, preset)
    num_episodes = args.num_episodes or preset.get('num_episodes', 1000)
    
    print(f"\n{'='*60}")
    print(f"TRAINING WITH PRESET: {args.preset.upper()}")
    print(f"{'='*60}")
    print(f"Algorithm: {algorithm_name}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Network config: {network_config}")
    print(f"Algorithm config: {algorithm_config}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = FactoryEnv(num_agents=args.num_agents)
    state_dim = env.observation_space.shape[0]
    
    # Calculate action space size
    action_spaces = {
        'wheel_1': 2,
        'wheel_2': 2,
        'door_1': 2,
        'door_2': 2,
        'chassis': 3,
        'engine': 2,
        'assembly': 5,
        'fast_charge': args.num_agents + 1
    }
    total_actions = 1
    for dim in action_spaces.values():
        total_actions *= dim
    
    # Create policy network
    policy_network = create_policy_network(state_dim, total_actions, network_config).to(device)
    
    # Create agent
    if algorithm_name == 'ppo':
        agent = PPO(policy_network, algorithm_config)
    else:
        agent = A2C(policy_network, algorithm_config)
    
    # Set random seed if specified
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if hasattr(env, 'seed'):
            env.seed(args.seed)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Training statistics
    episode_rewards = []
    episode_losses = []
    best_reward = -float('inf')
    last_rewards = deque(maxlen=100)
    
    # Training loop
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        
        for step in range(16):
            # Get action from agent
            action_int, log_prob = agent.get_action(state)
            action_dict = convert_action(action_int, args.num_agents)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action_dict)
            
            # Get value estimate
            with torch.no_grad():
                _, value = policy_network(torch.FloatTensor(state).unsqueeze(0).to(device))
                value = value.item()
            
            # Store transition for PPO
            if algorithm_name == 'ppo':
                agent.store_transition(state, action_int, log_prob, reward, done, value)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # Perform update for PPO
        if algorithm_name == 'ppo':
            episode_loss = agent.update()
        
        # Store statistics
        last_rewards.append(episode_reward)
        episode_rewards.append(episode_reward)
        episode_losses.append(episode_loss if algorithm_name == 'ppo' else 0)
        
        # Track best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Logging
        if (episode + 1) % args.save_interval == 0:
            avg_reward = sum(last_rewards) / len(last_rewards)
            elapsed_time = time.time() - start_time
            episodes_per_sec = (episode + 1) / elapsed_time if elapsed_time > 0 else 0
            
            model_path = os.path.join(args.save_dir, f'{algorithm_name}_ep{episode+1}.pth')
            agent.save(model_path)
            
            print(f'\rEpisode {episode+1:6d}/{num_episodes:6d} | '
                  f'Avg Reward (last {len(last_rewards)}): {avg_reward:7.2f} | '
                  f'Best Reward: {best_reward:7.2f} | '
                  f'Time: {elapsed_time:6.1f}s | '
                  f'Speed: {episodes_per_sec:6.2f} ep/s | '
                  f'Model saved to {model_path}', end='', flush=True)
            
            # If monitoring is enabled, print more details on new lines
            if args.monitor:
                print()  # New line after the progress line
                print(f'  Current episode reward: {episode_reward:.2f}')
                if algorithm_name == 'ppo':
                    print(f'  Current loss: {episode_loss:.4f}')
    
    # Print newline at the end to ensure clean output
    print()
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'{algorithm_name}_final.pth')
    agent.save(final_path)
    
    # Print summary
    print(f'\nTraining complete!')
    print(f'Final model saved to {final_path}')
    print(f'Best reward achieved: {best_reward:.2f}')
    print(f'Average reward (last 100): {sum(last_rewards) / len(last_rewards):.2f}')
    print(f'Total training time: {time.time() - start_time:.1f} seconds')
    
    return episode_rewards, episode_losses


def main():
    """Main training function"""
    args = parse_args()
    
    # Print hyperparameter guide if requested
    if args.preset == 'help':
        print(HYPERPARAMETER_GUIDE)
        return
    
    # Run training
    train_with_preset(args)


if __name__ == "__main__":
    main()
