#!/usr/bin/env python3
"""Training Script"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np

from env.factory_env import FactoryEnv
from models.policy_network import PolicyNetwork
from algorithms.ppo import PPO
from algorithms.a2c import A2C
from config.rl_config import PPO_CONFIG, A2C_CONFIG

def parse_args():
    parser = argparse.ArgumentParser(description='Train RL agent')
    parser.add_argument('--algorithm', type=str, default='ppo', choices=['ppo', 'a2c'])
    parser.add_argument('--num-agents', type=int, default=10)
    parser.add_argument('--num-episodes', type=int, default=1000)
    parser.add_argument('--save-dir', type=str, default='results/models')
    parser.add_argument('--log-dir', type=str, default='results/logs')
    parser.add_argument('--save-interval', type=int, default=10)
    return parser.parse_args()

def convert_action(action_int, num_agents):
    """Convert integer action to dictionary action for FactoryEnv"""
    # Define the action space dimensions
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
    
    # Calculate total number of actions
    total_actions = 1
    for dim in action_spaces.values():
        total_actions *= dim
    
    # Convert integer to multi-dimensional index
    action_dict = {}
    remaining = action_int % total_actions
    
    # Process in reverse order for easier division
    for key in reversed(list(action_spaces.keys())):
        dim = action_spaces[key]
        action_dict[key] = remaining % dim
        remaining = remaining // dim
    
    return action_dict

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = FactoryEnv(num_agents=args.num_agents)
    state_dim = env.observation_space.shape[0]
    
    # Calculate total number of possible actions
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
    
    policy_network = PolicyNetwork(state_dim, total_actions).to(device)
    
    if args.algorithm == 'ppo':
        agent = PPO(policy_network, PPO_CONFIG)
    else:
        agent = A2C(policy_network, A2C_CONFIG)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(16):
            action_int, _ = agent.get_action(state)
            action_dict = convert_action(action_int, args.num_agents)
            next_state, reward, done, info = env.step(action_dict)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        if (episode + 1) % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, f'{args.algorithm}_ep{episode+1}.pth')
            agent.save(model_path)
            print(f'Episode {episode+1}: Reward = {episode_reward:.2f}, Model saved to {model_path}')
        else:
            print(f'Episode {episode+1}: Reward = {episode_reward:.2f}')
    
    final_path = os.path.join(args.save_dir, f'{args.algorithm}_final.pth')
    agent.save(final_path)
    print(f'Training complete! Final model saved to {final_path}')

if __name__ == "__main__":
    main()
