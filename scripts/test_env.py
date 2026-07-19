#!/usr/bin/env python3
"""Environment Test Script"""
import argparse
from env.factory_env import FactoryEnv

def main():
    parser = argparse.ArgumentParser(description='Test Factory Environment')
    parser.add_argument('--num-agents', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num-episodes', type=int, default=5)
    args = parser.parse_args()
    
    env = FactoryEnv(num_agents=args.num_agents)
    
    for episode in range(args.num_episodes):
        print(f'\n=== Episode {episode+1} ===')
        state = env.reset()
        
        action = {
            'wheel_1': 1, 'wheel_2': 1, 'door_1': 1, 'door_2': 1,
            'chassis': 2, 'engine': 1, 'assembly': 4, 'fast_charge': 0
        }
        
        for hour in range(16):
            next_state, reward, done, info = env.step(action)
            if args.render:
                env.render('human')
            
            if done:
                break
        
        print(f'Final warehouse: {env.warehouse.get_state()}')
        print(f'Cars produced: {env.warehouse.get_resource_count("car")}')
        for name, station in env.stations.items():
            print(f'  {name}: {station.total_produced} produced')

if __name__ == "__main__":
    main()
