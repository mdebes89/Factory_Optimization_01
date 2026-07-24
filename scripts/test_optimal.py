#!/usr/bin/env python3
"""
Test script to calculate the mathematical optimal reward for the factory environment.
This helps understand the upper bound of what the RL agent should achieve.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.factory_env import FactoryEnv
from config.factory_config import (
    WORKING_HOURS_PER_DAY, DEFAULT_NUM_AGENTS,
    REWARDS, AGENT_USAGE_COST, FAST_CHARGE_COST,
    CAR_REQUIREMENTS
)


def test_greedy_allocation():
    """Test greedy allocation strategy to find near-optimal reward"""
    print("=" * 60)
    print("TESTING GREEDY ALLOCATION STRATEGY")
    print("=" * 60)
    
    env = FactoryEnv(num_agents=DEFAULT_NUM_AGENTS)
    state = env.reset()
    
    total_reward = 0
    total_production = {k: 0 for k in ['wheel', 'door', 'chassis', 'engine', 'car']}
    
    for hour in range(WORKING_HOURS_PER_DAY):
        # Greedy strategy: allocate agents to maximize component production
        # Priority: engine (bottleneck), chassis, wheels, doors, then assembly
        
        # We have 10 agents to allocate
        # Engine station: 1 agent -> 0.5 engines/hour (bottleneck!)
        # Chassis station: 2 agents -> 1 chassis/hour
        # Wheel stations: 1 agent each -> 2 wheels/hour
        # Door stations: 1 agent each -> 2 doors/hour
        # Assembly: remaining agents
        
        action = {
            'wheel_1': 1,
            'wheel_2': 1,
            'door_1': 1,
            'door_2': 1,
            'chassis': 2,
            'engine': 1,
            'assembly': 4,  # Use remaining 4 agents (10 - 1 - 1 - 1 - 1 - 2 = 4)
            'fast_charge': 0
        }
        
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        for resource in total_production:
            total_production[resource] += info['production'].get(resource, 0)
        
        print(f"Hour {hour + 1:2d}: Reward={reward:7.2f} | "
              f"Wheels={info['production']['wheel']:2d} | "
              f"Doors={info['production']['door']:2d} | "
              f"Chassis={info['production']['chassis']:2d} | "
              f"Engines={info['production']['engine']:2d} | "
              f"Cars={info['production']['car']:2d} | "
              f"Agents used={info['agents_used']:2d}")
    
    print("\n" + "=" * 60)
    print("TOTAL RESULTS (Greedy Strategy)")
    print("=" * 60)
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Production:")
    for resource, count in total_production.items():
        reward_value = count * REWARDS[resource]
        print(f"  {resource:10s}: {count:3d} (reward: {reward_value:6.1f})")
    
    # Calculate theoretical maximum
    print("\n" + "=" * 60)
    print("THEORETICAL MAXIMUM CALCULATION")
    print("=" * 60)
    
    # Calculate max possible production given station constraints
    max_wheels = 2 * WORKING_HOURS_PER_DAY  # 2 stations * 1 wheel/hour
    max_doors = 2 * WORKING_HOURS_PER_DAY    # 2 stations * 1 door/hour
    max_chassis = 1 * WORKING_HOURS_PER_DAY  # 1 station * 1 chassis/hour (with 2 agents)
    max_engines = 0.5 * WORKING_HOURS_PER_DAY  # 1 station * 0.5 engine/hour
    
    # Cars limited by the scarcest component
    max_cars_by_component = {
        'wheel': max_wheels // CAR_REQUIREMENTS['wheel'],
        'door': max_doors // CAR_REQUIREMENTS['door'],
        'chassis': max_chassis // CAR_REQUIREMENTS['chassis'],
        'engine': max_engines // CAR_REQUIREMENTS['engine']
    }
    
    max_possible_cars = min(max_cars_by_component.values())
    
    print(f"Max wheels: {max_wheels} (enough for {max_cars_by_component['wheel']} cars)")
    print(f"Max doors: {max_doors} (enough for {max_cars_by_component['door']} cars)")
    print(f"Max chassis: {max_chassis} (enough for {max_cars_by_component['chassis']} cars)")
    print(f"Max engines: {max_engines} (enough for {max_cars_by_component['engine']} cars)")
    print(f"\nBottleneck: {min(max_cars_by_component, key=max_cars_by_component.get)}")
    print(f"Maximum possible cars: {max_possible_cars}")
    
    # Calculate theoretical max reward
    theoretical_max = 0
    theoretical_max += max_wheels * REWARDS['wheel']
    theoretical_max += max_doors * REWARDS['door']
    theoretical_max += max_chassis * REWARDS['chassis']
    theoretical_max += max_engines * REWARDS['engine']
    theoretical_max += max_possible_cars * REWARDS['car']
    
    # Subtract agent costs (assuming ~7 agents used per hour)
    agents_per_hour = 1 + 1 + 1 + 1 + 2 + 1 + 4  # wheel1, wheel2, door1, door2, chassis, engine, assembly
    theoretical_max += agents_per_hour * WORKING_HOURS_PER_DAY * AGENT_USAGE_COST
    
    print(f"\nTheoretical maximum reward: {theoretical_max:.2f}")
    print(f"Greedy strategy achieved: {total_reward:.2f}")
    print(f"Efficiency: {(total_reward / theoretical_max * 100):.1f}%")
    
    return total_reward, theoretical_max


def test_bottleneck_analysis():
    """Analyze which station is the bottleneck"""
    print("\n" + "=" * 60)
    print("BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    hours = WORKING_HOURS_PER_DAY
    
    # Production rates with optimal agent allocation
    wheel_rate = 2  # 2 stations * 1 wheel/hour each
    door_rate = 2   # 2 stations * 1 door/hour each
    chassis_rate = 1  # 1 station * 1 chassis/hour (with 2 agents, rate=2 hours -> 0.5/hour * 2 agents = 1/hour)
    engine_rate = 0.5  # 1 station * 0.5 engine/hour (rate=2 hours)
    
    print(f"Production rates (per hour):")
    print(f"  Wheels: {wheel_rate}/hour")
    print(f"  Doors: {door_rate}/hour")
    print(f"  Chassis: {chassis_rate}/hour")
    print(f"  Engines: {engine_rate}/hour")
    
    print(f"\nIn {hours} hours:")
    print(f"  Wheels: {wheel_rate * hours}")
    print(f"  Doors: {door_rate * hours}")
    print(f"  Chassis: {chassis_rate * hours}")
    print(f"  Engines: {engine_rate * hours}")
    
    print(f"\nCar requirements: {CAR_REQUIREMENTS}")
    print(f"Cars limited by wheels: {wheel_rate * hours // CAR_REQUIREMENTS['wheel']}")
    print(f"Cars limited by doors: {door_rate * hours // CAR_REQUIREMENTS['door']}")
    print(f"Cars limited by chassis: {chassis_rate * hours // CAR_REQUIREMENTS['chassis']}")
    print(f"Cars limited by engines: {engine_rate * hours // CAR_REQUIREMENTS['engine']}")
    
    # The engine is the bottleneck!
    max_cars = engine_rate * hours // CAR_REQUIREMENTS['engine']
    print(f"\n*** ENGINE STATION IS THE BOTTLENECK ***")
    print(f"Maximum cars per day: {max_cars}")


def test_reward_calculation():
    """Test the reward calculation logic"""
    print("\n" + "=" * 60)
    print("REWARD CALCULATION TEST")
    print("=" * 60)
    
    env = FactoryEnv(num_agents=DEFAULT_NUM_AGENTS)
    
    # Test with known production values
    env.new_production = {'wheel': 2, 'door': 2, 'chassis': 1, 'engine': 0, 'car': 0}
    env.agents_used_this_hour = 7
    env.fast_charges_this_hour = 0
    
    reward = env._calculate_reward()
    expected = 2 * 0.5 + 2 * 0.5 + 1 * 1.0 + 0 * 1.0 + 0 * 10.0 + 7 * (-0.1) + 0 * (-0.2)
    expected = 1 + 1 + 1 - 0.7
    
    print(f"Production: wheels=2, doors=2, chassis=1, engines=0, cars=0")
    print(f"Agents used: 7")
    print(f"Calculated reward: {reward:.2f}")
    print(f"Expected reward: {expected:.2f}")
    print(f"Match: {abs(reward - expected) < 0.01}")


def main():
    """Run all tests"""
    print("Factory Optimization - Optimal Reward Test")
    print("=" * 60)
    
    # Test reward calculation
    test_reward_calculation()
    
    # Test bottleneck analysis
    test_bottleneck_analysis()
    
    # Test greedy allocation
    greedy_reward, theoretical_max = test_greedy_allocation()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Greedy strategy reward: {greedy_reward:.2f}")
    print(f"Theoretical maximum: {theoretical_max:.2f}")
    print(f"\nYour RL agent should aim for: ~{theoretical_max:.0f} reward per episode")
    print(f"Current performance (25-30): {greedy_reward / theoretical_max * 100:.1f}% of optimal")
    print("\nThe gap indicates significant room for improvement!")


if __name__ == "__main__":
    main()
