"""Factory Environment"""
import gym
import numpy as np
from gym import spaces
from .stations import create_stations
from .warehouse import Warehouse
from .agents import AgentManager
from config.factory_config import (
    WORKING_HOURS_PER_DAY, DEFAULT_NUM_AGENTS,
    REWARDS, AGENT_USAGE_COST, FAST_CHARGE_COST,
    ALL_STATION_NAMES, CAR_REQUIREMENTS
)

class FactoryEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, num_agents=DEFAULT_NUM_AGENTS):
        super(FactoryEnv, self).__init__()
        self.num_agents = num_agents
        self.warehouse = Warehouse()
        self.agent_manager = AgentManager(num_agents)
        self.stations = create_stations()
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()
        self.current_hour = 0
        self.day_complete = False
        self.episode_reward = 0
        self.new_production = {k: 0 for k in ['wheel', 'door', 'chassis', 'engine', 'car']}
        self.agents_used_this_hour = 0
        self.fast_charges_this_hour = 0
    
    def _create_action_space(self):
        return spaces.Dict({
            'wheel_1': spaces.Discrete(2),
            'wheel_2': spaces.Discrete(2),
            'door_1': spaces.Discrete(2),
            'door_2': spaces.Discrete(2),
            'chassis': spaces.Discrete(3),
            'engine': spaces.Discrete(2),
            'assembly': spaces.Discrete(5),
            'fast_charge': spaces.Discrete(self.num_agents + 1)
        })
    
    def _create_observation_space(self):
        obs_size = 1 + 5 + self.num_agents + len(self.stations) * 2
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
    
    def _get_observation(self):
        observation = []
        observation.append(self.current_hour / WORKING_HOURS_PER_DAY)
        for resource in ['wheel', 'door', 'chassis', 'engine', 'car']:
            observation.append(self.warehouse.get_resource_count(resource) / 100.0)
        for battery in self.agent_manager.battery_levels:
            observation.append(battery)
        for station_name in ALL_STATION_NAMES:
            station = self.stations[station_name]
            observation.append(station.assigned_agents / station.max_agents)
            observation.append(station.time_since_last_production / station.production_rate)
        return np.array(observation, dtype=np.float32)
    
    def reset(self):
        self.warehouse.reset()
        self.agent_manager.overnight_charge()
        for station in self.stations.values():
            station.reset()
        self.current_hour = 0
        self.day_complete = False
        self.episode_reward = 0
        self.new_production = {k: 0 for k in self.new_production.keys()}
        return self._get_observation()
    
    def step(self, action):
        if self.day_complete:
            raise RuntimeError("Day is complete, call reset()")
        self.new_production = {k: 0 for k in self.new_production.keys()}
        self.agents_used_this_hour = 0
        self.fast_charges_this_hour = 0
        self._process_action(action)
        self.current_hour += 1
        self._produce_resources()
        self._assemble_cars()
        self.agent_manager.end_work_hour()
        reward = self._calculate_reward()
        self.episode_reward += reward
        done = self.current_hour >= WORKING_HOURS_PER_DAY
        if done:
            self.day_complete = True
        return self._get_observation(), reward, done, {
            'hour': self.current_hour,
            'production': self.new_production.copy(),
            'agents_used': self.agents_used_this_hour,
            'fast_charges': self.fast_charges_this_hour,
            'warehouse': self.warehouse.get_state()
        }
    
    def _process_action(self, action):
        total_assigned = 0
        for station_name in ['wheel_1', 'wheel_2', 'door_1', 'door_2', 'chassis', 'engine', 'assembly']:
            if station_name in action:
                self.stations[station_name].assign_agents(action[station_name])
                total_assigned += action[station_name]
        if 'fast_charge' in action:
            self.fast_charges_this_hour = self.agent_manager.assign_to_charge(action['fast_charge'])
        self.agents_used_this_hour = self.agent_manager.assign_to_work(total_assigned)
    
    def _produce_resources(self):
        for name, station in self.stations.items():
            if name != 'assembly':
                produced = station.produce()
                if produced:
                    for resource, amount in produced.items():
                        self.warehouse.add_resources(resource, amount)
                        self.new_production[resource] += amount
    
    def _assemble_cars(self):
        assembly = self.stations['assembly']
        if assembly.assigned_agents == 0:
            assembly.time_since_last_production = 0
            return
        produced = assembly.produce(self.warehouse)
        if produced:
            for resource, amount in produced.items():
                self.warehouse.add_resources(resource, amount)
                self.new_production[resource] += amount
    
    def _calculate_reward(self):
        reward = 0.0
        reward += self.new_production['wheel'] * REWARDS['wheel']
        reward += self.new_production['door'] * REWARDS['door']
        reward += self.new_production['chassis'] * REWARDS['chassis']
        reward += self.new_production['engine'] * REWARDS['engine']
        reward += self.new_production['car'] * REWARDS['car']
        reward += self.agents_used_this_hour * AGENT_USAGE_COST
        reward += self.fast_charges_this_hour * FAST_CHARGE_COST
        return reward
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\n=== Hour {self.current_hour} ===")
            print(f"Reward: {self._calculate_reward():.2f}")
            print("Warehouse:", self.warehouse.get_state())
            print(self.agent_manager)
    
    def close(self):
        pass
