"""Environment Tests"""
import unittest
import numpy as np
from env.factory_env import FactoryEnv
from env.warehouse import Warehouse
from env.agents import AgentManager
from config.factory_config import CAR_REQUIREMENTS

class TestWarehouse(unittest.TestCase):
    def setUp(self):
        self.warehouse = Warehouse()
    
    def test_initial_state(self):
        state = self.warehouse.get_state()
        for count in state.values():
            self.assertEqual(count, 0)
    
    def test_add_and_consume(self):
        self.warehouse.add_resources('wheel', 10)
        self.assertEqual(self.warehouse.get_resource_count('wheel'), 10)
        result = self.warehouse.consume_resources({'wheel': 5})
        self.assertTrue(result)
        self.assertEqual(self.warehouse.get_resource_count('wheel'), 5)
    
    def test_car_assembly(self):
        for resource, amount in CAR_REQUIREMENTS.items():
            self.warehouse.add_resources(resource, amount)
        self.assertTrue(self.warehouse.can_assemble_car())

class TestAgentManager(unittest.TestCase):
    def setUp(self):
        self.agent_manager = AgentManager(num_agents=10)
    
    def test_initial_state(self):
        state = self.agent_manager.get_state()
        self.assertEqual(state['available'], 10)
        self.assertEqual(state['working'], 0)
    
    def test_assign_work(self):
        assigned = self.agent_manager.assign_to_work(5)
        self.assertEqual(assigned, 5)
        state = self.agent_manager.get_state()
        self.assertEqual(state['working'], 5)
    
    def test_overnight_charge(self):
        self.agent_manager.assign_to_work(8)
        self.agent_manager.end_work_hour()
        self.agent_manager.overnight_charge()
        state = self.agent_manager.get_state()
        self.assertEqual(state['available'], 10)

class TestFactoryEnv(unittest.TestCase):
    def setUp(self):
        self.env = FactoryEnv(num_agents=10)
    
    def test_reset(self):
        state = self.env.reset()
        self.assertIsInstance(state, np.ndarray)
    
    def test_step(self):
        state = self.env.reset()
        action = {s: 0 for s in ['wheel_1', 'wheel_2', 'door_1', 'door_2', 'chassis', 'engine', 'assembly', 'fast_charge']}
        next_state, reward, done, info = self.env.step(action)
        self.assertFalse(done)

if __name__ == '__main__':
    unittest.main()
