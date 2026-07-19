"""Warehouse Class"""
from config.factory_config import RESOURCE_NAMES, CAR_REQUIREMENTS

class Warehouse:
    def __init__(self):
        self.resources = {r: 0 for r in RESOURCE_NAMES}
        self.initial_resources = self.resources.copy()
        self.total_produced = {r: 0 for r in RESOURCE_NAMES}
        self.total_consumed = {r: 0 for r in RESOURCE_NAMES}
    
    def add_resources(self, resource_type, quantity):
        if resource_type in self.resources:
            self.resources[resource_type] += quantity
            self.total_produced[resource_type] += quantity
        else:
            raise ValueError(f"Unknown resource: {resource_type}")
    
    def consume_resources(self, requirements):
        for resource, amount in requirements.items():
            if self.resources[resource] < amount:
                return False
        for resource, amount in requirements.items():
            self.resources[resource] -= amount
            self.total_consumed[resource] += amount
        return True
    
    def has_resources(self, requirements):
        for resource, amount in requirements.items():
            if self.resources[resource] < amount:
                return False
        return True
    
    def get_resource_count(self, resource_type):
        return self.resources.get(resource_type, 0)
    
    def get_state(self):
        return self.resources.copy()
    
    def reset(self):
        self.resources = self.initial_resources.copy()
        self.total_produced = {r: 0 for r in RESOURCE_NAMES}
        self.total_consumed = {r: 0 for r in RESOURCE_NAMES}
    
    def can_assemble_car(self):
        return self.has_resources(CAR_REQUIREMENTS)
