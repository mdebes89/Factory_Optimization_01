"""Station Classes"""
from config.factory_config import (
    WHEEL_PRODUCTION_RATE, DOOR_PRODUCTION_RATE,
    CHASSIS_PRODUCTION_RATE, ENGINE_PRODUCTION_RATE,
    ASSEMBLY_PRODUCTION_RATE, CAR_REQUIREMENTS,
    WHEEL_STATION_CAPACITY, DOOR_STATION_CAPACITY,
    CHASSIS_STATION_CAPACITY, ENGINE_STATION_CAPACITY,
    ASSEMBLY_STATION_CAPACITY
)

class Station:
    def __init__(self, name, production_rate, max_agents, output_resource):
        self.name = name
        self.production_rate = production_rate
        self.max_agents = max_agents
        self.output_resource = output_resource
        self.assigned_agents = 0
        self.time_since_last_production = 0
        self.total_produced = 0
    
    def assign_agents(self, num_agents):
        self.assigned_agents = min(max(0, num_agents), self.max_agents)
    
    def produce(self):
        if self.assigned_agents == 0:
            self.time_since_last_production = 0
            return None
        self.time_since_last_production += 1
        if self.time_since_last_production >= self.production_rate:
            units = self.assigned_agents
            self.time_since_last_production = 0
            self.total_produced += units
            return {self.output_resource: units}
        return None
    
    def reset(self):
        self.assigned_agents = 0
        self.time_since_last_production = 0
    
    def get_state(self):
        return {
            'name': self.name,
            'assigned_agents': self.assigned_agents,
            'time_since_last_production': self.time_since_last_production,
            'max_agents': self.max_agents,
            'production_rate': self.production_rate,
            'total_produced': self.total_produced
        }

class WheelStation(Station):
    def __init__(self, station_id):
        super().__init__(f"wheel_{station_id}", WHEEL_PRODUCTION_RATE, WHEEL_STATION_CAPACITY, "wheel")

class DoorStation(Station):
    def __init__(self, station_id):
        super().__init__(f"door_{station_id}", DOOR_PRODUCTION_RATE, DOOR_STATION_CAPACITY, "door")

class ChassisStation(Station):
    def __init__(self):
        super().__init__("chassis", CHASSIS_PRODUCTION_RATE, CHASSIS_STATION_CAPACITY, "chassis")

class EngineStation(Station):
    def __init__(self):
        super().__init__("engine", ENGINE_PRODUCTION_RATE, ENGINE_STATION_CAPACITY, "engine")

class AssemblyStation(Station):
    def __init__(self):
        super().__init__("assembly", ASSEMBLY_PRODUCTION_RATE, ASSEMBLY_STATION_CAPACITY, "car")
        self.car_requirements = CAR_REQUIREMENTS.copy()
        self.can_produce = False
    
    def produce(self, warehouse):
        if self.assigned_agents == 0:
            self.time_since_last_production = 0
            self.can_produce = False
            return None
        
        # Check if we have enough resources to assemble
        if not warehouse.has_resources(self.car_requirements):
            # No resources available, agents are idle
            self.can_produce = False
            return None
        
        self.can_produce = True
        self.time_since_last_production += 1
        effective_rate = max(1, self.production_rate - self.assigned_agents + 1)
        if self.time_since_last_production >= effective_rate:
            if warehouse.has_resources(self.car_requirements):
                warehouse.consume_resources(self.car_requirements)
                self.time_since_last_production = 0
                self.total_produced += 1
                return {"car": 1}
        return None
    
    def get_effective_rate(self):
        if self.assigned_agents == 0:
            return float('inf')
        return max(1, self.production_rate - self.assigned_agents + 1)
    
    def can_produce_this_hour(self, warehouse):
        """Check if this station can produce given current warehouse state"""
        return self.assigned_agents > 0 and warehouse.has_resources(self.car_requirements)

def create_stations():
    return {
        'wheel_1': WheelStation(1),
        'wheel_2': WheelStation(2),
        'door_1': DoorStation(1),
        'door_2': DoorStation(2),
        'chassis': ChassisStation(),
        'engine': EngineStation(),
        'assembly': AssemblyStation()
    }
