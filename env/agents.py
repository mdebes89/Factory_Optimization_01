"""Agent Manager"""
from config.factory_config import AGENT_BATTERY_CAPACITY, FAST_CHARGE_TIME, DEFAULT_NUM_AGENTS

class AgentManager:
    def __init__(self, num_agents=DEFAULT_NUM_AGENTS):
        self.total_agents = num_agents
        self.available = num_agents
        self.working = 0
        self.charging = 0
        self.battery_levels = [1.0] * num_agents
        self.working_hours = [0] * num_agents
        self.charging_progress = [0.0] * num_agents
        self.agent_status = ['available'] * num_agents
        self.agent_station = [None] * num_agents  # Track which station each agent is assigned to
    
    def assign_to_work(self, num_agents):
        available_indices = [i for i in range(self.total_agents) 
                          if self.agent_status[i] == 'available' and self.battery_levels[i] > 0]
        num_to_assign = min(num_agents, len(available_indices))
        for i in range(num_to_assign):
            idx = available_indices[i]
            self.agent_status[idx] = 'working'
            self.battery_levels[idx] = max(0, self.battery_levels[idx] - 1/AGENT_BATTERY_CAPACITY)
        self.available -= num_to_assign
        self.working += num_to_assign
        return num_to_assign
    
    def assign_to_charge(self, num_agents):
        chargeable = [i for i in range(self.total_agents) 
                     if self.agent_status[i] in ['available', 'working'] and self.battery_levels[i] < 1.0]
        num_to_charge = min(num_agents, len(chargeable))
        for i in range(num_to_charge):
            idx = chargeable[i]
            if self.agent_status[idx] == 'working':
                self.working -= 1
                self.available += 1
            self.agent_status[idx] = 'charging'
            self.agent_station[idx] = 'fast_charge'
            self.charging_progress[idx] = 0.0
        self.available -= num_to_charge
        self.charging += num_to_charge
        return num_to_charge
    
    def assign_agent_to_station(self, agent_id, station_name):
        """Assign a specific agent to a station"""
        if agent_id < self.total_agents:
            self.agent_station[agent_id] = station_name
    
    def get_agent_allocations(self):
        """Get a dictionary mapping station names to list of agent IDs assigned to them"""
        allocations = {}
        for agent_id in range(self.total_agents):
            station = self.agent_station[agent_id]
            if station is not None:
                if station not in allocations:
                    allocations[station] = []
                allocations[station].append(agent_id)
        return allocations
    
    def clear_agent_allocations(self):
        """Clear all agent station assignments"""
        self.agent_station = [None] * self.total_agents
    
    def progress_charging(self):
        for i in range(self.total_agents):
            if self.agent_status[i] == 'charging':
                self.charging_progress[i] += 1/FAST_CHARGE_TIME
                self.battery_levels[i] = min(1.0, self.battery_levels[i] + 1/FAST_CHARGE_TIME)
                if self.charging_progress[i] >= 1.0 or self.battery_levels[i] >= 1.0:
                    self.agent_status[i] = 'available'
                    self.agent_station[i] = None
                    self.charging_progress[i] = 0.0
        self._recalculate_counts()
    
    def overnight_charge(self):
        self.available = self.total_agents
        self.working = 0
        self.charging = 0
        for i in range(self.total_agents):
            self.agent_status[i] = 'available'
            self.battery_levels[i] = 1.0
            self.working_hours[i] = 0
            self.charging_progress[i] = 0.0
            self.agent_station[i] = None
    
    def end_work_hour(self):
        for i in range(self.total_agents):
            if self.agent_status[i] == 'working':
                self.working_hours[i] += 1
                self.battery_levels[i] = max(0, self.battery_levels[i] - 1/AGENT_BATTERY_CAPACITY)
                if self.battery_levels[i] <= 0:
                    self.agent_status[i] = 'available'
                    self.agent_station[i] = None
        self.progress_charging()
        self._recalculate_counts()
    
    def _recalculate_counts(self):
        self.available = sum(1 for s in self.agent_status if s == 'available')
        self.working = sum(1 for s in self.agent_status if s == 'working')
        self.charging = sum(1 for s in self.agent_status if s == 'charging')
    
    def get_state(self):
        return {
            'total_agents': self.total_agents,
            'available': self.available,
            'working': self.working,
            'charging': self.charging,
            'battery_levels': self.battery_levels.copy()
        }
    
    def reset(self):
        self.overnight_charge()
    
    def __str__(self):
        return f"AgentManager(total={self.total_agents}, available={self.available}, working={self.working}, charging={self.charging})"
