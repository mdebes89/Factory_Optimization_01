"""
Factory Configuration Constants
"""

# Factory Stations Configuration
WHEEL_STATIONS = 2
DOOR_STATIONS = 2
CHASSIS_STATIONS = 1
ENGINE_STATIONS = 1
ASSEMBLY_STATIONS = 1

# Production Rates (hours per unit)
WHEEL_PRODUCTION_RATE = 1
DOOR_PRODUCTION_RATE = 1
CHASSIS_PRODUCTION_RATE = 2
ENGINE_PRODUCTION_RATE = 2
ASSEMBLY_PRODUCTION_RATE = 4

# Station Capacities
WHEEL_STATION_CAPACITY = 1
DOOR_STATION_CAPACITY = 1
CHASSIS_STATION_CAPACITY = 2
ENGINE_STATION_CAPACITY = 1
ASSEMBLY_STATION_CAPACITY = 4

# Car Assembly Requirements
CAR_REQUIREMENTS = {
    'wheel': 4,
    'door': 4,
    'chassis': 1,
    'engine': 1
}

# Working Day
WORKING_HOURS_PER_DAY = 16

# Agent Configuration
AGENT_BATTERY_CAPACITY = 8
FAST_CHARGE_TIME = 4
FAST_CHARGE_COST = -0.2

# Rewards
REWARDS = {
    'wheel': 0.5,
    'door': 0.5,
    'chassis': 1.0,
    'engine': 1.0,
    'car': 10.0
}

AGENT_USAGE_COST = -0.1
DEFAULT_NUM_AGENTS = 10
RESOURCE_NAMES = ['wheel', 'door', 'chassis', 'engine', 'car']
ALL_STATION_NAMES = ['wheel_1', 'wheel_2', 'door_1', 'door_2', 'chassis', 'engine', 'assembly']

# Penalty for unused components at end of day
# This incentivizes the agent to assemble cars rather than stockpile components
UNUSED_COMPONENT_PENALTY = -0.5  # Penalty per unused component at day end
