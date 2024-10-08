class Config:
    def __init__(self):
        self.time_interval = 1  # Time stamp interval(min)
        self.obs_dim = 9  # Observation dimension
        self.max_time = 60 - 1 # 仿真时间(min)
        self.reward_time_interval = 5 # Time interval for reward calculation
        self.safe_distance = 100  # (m) Safe distance between trains
        self.max_passengers_per_station_per_interval = 10  # Maximum number of passengers arriving at a station per time interval
        self.speed_delta = 5  # km/h Acceleration rate of the train
        self.min_speed = 10 # km/h Minimum speed of the train
        self.seed = 1
        self.network_config_path = '/home/chunhuili/Transportation/RailEnvironment/environment/network_config/easy_version2.json'
        self.action_space_dim = 3
        self.in_test = 0
        self.no_consider_collision = 1
        self.reward_type = 'waiting_time'
        self.use_agent_reward = 1
        self.train_reward_weight = 1
        self.use_real_data = 0
        self.version = 'ver1'