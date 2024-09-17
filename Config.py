class Config:
    def __init__(self):
        self.time_interval = 5  # Time interval in seconds
        self.obs_dim = 9  # Observation dimension
        self.max_time = 1000 # 仿真时间
        self.safe_distance = 100  # Safe distance between trains
        self.max_passengers_per_station_per_interval = 10  # Maximum number of passengers arriving at a station per time interval
        self.speed_delta = 1  # Acceleration rate of the train
        self.min_speed = 10