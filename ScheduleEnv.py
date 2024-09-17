import gym
from gym import spaces
import numpy as np
from Station import Station
from Line import Line, DualLine
from Train import Train
from World import World
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb

class TrainSchedulingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(TrainSchedulingEnv, self).__init__()
        self.config = config
        self.world = World(config)
        self._initialize_network()
        self._setup_spaces()

        self.is_record = False
        self.control_data = []
        self.traffic_data = []

    def _load_config(self, config):
        """
        Load configuration parameters from a config object.

        :param config: Configuration object (e.g., from configparser).
        """
        # Example configuration loading
        self.time_step = config.getint('time_step')  # Minutes per step
        self.max_time = config.getint('max_time')    # Total simulation time in minutes
        self.record = config.getboolean('record')    # Whether to record data

    def _initialize_network(self):
        """
        Initialize the railway network with lines, stations, and trains based on the configuration.
        This example uses hypothetical data for demonstration.
        """
        config = self.config

        # Example setup with 4 intersecting lines
        lineA_forward_position_y = 0
        lineA_backward_position_y = -1
        lineA_forward = Line(name="lineA_forward", line_position_y=lineA_forward_position_y)
        lineA_backward = Line(name="lineA_backward", line_position_y=lineA_backward_position_y, forward=False)
        lineA = DualLine(lineA_forward, lineA_backward)

        station_A_forward = Station(config, 0, "A", lineA.line_forward, 1000, station_position_x = 1000, station_position_y=lineA_forward_position_y)
        station_B_forward = Station(config, 1, "B", lineA.line_forward, 800, station_position_x = 2000,  station_position_y=lineA_forward_position_y)
        station_C_forward = Station(config, 2, "C", lineA.line_forward, 1200, station_position_x = 3500,  station_position_y=lineA_forward_position_y)
        station_D_forward = Station(config, 3, "D", lineA.line_forward, 1500, station_position_x = 4200,  station_position_y=lineA_forward_position_y)

        station_A_backward = Station(config, 4, "A", lineA.line_backward, 1000, station_position_x = 1000,  station_position_y=lineA_backward_position_y)
        station_B_backward = Station(config, 5, "B", lineA.line_backward, 800, station_position_x = 2000, station_position_y=lineA_backward_position_y)
        station_C_backward = Station(config, 6, "C", lineA.line_backward, 1200, station_position_x = 3500, station_position_y=lineA_backward_position_y)
        station_D_backward = Station(config, 7, "D", lineA.line_backward, 1500, station_position_x = 4200, station_position_y=lineA_backward_position_y)

        lineA.line_forward.set_line(start_station=station_A_forward, end_station=station_D_forward,
                     stations=[station_A_forward, station_B_forward, station_C_forward, station_D_forward],
                     station_distances={("A", "B"): 1000, ("B", "C"): 1500, ("C", "D"): 1200}, begin_distance=1000, end_distance=500)
        lineA.line_backward.set_line(start_station=station_D_backward, end_station=station_A_backward,
                     stations=[station_D_backward, station_C_backward, station_B_backward, station_A_backward],
                     station_distances={("D", "C"): 1200, ("C", "B"): 1500, ("B", "A"): 1000}, begin_distance=1000, end_distance=500)

        lineA.line_forward.stationX = [0,1000,2000,3500,4700,5200]
        lineA.line_backward.stationX = [0,1000,2000,3500,4700,5200]
        station_A_forward.station_position_x = lineA.line_forward.stationX[1]
        station_B_forward.station_position_x = lineA.line_forward.stationX[2]
        station_C_forward.station_position_x = lineA.line_forward.stationX[3]
        station_D_forward.station_position_x = lineA.line_forward.stationX[4]
        station_A_backward.station_position_x = lineA.line_backward.stationX[1]
        station_B_backward.station_position_x = lineA.line_backward.stationX[2]
        station_C_backward.station_position_x = lineA.line_backward.stationX[3]
        station_D_backward.station_position_x = lineA.line_backward.stationX[4]
        # Assuming the lines are straight, and we add more for intersection purposes
        self.world.add_line(lineA.line_forward)
        self.world.add_line(lineA.line_backward)

        train_1 = Train(config, train_id=0, line=lineA.line_forward, capacity=300, default_speed=20, max_speed=30, departure_time=0, length=200)
        train_2 = Train(config, train_id=1, line=lineA.line_forward, capacity=300, default_speed=15, max_speed=25, departure_time=20, length=200)
        train_3 = Train(config, train_id=2, line=lineA.line_backward, capacity=300, default_speed=22, max_speed=27, departure_time=0, length=200)

        self.world.add_train(train_1)
        self.world.add_train(train_2)
        self.world.add_train(train_3)

        self.world.init_train_to_line()

    def sample_action(self):
        """
        Sample a random action for each train based on its current state.
        
        :return: List of sampled actions for each train, ensuring that only valid actions are chosen.
        """
        actions = []
        
        for train in self.world.get_trains():
            if train.moving:
                # If the train is moving, valid actions are: accelerate, decelerate, stay
                valid_actions = [0, 1, 2]  # Assuming 0: accelerate, 1: decelerate, 2: stay
            else:
                # If the train is not moving, valid actions are: depart, stay
                valid_actions = [3, 4]  # Assuming 3: depart, 4: stay

            # Sample a valid action for the train
            action = np.random.choice(valid_actions)
            actions.append(action)
        print('world.get_trains():', self.world.get_trains())
        return actions

    
    def _setup_spaces(self):
        """
        Define the action and observation spaces for the environment.
        """
        # Example action space: for each train, if moving, decide to accelerate, decelerate, or maintain speed; if not moving, stay or depart
        self.action_space = spaces.MultiDiscrete([5] * self.world.get_train_num())
        
        # Observation space 
        train_obs_dim = self.config.obs_dim  # Example dimension
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.world.get_train_num() * train_obs_dim, ), dtype=np.float32)

    def step(self, actions):
        """
        Perform one step in the environment.

        :param action: List of actions for each train.
        :return: observation, reward, done, info
        """
        self.agents = self.world.get_trains()
        # print('self.agents:', self.agents)
        for idx, agent in enumerate(self.agents):
            agent.action = actions[idx]
            # print('agent.action:', agent.action)
        # Advance the world state
        collision = self.world.step()
        # print('Collision:', collision)
        # Collect observations
        observations = self._get_observations(self.agents)

        # Calculate rewards (example: minimize total passenger waiting time)
        reward = self._calculate_reward()

        # Check if done 时间到了或者发生碰撞都结束
        done = self.world.current_time >= self.config.max_time or collision

        # Optionally, collect additional info
        info = {}

        return observations, reward, done, info

    def reset(self, config):
        """
        Reset the environment to an initial state.

        :return: Initial observation.
        """
        # Reset world
        self.world = World(config)
        self._initialize_network()
        self.current_time = 0

        # Reset observations
        self.agents = self.world.get_trains()
        observations = self._get_observations(self.agents)

        return observations
    
    def _get_observations(self, agents):
        """
        Get observations for all trains.

        :return: Flattened observation array.
        """
        obs = []

        for train in agents:
            # 到下一站的距离，到终点的距离，现有乘客数，容量，下一站等待乘客数量，是否在移动，前车车距（没有就是极大)，前车容量（没有就是0）
            train_obs = [
                train.get_distance_to_next_station(),
                train.get_distance_to_final(),
                train.get_current_passengers(),
                train.get_capacity(),
                train.get_target_station_passengers(),
                train.moving,
                train.get_distance_to_previous_train(),
                train.get_previous_train_capacity(),
                train.on_line
            ]
            obs.extend(train_obs)
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self):
        """
        Calculate the reward for the current state.

        :return: Total reward.
        """
        total_waiting = sum(station.get_total_wait_time() for station in self.world.get_stations())
        # Example reward: negative of total waiting time (to minimize waiting time)
        reward = -total_waiting
        return reward


if __name__ == "__main__":
    from Config import Config
    config = Config()
    env = TrainSchedulingEnv(config)
    obs = env.reset(config)
    done = False
    while not done:
        action = env.sample_action()
        # print("Action:", action)
        obs, reward, done, info = env.step(action)
        # env.render()