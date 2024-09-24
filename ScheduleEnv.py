import gym
from gym import spaces
import numpy as np
import sys
sys.path.append('/home/chunhuili/Transportation')
from RailEnvironment.Station import Station
from RailEnvironment.Line import Line, DualLine
from RailEnvironment.Train import Train
from RailEnvironment.World import World
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import json

class TrainSchedulingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(TrainSchedulingEnv, self).__init__()
        self.config = config
        self.world = World(config)
        self.seed = self.config.seed
        network_json = json.load(open(config.network_config_path))
        # self._initialize_network()
        self._initialize_network_from_config(network_json)
        self._setup_spaces()
        self.is_record = False
        self.control_data = []
        self.traffic_data = []
        self.default_speed = None

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
        lineA_forward = Line(self.world, name="lineA_forward", line_position_y=lineA_forward_position_y)
        lineA_backward = Line(self.world, name="lineA_backward", line_position_y=lineA_backward_position_y, forward=False)
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

    def _initialize_network_from_config(self, network_config):
        """
        Initialize the railway network based on the configuration file.
        """
        config = network_config
        dual_line_dict = {}
        for line_data in config["lines"]:

            line_position_y = line_data.get("position_y", 0)
            line = Line(self.world, name=line_data["name"], line_position_y= line_position_y, forward = line_data["forward"])

            begin_distance=line_data["begin_distance"]
            
            stations = []
            station_position_xs  = [begin_distance]
            station_distances = sorted(line_data["station_distances"].items(), key=lambda x: x[0])  
            # print('station_distances:', station_distances)
            for value in station_distances:
                begin_distance += value[1]
                station_position_xs.append(begin_distance)
            stationX = [0] + station_position_xs + [station_position_xs[-1] + line_data["end_distance"]]
            line.stationX = stationX
            for i, station_data in enumerate(line_data["stations"]):
                station_position_x = station_position_xs[i] if line.forward else station_position_xs[len(station_position_xs) - i - 1]
                station = Station(self.config, station_id=station_data["id"], station_name=station_data["name"],
                                  line=line, max_capacity=station_data["capacity"],
                                  passenger_arrival_time_bin = station_data["passenger_arrival_time_bin"],
                                  station_position_x = station_position_x,
                                  station_position_y = line_position_y)
                stations.append(station)
            # Set up the line with stations
            station_distance_dict = {(key.split('-')[0],key.split('-')[1]):value for key, value in line_data["station_distances"].items()}

            line.set_line(start_station=stations[0], end_station=stations[-1], stations=stations,
                          station_distances=station_distance_dict,
                          begin_distance=line_data["begin_distance"], end_distance=line_data["end_distance"])

            dual_line_id = line_data["dual_line_id"]
            if dual_line_id not in dual_line_dict:
                dual_line_dict[dual_line_id] = [line]
            else:
                dual_line_dict[dual_line_id].append(line)
            self.world.add_line(line)

        for key, value in dual_line_dict.items():
            dual_line = DualLine(value[0], value[1])

        for train_data in config["trains"]:
            line = self.world.get_line_by_name(train_data["line"])
            train = Train(self.config, train_id=train_data["id"], line=line,
                          capacity=train_data["capacity"], default_speed=train_data["default_speed"],
                          max_speed=train_data["max_speed"], departure_time=train_data["departure_time"],
                          length=train_data["length"])
            self.world.add_train(train)

        self.world.init_train_to_line()

    def set_all_speed(self, speed):
        for train in self.world.get_trains():
            train.default_speed = speed
            train.speed = speed
        self.default_speed = speed

    def sample_action(self):
        """
        Sample a random action for each train based on its current state.
        
        :return: List of sampled actions for each train, ensuring that only valid actions are chosen.
        """
        actions = [np.random.choice([0, 1, 2]) for train in self.world.get_trains()]
        return actions

    def _setup_spaces(self):
        """
        Define the action and observation spaces for the environment.
        """
        # Example action space: for each train, if moving, decide to accelerate, decelerate, or maintain speed; if not moving, stay or depart
        self.action_space = spaces.Discrete(self.config.action_space_dim)
        
        # Observation space 
        train_obs_dim = self.config.obs_dim  # Example dimension
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.world.get_train_num()):
            self.observation_space.append(spaces.Box(low=0, high=np.inf, shape=(train_obs_dim, ), dtype=np.float32))
            self.share_observation_space.append(spaces.Box(low=0, high=np.inf, shape=(train_obs_dim * self.world.get_train_num(), ), dtype=np.float32))

    def step(self, actions):
        """
        Perform one step in the environment.

        :param action: List of actions for each train.
        :return: observation, reward, done, info
        """
        # Update the actions of all trains
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
        if not collision:
            reward = self._calculate_reward()
            reward = np.array([reward] * len(self.agents))
        else:
            reward = np.array([-10000000] * len(self.agents))
        # Check if done 时间到了或者发生碰撞都结束
        done = self.world.current_time >= self.config.max_time or collision

        done = np.array([done] * len(self.agents))

        # Optionally, collect additional info
        info = {
            agent.train_id: {
                'speed': agent.speed,
                'action': agent.action_dict[np.argmax(agent.action)]
            } for agent in self.agents
        }


        return observations, reward, done, info

    def reset(self):
        """
        Reset the environment to an initial state.

        :return: Initial observation.
        """
        # Reset world
        self.world = World(self.config)
        network_json = json.load(open(self.config.network_config_path))
        # self._initialize_network()
        self._initialize_network_from_config(network_json)
        if self.default_speed is not None:
            self.set_all_speed(self.default_speed)
        self.current_time = 0
        self.world.set_seed(self.seed)
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
            obs.append(train_obs)
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self):
        """
        Calculate the reward for the current state.

        :return: Total reward.
        """
        total_waiting = sum(station.get_total_wait_time() for station in self.world.get_stations())
        # Example reward: negative of total waiting time (to minimize waiting time)
        current_time = self.world.current_time
        if (current_time + 1) % self.config.reward_time_interval == 0:
        # Minimize total waiting time
            reward = - total_waiting
        else:
        # Set small penalties for time passing
            reward = - 0.1 * total_waiting
        return reward

    def set_seed(self, seed=None):
        self.seed = seed
        self.world.set_seed(seed)


if __name__ == "__main__":
    from Config import Config
    config = Config()
    env = TrainSchedulingEnv(config)
    obs = env.reset()
    done = False
    while not done:
        actions = [env.sample_action() for _ in range(3)]
        actions = np.eye(config.action_space_dim)[actions]
        # print("Action:", action)
        obs, reward, done, info = env.step(actions)
        # env.render()