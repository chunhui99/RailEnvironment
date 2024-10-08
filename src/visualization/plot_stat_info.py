import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/chunhuili/Transportation/RailEnvironment/environment')
from Config import Config
from ScheduleEnv import TrainSchedulingEnv
from tqdm import tqdm

# 定义一个移动平均函数
def moving_average(data, window_size=5):
    """
    对数据应用移动平均滤波，窗口大小默认为5。
    :param data: 输入数据列表或数组
    :param window_size: 移动平均的窗口大小
    :return: 平滑后的数据
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 设置实验参数
hours = 15
max_time = hours * 60 - 1
time_str = str(hours) + 'h'
network_root = '/home/chunhuili/Transportation/RailEnvironment/environment/network_config/'
network_files = []
network_files += [
                'easy_version2.json', 
                'toy_version2.json', 
                'toy_version3.json',
                'easy_version3_big_car_capacity.json', 
                'toy_version7_big_car_capacity.json',
                  'toy_version6_big_station_capacity.json', 
                  'easy_version8_big_car_capacity_big_station_capacity.json',
                  'easy_version9_big_station_capacity.json',
                  'toy_version8_big_station_capacity.json',
                  ]

# 设置实验的随机种子数
seeds = [
        0, 42, 100, 2023, 9999, 
        1,2,3,4,5
        ]

# 再加十个
# seeds += [6,7,8,9,10,11,12,13,14,15]

# 用于记录所有文件的奖励、列车人数、车站人数
file_rewards = {}
file_train_passengers = {}
file_station_passengers = {}

for network_file in tqdm(network_files):
    # 初始化用于统计的数据结构
    reward_per_seed = []
    train_passengers_per_seed = []
    station_passengers_per_seed = []

    config = Config()
    config.use_agent_reward = 0
    config.max_time = max_time
    config.network_config_path = network_root + network_file

    for seed in seeds:
        # 设置随机种子
        config.seed = seed

        # 初始化环境
        env = TrainSchedulingEnv(config)
        obs = env.reset()
        
        # 存储当前种子的各个时刻的奖励、列车人数和车站人数
        rewards = []
        train_passengers = {train.train_id: [] for train in env.world.get_trains()}
        station_passengers = {station.station_id: [] for station in env.world.get_stations()}

        done = False
        while not done:
            # 采样动作并进行一步仿真
            actions = env.sample_action() 
            # actions = [0] * 3  # 0 表示 'accelerate' 动作
            actions = np.eye(config.action_space_dim)[actions]
            obs, reward, done, info = env.step(actions)
            
            # 记录奖励
            rewards.append(np.mean(reward))

            # 记录各列车的人数变化
            for train in env.world.get_trains():
                train_passengers[train.train_id].append(train.get_current_passengers())

            # 记录各车站的等待人数变化
            for station in env.world.get_stations():
                station_passengers[station.station_id].append(station.get_waiting_passengers())

            # 结束条件判断
            done = done[0]

        # 保存当前种子的所有结果
        reward_per_seed.append(rewards)
        train_passengers_per_seed.append(train_passengers)
        station_passengers_per_seed.append(station_passengers)

    # 计算奖励均值和方差
    mean_rewards = np.mean(reward_per_seed, axis=0)
    std_rewards = np.std(reward_per_seed, axis=0)

    # 记录奖励结果
    file_rewards[network_file] = (mean_rewards, std_rewards)

    # 计算列车上乘客人数的均值和方差
    # 初始化用于存储均值和方差的字典
    train_mean_passengers = {train_id: [] for train_id in train_passengers.keys()}
    train_std_passengers = {train_id: [] for train_id in train_passengers.keys()}

    for train_id in train_mean_passengers.keys():
        train_data = np.array([seed_data[train_id] for seed_data in train_passengers_per_seed])
        train_mean_passengers[train_id] = np.mean(train_data, axis=0)
        train_std_passengers[train_id] = np.std(train_data, axis=0)

    file_train_passengers[network_file] = (train_mean_passengers, train_std_passengers)

    # 计算车站等待乘客人数的均值和方差
    station_mean_passengers = {station_id: [] for station_id in station_passengers.keys()}
    station_std_passengers = {station_id: [] for station_id in station_passengers.keys()}

    for station_id in station_mean_passengers.keys():
        station_data = np.array([seed_data[station_id] for seed_data in station_passengers_per_seed])
        station_mean_passengers[station_id] = np.mean(station_data, axis=0)
        station_std_passengers[station_id] = np.std(station_data, axis=0)

    file_station_passengers[network_file] = (station_mean_passengers, station_std_passengers)


    # 绘制图表
    plt.figure(figsize=(15, 5))

    # 图1：奖励随时间的变化（过滤掉为0的奖励）
    plt.subplot(1, 3, 1)
    # 找到非零奖励的位置
    non_zero_indices = [i for i, reward in enumerate(mean_rewards) if reward != 0]
    non_zero_rewards = [mean_rewards[i] for i in non_zero_indices]
    non_zero_std = [std_rewards[i] for i in non_zero_indices]

    # 只绘制非零奖励的部分
    plt.plot(non_zero_indices, non_zero_rewards, label='Mean Reward')
    plt.fill_between(non_zero_indices, np.array(non_zero_rewards) - np.array(non_zero_std), 
                    np.array(non_zero_rewards) + np.array(non_zero_std), alpha=0.2)
    plt.title(f'Reward Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.legend()

    # 图2：各列车人数随时间变化（应用移动平均平滑）
    plt.subplot(1, 3, 2)
    for train_id, mean_values in train_mean_passengers.items():
        std_values = train_std_passengers[train_id]
        # 对列车乘客人数应用移动平均
        smooth_mean_values = moving_average(mean_values, window_size=50)
        smooth_std_values = moving_average(std_values, window_size=50)
        
        # 绘制平滑后的曲线
        plt.plot(range(len(smooth_mean_values)), smooth_mean_values, label=f'Train {train_id} Mean')
        plt.fill_between(range(len(smooth_mean_values)), smooth_mean_values - smooth_std_values, 
                        smooth_mean_values + smooth_std_values, alpha=0.2)
    plt.title(f'Train Passengers Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Passengers')
    plt.legend()

    # 图3：各车站人数随时间变化（应用移动平均平滑）
    plt.subplot(1, 3, 3)
    for station_id, mean_values in station_mean_passengers.items():
        std_values = station_std_passengers[station_id]
        # 对车站等待乘客人数应用移动平均
        smooth_mean_values = moving_average(mean_values, window_size=50)
        smooth_std_values = moving_average(std_values, window_size=50)
        
        # 绘制平滑后的曲线
        plt.plot(range(len(smooth_mean_values)), smooth_mean_values, label=f'Station {station_id} Mean')
        plt.fill_between(range(len(smooth_mean_values)), smooth_mean_values - smooth_std_values, 
                        smooth_mean_values + smooth_std_values, alpha=0.2)
    plt.title(f'Station Passengers Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Waiting Passengers')
    plt.legend()

    # 显示并保存图表
    plt.tight_layout()
    plt.savefig(f'{network_file}_statistics.png')