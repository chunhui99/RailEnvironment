import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/chunhuili/Transportation/RailEnvironment/environment')
from Config import Config
from ScheduleEnv import TrainSchedulingEnv
from tqdm import tqdm

# 加载 JSON 文件内容（动作序列）
# path = '/home/chunhuili/Transportation/RailEnvironment/src/MAPPO/log_infos/toy6\_config_toy_version6_big_station_capacity\_speed_15_ep_900\_steps_10000000.0_eval_env_infos.json'
path = '/home/chunhuili/Transportation/RailEnvironment/src/MAPPO/log_infos/toy6_plus_agent_reward\_config_toy_version6_big_station_capacity\_speed_15_ep_900\_steps_10000000.0_eval_env_infos.json'
with open(path, 'r') as f:
    action_data = json.load(f)

# 定义用于存储奖励的字典
accelerate_rewards = {}
stay_rewards = {}
random_rewards = {}
file_action_rewards = []  # 存储读取文件中动作策略的奖励

# 设置实验参数
hours = 15
max_time = hours * 60 - 1
time_str = str(hours) + 'h'
network_root = '/home/chunhuili/Transportation/RailEnvironment/environment/network_config/'

network_path = network_root + 'toy_version6_big_station_capacity.json'
use_agent_reward = 1

# 假设的列车速度范围
speed_values = np.arange(15, 60, 5)  # 速度从15到45，每隔1个单位

# 统一速度并记录奖励 (Accelerate Action)
for speed in tqdm(speed_values, desc='Accelerate Action'):
    config = Config()
    config.max_time = max_time
    config.use_agent_reward = use_agent_reward
    config.network_config_path = network_path
    env = TrainSchedulingEnv(config)
    obs = env.reset()

    # 设置所有列车的初始速度为 speed
    for train in env.world.get_trains():
        train.default_speed = speed
    
    total_reward = 0
    done = False

    # 运行一段时间，计算总奖励 (All trains accelerating)
    while not done:
        actions = [0] * 3  # 0 表示 'accelerate' 动作
        actions = np.eye(config.action_space_dim)[actions]
        obs, reward, done, info = env.step(actions)
        total_reward += np.mean(reward)
        done = done[0]
    
    accelerate_rewards[speed] = total_reward

# 统一速度并记录奖励 (Stay Action)
for speed in tqdm(speed_values, desc='Stay Action'):
    config = Config()
    config.max_time = max_time
    config.use_agent_reward = use_agent_reward
    config.network_config_path = network_path
    env = TrainSchedulingEnv(config)
    obs = env.reset()
    env.set_all_speed(speed)
    
    total_reward = 0
    done = False

    # 运行一段时间，计算总奖励 (All trains staying)
    while not done:
        actions = [2] * 3  # 2 表示 'stay' 动作
        actions = np.eye(config.action_space_dim)[actions]
        obs, reward, done, info = env.step(actions)
        # print('Reward:', reward)
        total_reward += np.mean(reward)
        done = done[0]

    stay_rewards[speed] = total_reward

# 统一速度并记录奖励 (Stay Action)
for speed in tqdm(speed_values, desc='Stay Action'):
    config = Config()
    config.max_time = max_time
    config.use_agent_reward = use_agent_reward
    config.network_config_path = network_path
    env = TrainSchedulingEnv(config)
    obs = env.reset()
    env.set_all_speed(speed)
    
    total_reward = 0
    done = False

    # 运行一段时间，计算总奖励 (All trains staying)
    while not done:
        actions = env.sample_action() 
        actions = np.eye(config.action_space_dim)[actions]
        obs, reward, done, info = env.step(actions)
        # print('Reward:', reward)
        total_reward += np.mean(reward)
        done = done[0]

    random_rewards[speed] = total_reward

# 计算文件中提供的动作序列下的奖励
config = Config()
config.max_time = max_time
config.use_agent_reward = use_agent_reward
config.network_config_path = network_path
env = TrainSchedulingEnv(config)
obs = env.reset()

total_file_reward = 0
for step, actions_info in enumerate(action_data):
    if step >= config.max_time:
        break

    # 将 JSON 文件中的动作转换为整数表示
    actions = []
    for train_id in range(3):  # 假设有 3 个火车
        action_str = actions_info[str(train_id)]["action"]
        if action_str == "accelerate":
            actions.append(0)
        elif action_str == "decelerate":
            actions.append(1)
        else:  # stay
            actions.append(2)
    
    # 将动作转换为 one-hot 编码
    actions = np.eye(config.action_space_dim)[actions]
    obs, reward, done, info = env.step(actions)
    total_file_reward += np.mean(reward)

print('Total Reward for File Actions:', total_file_reward)
file_action_rewards.append(total_file_reward)

# 绘制速度与奖励的关系曲线
plt.figure(figsize=(10, 6))

# Plot for Accelerate Action
plt.plot(speed_values, accelerate_rewards.values(), marker='o', linestyle='-', color='r', label='Accelerate Action')

# Plot for Stay Action
# plt.plot(speed_values, stay_rewards.values(), marker='x', linestyle='--', color='b', label='Stay Action')

# Plot for random Action
# plt.plot(speed_values, random_rewards.values(), marker='x', linestyle='--', color='yellow', label='Random Action')

# Plot for File Action
plt.plot([15], file_action_rewards, marker='s', linestyle='-', color='g', label='File Actions', ms=12, mew=2, mec='black')  # 使用 `ms` 调整点的大小, `mew` 调整边缘宽度, `mec` 设定边缘颜色为黑

# 添加标题和标签
plt.title('Comparison of Reward vs Speed (Accelerate vs Stay vs File Actions) Time = {}'.format(time_str))
plt.xlabel('Train Speed (km/h)')
plt.ylabel('Total Reward')

# 添加网格、图例并保存图形
plt.grid(True)
plt.legend(loc='best')
plt.savefig('plot_figs/toy6_speed_vs_reward_comparison_{}_file_actions_use_agent_reward_1.png'.format(time_str))
plt.show()
