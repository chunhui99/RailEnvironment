import numpy as np
import matplotlib.pyplot as plt
from Config import Config
from ScheduleEnv import TrainSchedulingEnv
from tqdm import tqdm

# 假设的列车速度范围
speed_values = np.arange(15, 45, 1)  # 速度从15到45，每隔1个单位
accelerate_rewards = {}
stay_rewards = {}
max_time = 60 - 1

# 统一速度并记录奖励 (Accelerate Action)
for speed in tqdm(speed_values, desc='Accelerate Action'):
    config = Config()
    config.max_time = max_time
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
        total_reward += np.mean(reward)
        done = done[0]
    
    stay_rewards[speed] = total_reward

print('Accelerate Rewards:', accelerate_rewards)
print('Stay Rewards:', stay_rewards)

# 绘制速度与奖励的关系曲线
plt.figure(figsize=(10, 6))

# Plot for Accelerate Action
plt.plot(speed_values, accelerate_rewards.values(), marker='o', linestyle='-', color='r', label='Accelerate Action')

# Plot for Stay Action
plt.plot(speed_values, stay_rewards.values(), marker='x', linestyle='--', color='b', label='Stay Action')

# 添加 MAPPO 数据点
mappo_speeds = [15, 20, 25]
mappo_rewards = [-386988, -321042, -267074]
plt.plot(mappo_speeds, mappo_rewards, marker='s', linestyle='-', color='g', label='MAPPO')  # 绘制折线并设置图例

# Add title and labels
plt.title('Comparison of Reward vs Speed (Accelerate vs Stay vs MAPPO) Time = 1h')
plt.xlabel('Train Speed (km/h)')
plt.ylabel('Total Reward')

# Add grid, legend, and save figure
plt.grid(True)
plt.legend(loc='best')  # Add legend to differentiate between accelerate, stay, and MAPPO actions
plt.savefig('speed_vs_reward_comparison_1h_with_mappo.png')