import numpy as np
import matplotlib.pyplot as plt
import copy
from Config import Config
from ScheduleEnv import TrainSchedulingEnv
from tqdm import trange

# 配置环境参数
config = Config()
config.max_time = 239  # 设置最大时间步
config.reward_time_interval = 1 # 设置奖励计算的时间间隔
num_episodes = 5   # 设置采样的 episode 数量
config.no_consider_collision = 1  # 测试模式
config.network_config_path = 'network_config/easy.json'  # 设置网络配置文件路径
# 创建环境实例
env = TrainSchedulingEnv(config)


# 定义动作策略
fixed_action = 0  # 固定动作为加速（假设加速动作索引为 0）
comparison_actions = [0, 1, 2]  # 比较的动作：加速（0），减速（1），保持（2）
repeat_times = 10  # 在复制的环境中每个动作重复执行的次数

# 用于记录各时间步的奖励方差
reward_variances_over_time = []
total_total_rewards = []
# 采样多次以获得平均方差
for episode in trange(num_episodes, desc="Sampling Episodes"):
    env.set_seed(episode)  # 设置种子，确保每个 episode 的初始状态一致
    obs = env.reset()
    done = False
    time_step = 0
    reward_variances_per_episode = []
    total_rewards_for_comparison_actions = []
    while not done and time_step < config.max_time:
        # 1. 在实际环境中执行固定动作（加速）
        actions = [fixed_action for _ in range(3)]  # 所有 agent 都执行固定动作
        actions = np.eye(config.action_space_dim)[actions]  # 将动作转换为 one-hot 编码

        # 2. 与环境交互
        obs, reward, done, info = env.step(actions)
        done = done[0]
        # 记录实际动作执行后的状态
        base_state = copy.deepcopy(env)  # 复制环境状态
        # for train in env.world.get_trains():
        #     print("ENV Train ID: {}, Speed: {}".format(train.train_id, train.speed))
        # for train in base_state.world.get_trains():
        #     print("Train ID: {}, Speed: {}".format(train.train_id, train.speed))
        # 3. 在复制的环境中分别执行其他动作，计算累计奖励方差
        rewards_for_comparison_actions = []
        for comp_action in comparison_actions:
            # 在复制的环境中执行新的动作
            comp_env = copy.deepcopy(base_state)  # 使用原始环境状态复制一个新环境
            # 打印这个环境所有列车的速度
            # for train in comp_env.world.get_trains():
            #     print("Train ID: {}, Speed: {}".format(train.train_id, train.speed))
            # 累计奖励
            total_reward = 0
            for _ in range(repeat_times):
                comp_actions = [comp_action for _ in range(3)]
                comp_actions = np.eye(config.action_space_dim)[comp_actions]

                # 在复制环境中执行动作
                _, comp_reward, comp_done, _ = comp_env.step(comp_actions)
                total_reward += comp_reward[0]  # 累计第一个智能体的奖励
                
                # 若到达终止状态，则提前停止
                if comp_done[0]:
                    break

            # 记录当前动作的累计奖励
            rewards_for_comparison_actions.append(total_reward)
        # print('rewards_for_comparison_actions', rewards_for_comparison_actions)
        # 计算当前时间步的累计奖励方差
        variance_at_step = np.var(rewards_for_comparison_actions)
        reward_variances_per_episode.append(variance_at_step)
        total_rewards_for_comparison_actions.append(rewards_for_comparison_actions)
        # 增加时间步
        time_step += 1

    # 记录每个 episode 在各时间步上的奖励方差
    reward_variances_over_time.append(reward_variances_per_episode)
    total_total_rewards.append(total_rewards_for_comparison_actions)
# 计算每个时间步上的平均奖励方差
reward_variances_mean = np.mean(reward_variances_over_time, axis=0)
total_rewards_for_comparison_actions_mean = np.mean(total_total_rewards, axis=0)
print('total_rewards_for_comparison_actions_mean', total_rewards_for_comparison_actions_mean)
# 绘制方差随时间变化的图
plt.figure(figsize=(10, 6))
plt.plot(range(len(reward_variances_mean)), reward_variances_mean, marker='o', linestyle='-', color='b')
plt.title('Average Reward Variance Over Time for Different Actions (Accumulated over 5 Steps)')
plt.xlabel('Time Step')
plt.ylabel('Average Reward Variance (Accumulated Reward)')
plt.grid(True)
plt.savefig("reward_variance_comparison_fixed_action_accumulated_easy_4h_horizon10.png")

# 绘制三种动作的平均奖励曲线
plt.figure(figsize=(10, 6))
action_labels = ['Accelerate (0)', 'Decelerate (1)', 'Stay (2)']
colors = ['r', 'g', 'orange']  # 设置三种动作的颜色
for i in range(len(action_labels)):
    plt.plot(range(total_rewards_for_comparison_actions_mean.shape[0]),  # 时间步数
             total_rewards_for_comparison_actions_mean[:, i],  # 第 i 种动作的平均奖励
             marker='x', linestyle='--', color=colors[i], label=f'Average Reward - {action_labels[i]}')

# 设置图形标题、坐标轴标签和图例
plt.title('Average Rewards Over Time for Different Actions')
plt.xlabel('Time Step')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.savefig("reward_total_comparison_fixed_action_accumulated_easy_4h_horizon10.png")
