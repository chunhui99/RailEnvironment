from config import get_config
import sys
sys.path.append('/home/chunhuili/Transportation')
sys.path.append('/home/chunhuili/Transportation/RailEnvironment/src')
sys.path.append('/home/chunhuili/Transportation/RailEnvironment')
from MAPPO.utils.util import set_device
from pathlib import Path
import os
import wandb
import socket
import setproctitle
import torch
import numpy as np
from environment.ScheduleEnv import TrainSchedulingEnv
from environment.envs_wrappers import SubprocVecEnv, DummyVecEnv
import pandas as pd

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='rail_schedule')
    parser.add_argument("--num_stations", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=3, help="number of players")
    parser.add_argument('--time_interval', type=int, default=1, help='Time interval in seconds')
    parser.add_argument('--reward_time_interval', type=int, default=5, help='Time interval in seconds')
    parser.add_argument('--obs_dim', type=int, default=9, help='Observation dimension')
    parser.add_argument('--no_consider_collision', type=int, default = 1, help = '1 is no collision')
    parser.add_argument('--use_agent_reward', type=int, default = 0, help = '1 is use agent reward')
    parser.add_argument('--safe_distance', type=int, default=100, help='Safe distance between trains')
    parser.add_argument('--max_passengers_per_station_per_interval', type=int, default=10, help='Maximum number of passengers arriving at a station per time interval')
    parser.add_argument('--speed_delta', type=int, default=5, help='Acceleration rate of the train')
    parser.add_argument('--min_speed', type=int, default=10, help='Minimum speed of the train')
    parser.add_argument('--default_speed', type=int, default=15, help='Minimum speed of the train')
    parser.add_argument('--action_space_dim', type=int, default=3, help='Minimum speed of the train')
    parser.add_argument('--network_config_file', type=str, default='easy.json', help='Init config of the network')
    parser.add_argument('--network_config_root', type=str, default='/home/chunhuili/Transportation/RailEnvironment/environment/network_config/', help='Init config of the network')
    parser.add_argument('--reward_type', type=str, default='waiting_time')
    parser.add_argument("--episode_length", type=int, default=900, help="Max length for any episode")
    parser.add_argument("--train_reward_weight", type=int, default=1)
    parser.add_argument("--use_real_data", type=int, default=0)
    parser.add_argument("--version", type=str, default='ver1')
    all_args = parser.parse_known_args(args)[0]
    return all_args

def make_train_env(all_args, mode = 'train', speed = None):
    def get_env_fn(rank):
        def init_env():
            env = TrainSchedulingEnv(all_args)
            if speed is not None:
                env.set_all_speed(speed)
            if mode == 'train':
                env.set_seed(all_args.seed + rank * 1000)
            else:
                env.set_seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if mode == 'train':
        if all_args.n_rollout_threads > 1:
            envs = SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)]) 
        else:
            envs = DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.n_eval_rollout_threads > 1:
            envs = SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)]) 
        else:
            envs = DummyVecEnv([get_env_fn(0)])
    return envs

def read_real_data(file_path):

    # 读取保存的平均出站人数的 CSV 文件
    df = pd.read_csv(file_path)

    # 创建时间步与平均出站人数的映射（时间步从1到900）
    time_steps = list(range(1, 901))

    # 提取前900个时间点的数据
    average_exit_counts = df['average_exit_count'].head(900)

    # 将时间步与对应的平均出站人数（取整）组成字典
    time_step_to_avg_exit = {time_step: round(count) for time_step, count in zip(time_steps, average_exit_counts)}
    return time_step_to_avg_exit

def main(args):
    # Get the config
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.max_time = all_args.episode_length - 1
    all_args.eval_episode_length = all_args.max_time + 1
    # rmappo
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False
    all_args.network_config_path = all_args.network_config_root + all_args.network_config_file
    all_args.real_data_path = '/home/chunhuili/Transportation/RailEnvironment/data/average_exit_count_per_minute.csv'
    all_args.real_data_dict = read_real_data(all_args.real_data_path)
    if all_args.version == 'ver2':
        all_args.obs_dim = 10
    # Set device
    device = set_device(all_args)

    
    # Dir
    # /上一级目录/results/环境名称/场景名称/算法名称/实验名称
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Wandb

    run = wandb.init(config=all_args, project=all_args.env_name,
                        notes=socket.gethostname(), # 返回运行实验机器的主机名
                        name=str(all_args.algorithm_name) + "_" + str(all_args.experiment_name) + "_seed" + str(all_args.seed),
                        group=all_args.scenario_name, # 实验分组
                        dir=str(run_dir), # 实验结果保存路径
                        job_type="training", # 任务类型
                        reinit=True)
    
    # 设置当前运行的进程标题
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args, speed = all_args.default_speed)
    eval_envs = make_train_env(all_args, mode = 'eval', speed = all_args.default_speed) if all_args.use_eval else None
    # eval_envs = None
    # TODO: num_agents
    num_agents = all_args.num_agents

    # 假设share_policy 因为agent是同质的
    from runner.rail_runner import RailRunner as Runner

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }


    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])