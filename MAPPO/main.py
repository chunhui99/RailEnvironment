from config import get_config
import sys
sys.path.append('/home/chunhuili/Transportation')
sys.path.append('/home/chunhuili/Transportation/RailEnvironment')
from MAPPO.utils.util import set_device
from pathlib import Path
import os
import wandb
import socket
import setproctitle
import torch
import numpy as np
from RailEnvironment.ScheduleEnv import TrainSchedulingEnv
from RailEnvironment.envs_wrappers import SubprocVecEnv, DummyVecEnv

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='rail_schedule')
    parser.add_argument("--num_stations", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=3, help="number of players")
    parser.add_argument('--time_interval', type=int, default=1, help='Time interval in seconds')
    parser.add_argument('--reward_time_interval', type=int, default=5, help='Time interval in seconds')
    parser.add_argument('--obs_dim', type=int, default=9, help='Observation dimension')
    parser.add_argument('--max_time', type=int, default=60 - 1, help='Maximum simulation time')
    parser.add_argument('--safe_distance', type=int, default=100, help='Safe distance between trains')
    parser.add_argument('--max_passengers_per_station_per_interval', type=int, default=10, help='Maximum number of passengers arriving at a station per time interval')
    parser.add_argument('--speed_delta', type=int, default=1, help='Acceleration rate of the train')
    parser.add_argument('--min_speed', type=int, default=10, help='Minimum speed of the train')
    parser.add_argument('--default_speed', type=int, default=25, help='Minimum speed of the train')
    parser.add_argument('--action_space_dim', type=int, default=3, help='Minimum speed of the train')
    parser.add_argument('--network_config_path', type=str, default='/home/chunhuili/Transportation/RailEnvironment/network_config/easy.json', help='Init config of the network')

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

def main(args):
    # Get the config
    parser = get_config()
    all_args = parse_args(args, parser)
    all_args.eval_episode_length = all_args.max_time + 1
    # rmappo
    all_args.use_recurrent_policy = True
    all_args.use_naive_recurrent_policy = False


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