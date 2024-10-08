# CUDA_VISIBLE_DEVICES=0 python main.py --experiment_name toy6_plus_agent_reward_weight100 \
#     --network_config_file 'toy_version6_big_station_capacity.json' --use_agent_reward 1 --train_reward_weight 100 &
# CUDA_VISIBLE_DEVICES=1 python main.py --experiment_name toy7_plus_agent_reward_weight100 \
#     --network_config_file 'toy_version7_big_car_capacity.json' --use_agent_reward 1 --train_reward_weight 100 &
# CUDA_VISIBLE_DEVICES=2 python main.py --experiment_name easy3_plus_agent_reward_weight100 \
#     --network_config_file 'easy_version3_big_car_capacity.json' --use_agent_reward 1 --train_reward_weight 100 &

# CUDA_VISIBLE_DEVICES=3 python main.py --experiment_name toy2_plus_agent_reward_weight100 \
#     --network_config_file 'toy_version2.json' --use_agent_reward 1 --train_reward_weight 100 &
# CUDA_VISIBLE_DEVICES=4 python main.py --experiment_name toy3_plus_agent_reward_weight100 \
#     --network_config_file 'toy_version3.json' --use_agent_reward 1 --train_reward_weight 100 &
# CUDA_VISIBLE_DEVICES=5 python main.py --experiment_name easy2_plus_agent_reward_weight100 \
#     --network_config_file 'easy_version2.json' --use_agent_reward 1 --train_reward_weight 100 &

# CUDA_VISIBLE_DEVICES=6 python main.py --experiment_name toy2_plus_agent_reward_weight100 \
#     --network_config_file 'toy_version2.json' --use_agent_reward 1 --train_reward_weight 100 &
# CUDA_VISIBLE_DEVICES=7 python main.py --experiment_name toy3_plus_agent_reward_weight100 \
#     --network_config_file 'toy_version3.json' --use_agent_reward 1 --train_reward_weight 100 &
# CUDA_VISIBLE_DEVICES=0 python main.py --experiment_name easy2_plus_agent_reward_weight100 \
#     --network_config_file 'easy_version2.json' --use_agent_reward 1 --train_reward_weight 100

# CUDA_VISIBLE_DEVICES=7 python main.py --experiment_name easy8_plus_agent_reward_weight1 \
#     --network_config_file 'easy_version8_big_car_capacity_big_station_capacity.json' \
#     --use_agent_reward 1 --train_reward_weight 1 &

# CUDA_VISIBLE_DEVICES=0 python main.py --experiment_name easy9 --network_config_file 'easy_version9_big_station_capacity.json' &

CUDA_VISIBLE_DEVICES=6 python main.py --experiment_name easy8_real_data \
    --network_config_file 'easy_version8_big_car_capacity_big_station_capacity.json' --use_real_data 1 &

CUDA_VISIBLE_DEVICES=5 python main.py --experiment_name easy8_real_data_use_time \
    --network_config_file 'easy_version8_big_car_capacity_big_station_capacity.json' \
    --use_real_data 1 --version 'ver2' &

CUDA_VISIBLE_DEVICES=1 python main.py --experiment_name easy8_real_data_plus_agent_reward \
    --network_config_file 'easy_version8_big_car_capacity_big_station_capacity.json' \
    --use_real_data 1  --use_agent_reward 1 --train_reward_weight 1 &

CUDA_VISIBLE_DEVICES=2 python main.py --experiment_name easy8_real_data_use_time_plus_agent_reward \
    --network_config_file 'easy_version8_big_car_capacity_big_station_capacity.json' \
    --use_real_data 1 --use_agent_reward 1 --train_reward_weight 1 --version 'ver2' 
    
# CUDA_VISIBLE_DEVICES=3 python main.py --experiment_name toy8_plus_agent_reward \
#     --network_config_file 'toy_version8_big_station_capacity.json' --use_agent_reward 1 --train_reward_weight 1 &

# CUDA_VISIBLE_DEVICES=0 python main.py --experiment_name toy8 \
#     --network_config_file 'toy_version8_big_station_capacity.json'