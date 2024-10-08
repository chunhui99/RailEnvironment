import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
path = '/home/chunhuili/Transportation/RailEnvironment/src/MAPPO/log_infos/toy6_plus_agent_reward\_config_toy_version6_big_station_capacity\_speed_15_ep_900\_steps_10000000.0_eval_env_infos.json'
path2 = '/home/chunhuili/Transportation/RailEnvironment/src/MAPPO/log_infos/toy6\_config_toy_version6_big_station_capacity\_speed_15_ep_900\_steps_10000000.0_eval_env_infos.json'
with open(path, 'r') as f:
    data = json.load(f)
# Define a function to map actions to rewards
# Count the frequency of each action for each agent
action_counts = {str(i): {'accelerate': 0, 'stay': 0, 'decelerate': 0} for i in range(3)}
for step in data:
    for agent, info in step.items():
        action_counts[agent][info['action']] += 1

# Create a DataFrame for visualization
action_df = pd.DataFrame(action_counts).T

# Calculate the ratio of each action
action_ratio = action_df.div(action_df.sum(axis=1), axis=0)

# Plot the action ratio as a bar chart
action_ratio.plot(kind='bar', stacked=True, figsize=(10, 6), color=['r', 'b', 'g'])

# Customize plot
plt.title('Proportion of Accelerate, Stay, and Decelerate Actions for Each Agent (After)')
plt.xlabel('Agent')
plt.ylabel('Proportion')
plt.legend(title='Actions')
plt.grid(True)
plt.xticks(rotation=0)
plt.savefig('action_ratio_2.png')