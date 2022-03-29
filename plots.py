import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def num_env_steps_sampled(ds):
    return ds['num_env_steps_sampled'].values

def reward(ds):
    return ds['episode_reward_mean'].values

def rollout_len(ds):
    return ds['episode_len_mean'].values

# Inverted Pendulum: Current vs Old

ren_urwd_1_ds = pd.read_csv('../ray_results/URwd_PPO/ProjRENModel_OtherInvertedPendulumEnv_phiTanh_state2_hidden1rho1.0_0_2022-03-28_12-23-23/progress.csv')
rnn_old_urwd_1_ds = pd.read_csv('../ray_results/URwd_PPO/ProjRNNOldModel_OtherInvertedPendulumEnv_phiTanh_state2_hidden1rho1.0_0_hidden_size=1_2022-03-28_12-26-16/progress.csv')

ren_num_samples = num_env_steps_sampled(ren_urwd_1_ds)
rnn_old_num_samples = num_env_steps_sampled(rnn_old_urwd_1_ds)

ren_rwd = reward(ren_urwd_1_ds)
rnn_old_rwd = reward(rnn_old_urwd_1_ds)

max_reward = 804
upper_bound = max_reward*np.ones_like(ren_num_samples)

plt.figure()
plt.plot(ren_num_samples, ren_rwd)
plt.plot(rnn_old_num_samples, rnn_old_rwd)
# plt.plot(ren_num_samples, upper_bound)
plt.legend(["REN", "RNN from [4]"])
plt.title("Reward vs. Number Plant Step Samples")
plt.xlabel("Number of Plant Step Samples")
plt.ylabel("Reward")
plt.show()

# Inverted Pendulum section

# ren_urwd_16_file = pd.read_csv('../ray_results/URwd_PPO/ProjRENModel_InvertedPendulumEnv_phiTanh_state2_hidden16_1_hidden_size=16_2022-03-25_15-46-21/progress.csv')
# rnn_urwd_16_file = pd.read_csv('../ray_results/URwd_PPO/ProjRNNModel_InvertedPendulumEnv_phiTanh_state2_hidden16_1_hidden_size=16_2022-03-25_15-48-48/progress.csv')

# ren_num_env_steps_sampled = ren_urwd_16_file['num_env_steps_sampled'].values
# ren_rwds = ren_urwd_16_file['episode_reward_mean'].values

# rnn_num_env_steps_sampled = rnn_urwd_16_file['num_env_steps_sampled'].values
# rnn_rwds = rnn_urwd_16_file['episode_reward_mean'].values

# plt.figure()
# plt.plot(ren_num_env_steps_sampled, ren_rwds)
# plt.plot(rnn_num_env_steps_sampled, rnn_rwds)
# plt.legend(["REN", "RNN"])
# plt.title('REN and RNN: Reward vs No. Plant Step Samples')
# plt.xlabel('Number of Plant Step Samples')
# plt.ylabel('Reward')
# plt.show()

# Neural net plant section

# learned_plant_file = pd.read_csv('../ray_results/StableRwd_PPO/ProjRENModel_LearnedInvertedPendulumEnv_phiTanh_state2_hidden16_0_2022-03-25_15-42-07/progress.csv')
# true_plant_file = pd.read_csv('../ray_results/StableRwd_PPO/ProjRENModel_InvertedPendulumEnv_phiTanh_state2_hidden16_1_hidden_size=16_2022-03-25_15-29-16/progress.csv')

# Learned B2 as well
# learned_plant_file = pd.read_csv('../ray_results/StableRwd_PPO/ProjRENModel_LearnedInvertedPendulumEnv_phiTanh_state2_hidden16_0_2022-03-26_20-00-50/progress.csv')
# true_plant_file    = pd.read_csv('../ray_results/StableRwd_PPO/ProjRENModel_InvertedPendulumEnv_phiTanh_state2_hidden16_0_2022-03-26_20-02-35/progress.csv')

# learned_plant_steps = num_env_steps_sampled(learned_plant_file)
# true_plant_steps = num_env_steps_sampled(true_plant_file)

# learned_plant_rwd = reward(learned_plant_file)
# true_plant_rwd = reward(true_plant_file)

# learned_plant_rollout_len = rollout_len(learned_plant_file)
# true_plant_rollout_len = rollout_len(true_plant_file)

# plt.figure()
# plt.plot(learned_plant_steps, learned_plant_rwd)
# plt.plot(true_plant_steps, true_plant_rwd)
# plt.legend(["Learned Plant", "True Plant"])
# plt.xlabel('Number of Plant Step Samples')
# plt.ylabel('Reward')
# plt.title('Controller Reward: Learned Plant vs True Plant')
# plt.show()

# plt.figure()
# plt.plot(learned_plant_steps, learned_plant_rollout_len)
# plt.plot(true_plant_steps, true_plant_rollout_len)
# plt.legend(["Learned Plant", "True Plant"])
# plt.xlabel('Number of Plant Step Samples')
# plt.ylabel('Rollout Length')
# plt.show()