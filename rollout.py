import ray
from ray import tune
from envs import CartpoleEnv, InvertedPendulumEnv, LinearizedInvertedPendulumEnv, PendubotEnv, VehicleLateralEnv, PowergridEnv
from models.RNN import RNNModel
from models.ProjRNN import ProjRNNModel
from models.ProjREN import ProjRENModel
from models.ProjRNNOld import ProjRNNOldModel
from ray.rllib.agents import ppo, pg
import numpy as np
from trainers import ProjectedPGTrainer, ProjectedPPOTrainer
from activations import LeakyReLU, Tanh
import torch
import torch.nn as nn
from deq_lib.solvers import broyden, anderson
import json
import matplotlib.pyplot as plt
import copy

env_map = {
    "<class 'envs.inverted_pendulum.InvertedPendulumEnv'>": InvertedPendulumEnv,
    "<class 'envs.linearized_inverted_pendulum.LinearizedInvertedPendulumEnv'>": LinearizedInvertedPendulumEnv,
    "<class 'envs.cartpole.CartpoleEnv'>": CartpoleEnv,
    "<class 'envs.pendubot.PendubotEnv'>": PendubotEnv,
    "<class 'envs.vehicle.VehicleLateralEnv'>": VehicleLateralEnv,
    "<class 'envs.powergrid.PowergridEnv'>": PowergridEnv
}

model_map = {
    "<class 'models.ProjREN.ProjRENModel'>": ProjRENModel,
    "<class 'models.ProjRNN.ProjRNNModel'>": ProjRNNModel,
    "<class 'models.RNN.RNNModel'>": RNNModel,
    "<class 'models.ProjRNNOld.ProjRNNOldModel'>": ProjRNNOldModel,
}

phi_map = {
    "<class 'activations.LeakyReLU'>": LeakyReLU,
    "<class 'activations.Tanh'>": Tanh,
}

def load_agent(directory, checkpoint_path = None):
    if checkpoint_path is None:
        checkpoint_path = directory + '/checkpoint_001000/checkpoint-1000'

    config_file = open(directory + '/params.json', 'r')
    config = json.load(config_file)

    config['env'] = env_map[config['env']]
    config['model']['custom_model_config']['plant_cstor'] = config['env']
    config['model']['custom_model'] = model_map[config['model']['custom_model']]

    config['model']['custom_model_config']['phi_cstor'] = phi_map[config['model']['custom_model_config']['phi_cstor']]

    if 'broyden' in config['model']['custom_model_config']['solver']:
        config['model']['custom_model_config']['solver'] = broyden
    else:
        config['model']['custom_model_config']['solver'] = anderson

    config['num_workers'] = 3

    agent = ProjectedPPOTrainer(config = config)
    agent.restore(checkpoint_path)

    env = config['env'](config['env_config'])

    return agent, env

def compute_rollout(agent, env, init_obs):
    obs = init_obs
    policy_state = agent.get_policy().get_initial_state()
    prev_rew = 0
    prev_act = np.zeros_like(env.action_space.sample())
    rewards = []
    actions = []
    states = []
    done = False
    n_steps = 0
    while not done:
        states.append(env.state)
        action, policy_state, _ = agent.compute_single_action(
            obs, state = policy_state, prev_reward = prev_rew, prev_action = prev_act, explore = False
        )
        obs, rew, done, info = env.step(action)
        actions.append(action)
        rewards.append(rew)
        prev_rew = rew
        n_steps += 1
    if n_steps < env.time_max+1:
        print('failure', agent.__name__)

    states = np.stack(states).T
    actions = np.stack(actions).T

    return states, actions

# rnn_dir = '../ray_results/InvPend_BoundedActionReward_PPO/ProjRNNModel_InvertedPendulumEnv_0_2022-03-06_23-04-58'
# ren_dir = '../ray_results/InvPend_BoundedActionReward_PPO/ProjRENModel_InvertedPendulumEnv_0_2022-03-06_23-01-13'

# ren_dir = '../ray_results/InvPend_BoundedActionReward_ShortRollout_PPO/ProjRENModel_InvertedPendulumEnv_0_2022-03-07_13-47-48'
# rnn_dir = '../ray_results/InvPend_BoundedActionReward_ShortRollout_PPO/ProjRNNModel_InvertedPendulumEnv_0_2022-03-07_13-45-25'

# ren_dir = '../ray_results/scratch/ProjRENModel_InvertedPendulumEnv_0_2022-03-07_23-34-32' # checkpoint_path=ren_dir + '/checkpoint_000010/checkpoint-10'
# rnn_dir = '../ray_results/scratch/ProjRNNModel_InvertedPendulumEnv_0_2022-03-07_23-17-20' # checkpoint_path=rnn_dir + '/checkpoint_000100/checkpoint-100'

# ren_dir = '../ray_results/Vehicle_BoundedActionReward_PPO/ProjRENModel_VehicleLateralEnv_2022-03-08_00-06-53'
# rnn_dir = '../ray_results/Vehicle_BoundedActionReward_PPO/ProjRNNModel_VehicleLateralEnv_2022-03-08_00-07-05'

# ren_dir = '../ray_results/BoundedActionReward_PPO/ProjRENModel_InvertedPendulumEnv_state2_hidden1_0_2022-03-11_12-36-41'
# rnn_dir = '../ray_results/BoundedActionReward_PPO/ProjRNNModel_InvertedPendulumEnv_state2_hidden1_0_2022-03-11_12-37-15'

# ren_dir = '../ray_results/scratch/ProjRNNModel_InvertedPendulumEnv_state2_hidden1_0_2022-03-11_22-25-33'
# rnn_dir = '../ray_results/scratch/ProjRENModel_InvertedPendulumEnv_state2_hidden1_0_2022-03-11_22-28-53'

# ren_dir = "../ray_results/ComboRwd_PPO/ProjRENModel_VehicleLateralEnv_phiLeakyReLU_state4_hidden1_1_phi_cstor=<class 'activations.LeakyReLU'>_2022-03-11_23-27-06"
# rnn_dir = "../ray_results/ComboRwd_PPO/ProjRNNModel_VehicleLateralEnv_phiTanh_state4_hidden1_0_phi_cstor=<class 'activations.Tanh'>_2022-03-11_23-28-29"

ren_dir = "../ray_results/ComboRwd_PPO/ProjRENModel_InvertedPendulumEnv_phiLeakyReLU_state2_hidden1_2_custom_model=<class 'models.ProjREN.ProjRENModel'>,phi_cstor=<class_2022-03-12_14-27-47"
rnn_dir = "../ray_results/ComboRwd_PPO/ProjRNNModel_InvertedPendulumEnv_phiLeakyReLU_state2_hidden1_3_custom_model=<class 'models.ProjRNN.ProjRNNModel'>,phi_cstor=<class_2022-03-12_14-27-55"

ren_agent, _ = load_agent(ren_dir, checkpoint_path=ren_dir + '/checkpoint_002000/checkpoint-2000')
rnn_agent, env = load_agent(rnn_dir, checkpoint_path=rnn_dir + '/checkpoint_002000/checkpoint-2000')#, checkpoint_path=rnn_dir + '/checkpoint_000010/checkpoint-10')

N_iters = 10000
ren_states = []
ren_actions = []
rnn_states = []
rnn_actions = []

for _ in range(N_iters):
    obs = env.reset()
    env_copy = copy.deepcopy(env)

    ren_state, ren_action = compute_rollout(ren_agent, env, obs)
    assert ren_state.shape[1] == env.time_max + 1, 'fail long'
    ren_states.append(ren_state)
    ren_actions.append(ren_action)

    rnn_state, rnn_action = compute_rollout(rnn_agent, env_copy, obs)
    rnn_states.append(rnn_state)
    rnn_actions.append(rnn_action)


# bounds = [env.soft_max_steering for _ in range(env.time_max+1)]
bounds = [env.soft_max_torque for _ in range(env.time_max+1)]

plt.figure()

plt.subplot(311)
for i in range(N_iters):
    plt.plot(ren_states[i][0])
    plt.plot(rnn_states[i][0])
plt.title("theta")

plt.subplot(312)
for i in range(N_iters):
    plt.plot(ren_states[i][1])
    plt.plot(rnn_states[i][1])
plt.title("theta dot")

plt.subplot(313)
for i in range(N_iters):
    plt.plot(ren_actions[i][0], 'tab:orange')
    plt.plot(rnn_actions[i][0], 'tab:blue')
plt.title("u")
plt.plot(bounds, linestyle='dashed')
plt.plot([-x for x in bounds], linestyle='dashed')

plt.show()