import ray
from ray import tune
from envs import CartpoleEnv, InvertedPendulumEnv, LinearizedInvertedPendulumEnv, PendubotEnv, VehicleLateralEnv, PowergridEnv
from models.RNN import RNNModel
from models.ProjRNN import ProjRNNModel
from models.ProjREN import ProjRENModel
from ray.rllib.agents import ppo, pg
import numpy as np
from ray.tune.schedulers import ASHAScheduler
import optuna
from ray.tune.suggest.optuna import OptunaSearch
from trainers import ProjectedPGTrainer, ProjectedPPOTrainer
import torch
import torch.nn as nn
from deq_lib.solvers import broyden, anderson
import os
import math
import json
import matplotlib.pyplot as plt

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
    "<class 'models.RNN.RNNModel'>": RNNModel
}

phi_map = {
    "<class 'torch.nn.modules.activation.Tanh'>": nn.Tanh
}

# directory = '../ray_results/InvPend_BoundedActionReward_PPO/ProjRNNModel_InvertedPendulumEnv_0_2022-03-06_23-04-58'
directory = '../ray_results/InvPend_BoundedActionReward_PPO/ProjRENModel_InvertedPendulumEnv_0_2022-03-06_23-01-13'
checkpoint_path = directory + '/checkpoint_001000/checkpoint-1000'

config_file = open(directory + '/params.json', 'r')
config = json.load(config_file)

config['env'] = env_map[config['env']]
config['model']['custom_model_config']['plant_cstor'] = config['env']
config['model']['custom_model'] = model_map[config['model']['custom_model']]

if config['model']['custom_model_config']['phi_cstor'] == "<class 'torch.nn.modules.activation.Tanh'>":
    config['model']['custom_model_config']['phi_cstor'] = nn.Tanh
    config['model']['custom_model_config']['A_phi'] = torch.tensor(0)
    config['model']['custom_model_config']['B_phi'] = torch.tensor(1)

if 'broyden' in config['model']['custom_model_config']['solver']:
    config['model']['custom_model_config']['solver'] = broyden
else:
    config['model']['custom_model_config']['solver'] = anderson

config['num_workers'] = 7

agent = ProjectedPPOTrainer(config = config)
agent.restore(checkpoint_path)
env = config['env'](config['env_config'])
N_iters = 100
all_states = []
all_actions = []

for _ in range(N_iters):
    obs = env.reset()
    policy_state = agent.get_policy().get_initial_state()
    prev_rew = 0
    prev_act = np.zeros_like(env.action_space.sample())
    rewards = []
    actions = []
    states = []
    done = False
    while not done:
        states.append(env.state)
        action, policy_state, _ = agent.compute_single_action(
            obs, state = policy_state, prev_reward = prev_rew, prev_action = prev_act, explore = False
        )
        obs, rew, done, info = env.step(action)
        actions.append(action)
        rewards.append(rew)
        prev_rew = rew

    states = np.stack(states).T
    actions = np.stack(actions).T

    all_states.append(states)
    all_actions.append(actions)


bounds = [env.soft_max_torque for _ in range(31)]

plt.figure()

plt.subplot(311)
for i in range(N_iters):
    plt.plot(all_states[i][0])
plt.title("theta")

plt.subplot(312)
for i in range(N_iters):
    plt.plot(all_states[i][1])
plt.title("theta dot")

plt.subplot(313)
for i in range(N_iters):
    plt.plot(all_actions[i][0])
plt.title("u")
plt.plot(bounds, linestyle='dashed')
plt.plot([-x for x in bounds], linestyle='dashed')

plt.show()