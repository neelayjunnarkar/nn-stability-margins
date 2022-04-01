"""
Reconstructing policies from checkpoints to create plots with rollouts.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
import torch

from envs import CartpoleEnv, InvertedPendulumEnv, LinearizedInvertedPendulumEnv, PendubotEnv, VehicleLateralEnv, PowergridEnv
from models import ProjRENModel, ProjRNNModel, ProjRNNOldModel
from activations import LeakyReLU, Tanh
from deq_lib.solvers import broyden, anderson
from trainers import ProjectedPGTrainer, ProjectedPPOTrainer

env_map = {
    "<class 'envs.inverted_pendulum.InvertedPendulumEnv'>": InvertedPendulumEnv,
    "<class 'envs.linearized_inverted_pendulum.LinearizedInvertedPendulumEnv'>": LinearizedInvertedPendulumEnv,
    "<class 'envs.cartpole.CartpoleEnv'>": CartpoleEnv,
    "<class 'envs.pendubot.PendubotEnv'>": PendubotEnv,
    "<class 'envs.vehicle.VehicleLateralEnv'>": VehicleLateralEnv,
    "<class 'envs.powergrid.PowergridEnv'>": PowergridEnv,
    "<class 'envs.learned_inverted_pendulum.LearnedInvertedPendulumEnv'>": InvertedPendulumEnv
}

model_map = {
    "<class 'models.ProjREN.ProjRENModel'>": ProjRENModel,
    "<class 'models.ProjRNN.ProjRNNModel'>": ProjRNNModel,
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
        obs, rew, done, info = env.step(action, fail_on_state_space = False)
        actions.append(action)
        rewards.append(rew)
        prev_rew = rew
        n_steps += 1
    if n_steps < env.time_max+1:
        print('failure')

    states = np.stack(states).T
    actions = np.stack(actions).T

    return states, actions

# Phase portraits

def phase_portrait(agent_dir, N_PER_DIM, ROLLOUT_LEN):
    agent, env = load_agent(agent_dir, checkpoint_path=agent_dir + "/checkpoint_001667/checkpoint-1667")

    rollouts = torch.zeros(N_PER_DIM, N_PER_DIM, ROLLOUT_LEN, env.state_size)
    state_max = env.state_space.high/0.8
    theta_points = torch.linspace(-state_max[0], state_max[0], N_PER_DIM)
    thetadot_points = torch.linspace(-state_max[1], state_max[1], N_PER_DIM)
    for i in range(N_PER_DIM):
        for j in range(N_PER_DIM):
            print(f'{i}, {j}')
            init_obs = env.reset(np.array([theta_points[i], thetadot_points[j]]))
            states, actions = compute_rollout(agent, env, init_obs)
            rollouts[i, j] = torch.from_numpy(states.T)
    return rollouts, env

N_PER_DIM = 7
ROLLOUT_LEN = 200 + 1

# true_agent_dir = '../ray_results/StableRwd_PPO/ProjRENModel_InvertedPendulumEnv_phiTanh_state2_hidden16_1_hidden_size=16_2022-03-25_15-29-16'
# learned_agent_dir = '../ray_results/StableRwd_PPO/ProjRENModel_LearnedInvertedPendulumEnv_phiTanh_state2_hidden16_0_2022-03-25_15-42-07'

# Learned B2 as well
learned_agent_dir = '../ray_results/StableRwd_PPO/ProjRENModel_LearnedInvertedPendulumEnv_phiTanh_state2_hidden16_0_2022-03-26_20-00-50'
true_agent_dir    = '../ray_results/StableRwd_PPO/ProjRENModel_InvertedPendulumEnv_phiTanh_state2_hidden16_0_2022-03-26_20-02-35'

true_pp, _ = phase_portrait(true_agent_dir, N_PER_DIM, ROLLOUT_LEN)
learned_pp, env = phase_portrait(learned_agent_dir, N_PER_DIM, ROLLOUT_LEN)

plt.figure()
plt.subplot(121, title = 'True Plant Model', xlim=[-np.pi, np.pi], ylim=[-8, 8])
for i in range(N_PER_DIM):
    for j in range(N_PER_DIM):
        # print(f'{i}, {j}')
        rollout = true_pp[i, j]
        if torch.allclose(torch.zeros(env.state_size), rollout[-1]):
            color = 'C2' # green
        else:
            color = 'C3' # red
        plt.plot(rollout[:, 0], rollout[:, 1], color = color)
plt.xlabel('x1 (radians)')
plt.ylabel('x2 (radians/second)')

plt.subplot(122, title = 'Learned Plant Model', xlim=[-np.pi, np.pi], ylim=[-8, 8])
for i in range(N_PER_DIM):
    for j in range(N_PER_DIM):
        # print(f'{i}, {j}')
        rollout = learned_pp[i, j]
        if torch.allclose(torch.zeros(env.state_size), rollout[-1]):
            color = 'C2' # green
        else:
            color = 'C3' # red
        plt.plot(rollout[:, 0], rollout[:, 1], color = color)
plt.xlabel('x1 (radians)')
plt.show()

# Plotting rollouts of agents against each other from the same initial conditions.

# ren_dir = "../ray_results/Learned_InvPend/ProjRENModel_LearnedInvertedPendulumEnv_phiTanh_state2_hidden4_0_2022-03-25_10-08-38"
# rnn_dir = "../ray_results/Learned_InvPend/ProjRENModel_InvertedPendulumEnv_phiTanh_state2_hidden4_0_2022-03-25_10-08-52"

# ren_agent, _ = load_agent(ren_dir, checkpoint_path=ren_dir + "/checkpoint_000084/checkpoint-84")
# rnn_agent, env = load_agent(rnn_dir, checkpoint_path=rnn_dir + "/checkpoint_000084/checkpoint-84")#, checkpoint_path=rnn_dir + '/checkpoint_000010/checkpoint-10')

# # x vs t

# N_iters = 10
# ren_states = []
# ren_actions = []
# rnn_states = []
# rnn_actions = []

# for _ in range(N_iters):
#     obs = env.reset()
#     env_copy = copy.deepcopy(env)

#     ren_state, ren_action = compute_rollout(ren_agent, env, obs)
#     # assert ren_state.shape[1] == env.time_max + 1, 'fail long'
#     ren_states.append(ren_state)
#     ren_actions.append(ren_action)

#     rnn_state, rnn_action = compute_rollout(rnn_agent, env_copy, obs)
#     rnn_states.append(rnn_state)
#     rnn_actions.append(rnn_action)

# plt.figure()

# plt.subplot(311)
# for i in range(N_iters):
#     plt.plot(ren_states[i][0])
#     # plt.plot(rnn_states[i][0])
# plt.title("theta")

# plt.subplot(312)
# for i in range(N_iters):
#     plt.plot(ren_states[i][1])
#     # plt.plot(rnn_states[i][1])
# plt.title("theta dot")

# plt.subplot(313)
# for i in range(N_iters):
#     plt.plot(ren_actions[i][0]) #, 'tab:orange')
#     # plt.plot(rnn_actions[i][0], 'tab:blue')
# plt.title("u")

# plt.show()