"""
Main file for configuring and training controllers.
"""

import math
import os
import multiprocessing

import numpy as np
import ray
import torch
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from envs import FlexibleArmEnv, InvertedPendulumEnv

import lti_controllers

from models import (
    RINN,
    RNN,
    DissipativeRINN,
    DissipativeSimplestRINN,
    DissipativeThetaRINN,
    FullyConnectedNetwork,
    ImplicitModel,
    LTIModel
)
from trainers import ProjectedPPOTrainer

# N_CPUS = 1  # test
N_CPUS = multiprocessing.cpu_count() / 2
# N_CPUS  = int(os.getenv('SLURM_CPUS_ON_NODE'))
n_tasks = 1
n_workers_per_task = int(math.floor(N_CPUS / n_tasks)) - 1 - 1

# Same dt must be used in RNN and RINN and DissipativeRINN
dt = 0.01
env = InvertedPendulumEnv
env_config = {
    "observation": "partial",
    "normed": True,
    "factor": 1,
    "dt": dt,
    "supply_rate": "stability",
    "disturbance_model": "occasional"
}
# env = FlexibleArmEnv
# env_config = {
#     "observation": "full",
#     "normed": True,
#     "factor": 1,
#     "dt": dt,
#     "supply_rate": "stability",
#     "disturbance_model": "none",
#     "design_model": "flexible",
# }

# Configure the algorithm.
config = {
    "env": env,
    "env_config": env_config,
    "model": {
        # "custom_model": FullyConnectedNetwork,
        # "custom_model_config": {
        #     "n_layers": 2,
        #     "size": 16
        # }
        # "custom_model": ImplicitModel,
        # "custom_model_config": {"state_size": 16},
        # "custom_model": RINN,
        # "custom_model_config": {
        #     "state_size": 2,
        #     "nonlin_size": 16,
        #     "dt": dt,
        #     "log_std_init": np.log(1.0)
        # }
        # "custom_model": DissipativeRINN,
        # "custom_model_config": {
        #     "state_size": 2,
        #     "nonlin_size": 16,
        #     "log_std_init": np.log(1.0),
        #     "dt": dt,
        #     "plant": env,
        #     "plant_config": env_config,
        #     "eps": 1e-3,
        #     "trs_mode": "fixed",
        #     "min_trs": 1.44,
        # }
        "custom_model": RINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": 16,
            "log_std_init": np.log(1.0),
            "dt": dt,
            "plant": env,
            "plant_config": env_config,
            "eps": 1e-3,
        }
        # "custom_model": DissipativeThetaRINN,
        # "custom_model_config": {
        #     "state_size": 2,
        #     "nonlin_size": 16,
        #     "log_std_init": np.log(1.0),
        #     "dt": dt,
        #     "plant": env,
        #     "plant_config": env_config,
        #     "eps": 1e-3,
        #     "trs_mode": "fixed",
        #     "min_trs": 1.44,
        #     "project_delay": 1,
        #     "project_spacing": 1, # 10,
        # }
#         "custom_model": DissipativeSimplestRINN,
#         "custom_model_config": {
#             "state_size": 2,
#             "nonlin_size": 16,
#             "log_std_init": np.log(1.0),
#             "dt": dt,
#             "plant": env,
#             "plant_config": env_config,
#             "eps": 1e-3,
#             "mode": "simple",
# #             "P": np.array([[ 1.04159083e+02, -6.56387889e-01,  1.15737991e+01, -1.09562663e-02],
# #  [-6.56387889e-01,  2.00579719e-02, -1.73840137e-01, -7.29584950e-01],
# #  [ 1.15737991e+01, -1.73840137e-01,  2.42446125e+00,  8.20153731e+00],
# #  [-1.09562663e-02, -7.29584950e-01,  8.20153731e+00,  6.03135330e+01]]),
#             # "lti_initializer": "dissipative_theta",
#             # "lti_initializer_kwargs": {
#             #     "eps": 1e-3,
#                 ## "P0": np.array([[ 1.04159083e+02, -6.56387889e-01,  1.15737991e+01, -1.09562663e-02],
#                 ##     [-6.56387889e-01,  2.00579719e-02, -1.73840137e-01, -7.29584950e-01],
#                 ##     [ 1.15737991e+01, -1.73840137e-01,  2.42446125e+00,  8.20153731e+00],
#                 ##     [-1.09562663e-02, -7.29584950e-01,  8.20153731e+00,  6.03135330e+01]]),
#                 # "P0": np.array([[ 0.4622, -0.0031,  0.1563, -0.0719],
#                 #     [-0.0031,  0.0089, -0.0332, -0.0791],
#                 #     [ 0.1563, -0.0332,  0.2260,  0.4052],
#                 #     [-0.0719, -0.0791,  0.4052,  1.3935]])
#             # }
#             "lti_initializer": "dissipative_thetahat",
#             "lti_initializer_kwargs": {
#                 "trs_mode": "fixed",
#                 "min_trs": 1.44
#             }
#         }
        # "custom_model": LTIModel,
        # "custom_model_config": {
        #     "dt": dt,
        #     "plant": env,
        #     "plant_config": env_config,
        #     "lti_controller": "lqr",
        #     "lti_controller_kwargs": {
        #         "Q": np.eye(2, dtype=np.float32),
        #         "R": np.array([[1]], dtype=np.float32)
        #     },
        #     "learn": False,
        #     "log_std_init": np.log(1.0),
        # }
    },
    "lr": 1e-3,
    "num_workers": n_workers_per_task,
    "framework": "torch",
    "num_gpus": 0,  # 1,
    "evaluation_num_workers": 1,
    "evaluation_config": {"render_env": False, "explore": False},
    "evaluation_interval": 1,
    "evaluation_parallel_to_training": True,
}

print("==================================")
print("Number of workers per task: ", n_workers_per_task)
print("")

print("Config: ")
print(config)
print("")

test_env = env(env_config)
print(
    f"Max reward per: step: {test_env.max_reward}, rollout: {test_env.max_reward*(test_env.time_max+1)}"
)
print("==================================")


def name_creator(trial):
    config = trial.config
    name = f"{config['env'].__name__}"
    name += f"_{config['model']['custom_model'].__name__}"
    return name


ray.init()
results = tune.run(
    # PPOTrainer,
    ProjectedPPOTrainer,
    config=config,
    stop={
        "agent_timesteps_total": 6100e3,
    },
    verbose=1,
    trial_name_creator=name_creator,
    name="scratch",
    local_dir="ray_results",
    checkpoint_at_end=True,
    checkpoint_freq=100,
)
