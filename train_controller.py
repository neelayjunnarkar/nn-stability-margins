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

from envs import FlexibleArmEnv, InvertedPendulumEnv, TimeDelayInvertedPendulumEnv, DiskMarginExampleEnv, FlexibleArmDiskMarginEnv

import lti_controllers

from models import (
    RINN,
    RNN,
    DissipativeRINN,
    DissipativeSimplestRINN,
    FullyConnectedNetwork,
    ImplicitModel,
    LTIModel,
)
from trainers import ProjectedPPOTrainer


use_savio = False
if use_savio:
    print("\n\n\nUsing Savio\n===========\n\n\n")
    N_CPUS = int(os.getenv("SLURM_CPUS_ON_NODE"))
    JOB_ID = os.getenv("SLURM_JOB_ID")
else:
    # N_CPUS = 1 # test
    # N_CPUS = 2 # test
    N_CPUS = multiprocessing.cpu_count()
n_tasks = 1
n_workers_per_task = int(math.floor(N_CPUS / n_tasks)) - 1 - 1

seed = 0


# Same dt must be used in controller models
# dt = 0.01
# env = InvertedPendulumEnv
# env_config = {
#     "observation": "partial",
#     "normed": True,
#     "dt": dt,
#     "supply_rate": "l2_gain", # "stability",
#     "disturbance_model": "occasional"
# }
# dt = 0.01
# env = TimeDelayInvertedPendulumEnv  # 1.0
# env_config = {
#     "observation": "partial",
#     "normed": True,
#     "dt": dt,
#     "design_time_delay": 0.07,
#     "time_delay_steps": 5,
# }
# dt = 0.001 # 0.0001
# env = FlexibleArmEnv
# env_config = {
#     "observation": "full",
#     "normed": True,
#     "dt": dt,
#     "rollout_length": int(2.5/dt)-1, # 10000,
#     "supply_rate": "l2_gain",
#     "disturbance_model": "none",
#     "disturbance_design_model": "occasional",
#     "design_model": "rigid",
# }
# dt = 0.001
# env = FlexibleArmEnv
# env_config = {
#     "observation": "partial",
#     "normed": True,
#     "dt": dt,
#     "rollout_length": int(2 / dt) - 1,
#     "supply_rate": "l2_gain",
#     "disturbance_model": "occasional",
#     "disturbance_design_model": "occasional",
#     "design_model": "rigidplus_integrator",
#     "delta_alpha": 1.0,
#     # "design_integrator_type": "utox2",
#     # "supplyrate_scale": 0.5,
#     # "lagrange_multiplier": 5,
#     "design_integrator_type": "utoy",
#     "supplyrate_scale": 1,
#     "lagrange_multiplier": 1000,
# }
# dt = 0.001
# env = DiskMarginExampleEnv
# env_config = {
#     "dt": dt,
#     "seed": seed,
# }
dt = 0.001
env = FlexibleArmDiskMarginEnv
env_config = {
    "dt": dt,
    "seed": seed,
    "normed": True,
    "rollout_length": int(2 / dt) - 1,
    "disturbance_model": "occasional",
    "disk_margin_type": "6dB36deg",
    # "skew": 0,
    # "alpha": 0,
}

# Configure the algorithm.
config = {
    "env": env,
    "env_config": env_config,
    "model": {
        # "custom_model": FullyConnectedNetwork,
        # "custom_model_config": {
        #     "n_layers": 2,
        #     "size": 19
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
        #     "state_size": 4,
        #     "nonlin_size": 16,
        #     "log_std_init": np.log(1.0),
        #     "dt": dt,
        #     "plant": env,
        #     "plant_config": env_config,
        #     "eps": 1e-3,
        #     "trs_mode": "fixed",
        #     "min_trs": 1.0,
        #     "backoff_factor": 1.1,
        #     "lti_initializer": "dissipative_thetahat",
        #     "lti_initializer_kwargs": {
        #         "trs_mode": "fixed",
        #         "min_trs": 1.0,
        #         "backoff_factor": 1.1,
        #     }
        # }
        # "custom_model": RINN,
        # "custom_model_config": {
        #     "state_size": 2,
        #     "nonlin_size": 16,
        #     "log_std_init": np.log(1.0),
        #     "dt": dt,
        #     "plant": env,
        #     "plant_config": env_config,
        #     "eps": 1e-3,
        # },
        "custom_model": DissipativeSimplestRINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": 16,
            "log_std_init": np.log(1.0),
            "dt": dt,
            "plant": env,
            "plant_config": env_config,
            "eps": 1e-3,
            "mode": "thetahat",
            "trs_mode": "fixed",
            "min_trs": 1,
            "backoff_factor": 1.05,
            "lti_initializer": "dissipative_thetahat",
            "lti_initializer_kwargs": {
                "trs_mode": "fixed",
                "min_trs": 1,
                "backoff_factor": 1.05,
            },
            "fix_mdeltap": False
        },
        # "custom_model": LTIModel,
        # "custom_model_config": {
        #     "dt": dt,
        #     "plant": env,
        #     "plant_config": env_config,
        #     "learn": True,
        #     "log_std_init": np.log(1.0),
        #     "state_size": 2,
        #     "trs_mode": "fixed",
        #     "min_trs": 1.5,  # 1.5, # 1.44,
        #     "lti_controller": "dissipative_thetahat",
        #     "lti_controller_kwargs": {
        #         "trs_mode": "fixed",
        #         "min_trs": 1.5,  # 1.5 # 1.44
        #     },
        # },
    },
    ## Testing changes to training parameters
    "sgd_minibatch_size": 2048,
    "train_batch_size": 20480,
    "lr": 2e-4,
    "num_envs_per_worker": 10,
    ## End test
    "seed": seed,
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
    if use_savio:
        name += f"_{JOB_ID}"
    return name


ray.init()
results = tune.run(
    # PPOTrainer,
    ProjectedPPOTrainer,
    config=config,
    stop={
        "agent_timesteps_total": 1e7,
    },
    verbose=1,
    trial_name_creator=name_creator,
    name="scratch",
    local_dir="ray_results",
    checkpoint_at_end=True,
    checkpoint_freq=1000,
)
