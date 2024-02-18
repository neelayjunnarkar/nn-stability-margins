"""
Main file for configuring and training controllers on Savio with array job
"""

import math
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from envs import FlexibleArmEnv, InvertedPendulumEnv, TimeDelayInvertedPendulumEnv
from models import (
    RINN,
    RNN,
    DissipativeRINN,
    DissipativeSimplestRINN,
    DissipativeThetaRINN,
    FullyConnectedNetwork,
    ImplicitModel,
    LTIModel,
)
from trainers import ProjectedPPOTrainer

N_CPUS = int(os.getenv("SLURM_CPUS_ON_NODE"))
JOB_ID = os.getenv("SLURM_JOB_ID")
TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID"))

n_tasks = 1
n_workers_per_task = int(math.floor(N_CPUS / n_tasks)) - 1 - 1

# ## Env Config by Task
# T = None
# if TASK_ID == 0:
#     T = 2
# elif TASK_ID == 1:
#     T = 5
# elif TASK_ID == 2:
#     T = 2
# elif TASK_ID == 3:
#     T = 5
# else:
#     raise ValueError(f"Task ID {TASK_ID} unexpected.")

# assert T is not None

# Same dt must be used in the controller model (RNN and RINN and DissipativeRINN)
# dt = 0.01
# env = InvertedPendulumEnv
# env_config = {
#     "observation": "partial",
#     "normed": True,
#     "dt": dt,
#     "supply_rate": "l2_gain",  # "stability",
#     "disturbance_model": "occasional",
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
# dt = 0.001  # 0.0001
# env = FlexibleArmEnv
# env_config = {
#     "observation": "full",
#     "normed": True,
#     "dt": dt,
#     "rollout_length": int(T / dt) - 1,  # 10000,
#     "supply_rate": "l2_gain",
#     "disturbance_model": "none",
#     "disturbance_design_model": "occasional",
#     "design_model": "rigid",
# }
dt = 0.001
env = FlexibleArmEnv
env_config = {
    "observation": "partial",
    "normed": True,
    "dt": dt,
    "rollout_length": int(2 / dt) - 1,
    "supply_rate": "l2_gain",
    "disturbance_model": "none",
    "disturbance_design_model": "occasional",
    "design_model": "rigidplus",  # trs in [1, 2] seem kind of the same... Maybe use 1.5.
}

## Model Config by Task

custom_model = None
custom_model_config = None
learning_rate = 1e-3
trainer = ProjectedPPOTrainer
if TASK_ID == 0:
    custom_model = DissipativeSimplestRINN
    custom_model_config = {
        "state_size": 4,
        "nonlin_size": 16,
        "log_std_init": np.log(1.0),
        "dt": dt,
        "plant": env,
        "plant_config": env_config,
        "eps": 1e-3,
        "mode": "thetahat",
        "trs_mode": "fixed",
        "min_trs": 1.5,
        "lti_initializer": "dissipative_thetahat",
        "lti_initializer_kwargs": {
            "trs_mode": "fixed",
            "min_trs": 1.5,
        },
    }
elif TASK_ID == 1:
    custom_model = DissipativeSimplestRINN
    custom_model_config = {
        "state_size": 4,
        "nonlin_size": 16,
        "log_std_init": np.log(1.0),
        "dt": dt,
        "plant": env,
        "plant_config": env_config,
        "eps": 1e-3,
        "mode": "thetahat",
        "trs_mode": "fixed",
        "min_trs": 1.5,
        "lti_initializer": "dissipative_thetahat",
        "lti_initializer_kwargs": {
            "trs_mode": "fixed",
            "min_trs": 1.5,
        },
    }
    learning_rate = 1e-2
elif TASK_ID == 2:
    custom_model = LTIModel
    custom_model_config = {
        "dt": dt,
        "plant": env,
        "plant_config": env_config,
        "learn": True,
        "log_std_init": np.log(1.0),
        "state_size": 4,
        "trs_mode": "fixed",
        "min_trs": 1.5,
        "lti_controller": "dissipative_thetahat",
        "lti_controller_kwargs": {
            "trs_mode": "fixed",
            "min_trs": 1.5,
        },
    }
elif TASK_ID == 3:
    custom_model = LTIModel
    custom_model_config = {
        "dt": dt,
        "plant": env,
        "plant_config": env_config,
        "learn": True,
        "log_std_init": np.log(1.0),
        "state_size": 4,
        "trs_mode": "fixed",
        "min_trs": 1.5,
        "lti_controller": "dissipative_thetahat",
        "lti_controller_kwargs": {
            "trs_mode": "fixed",
            "min_trs": 1.5,
        },
    }
    learning_rate = 1e-2
# elif TASK_ID == 4:
#     custom_model = FullyConnectedNetwork
#     custom_model_config = {"n_layers": 2, "size": 19}
#     trainer = PPOTrainer
else:
    raise ValueError(f"Task ID {TASK_ID} unexpected.")

assert custom_model is not None
assert custom_model_config is not None
assert learning_rate is not None
assert trainer is not None

# Configure the algorithm.
config = {
    "env": env,
    "env_config": env_config,
    "model": {
        "custom_model": custom_model,
        "custom_model_config": custom_model_config,
    },
    "lr": learning_rate,
    "num_workers": n_workers_per_task,
    "framework": "torch",
    "num_gpus": 0,  # 1,
    "evaluation_num_workers": 1,
    "evaluation_config": {"render_env": False, "explore": False},
    "evaluation_interval": 1,
    "evaluation_parallel_to_training": True,
}

print("==================================")
print(f"Job: {JOB_ID}, Task: {TASK_ID}")
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
    name += f"_{JOB_ID}_{TASK_ID}"
    return name


ray.init()
results = tune.run(
    # PPOTrainer,
    # ProjectedPPOTrainer,
    trainer,
    config=config,
    stop={
        "agent_timesteps_total": 6100e3,
    },
    verbose=1,
    trial_name_creator=name_creator,
    name="FlexArm_RigidPlus",
    local_dir="ray_results",
    checkpoint_at_end=True,
    checkpoint_freq=1000,
)
