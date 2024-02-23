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
SEED = int(os.getenv("SEED"))

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
    "disturbance_model": "occasional",
    "disturbance_design_model": "occasional",
    "design_model": "rigidplus",  # trs in [1, 2] seem kind of the same... Maybe use 1.2.
    "delta_alpha": 1.0,
    "supplyrate_scale": 1.6,
}

## Model Config by Task

custom_model = None
custom_model_config = None
learning_rate = None
sgd_minibatch_size = None
train_batch_size = None
trainer = None
log_std_init = np.log(1.0)

learning_rates = 9 * [5e-5] + 9 * [1e-4] + 9 * [2e-4]
assert len(learning_rates) == 27
sgd_minibatch_sizes = 9 * [512, 1024, 2048]
assert len(sgd_minibatch_sizes) == 27
train_batch_sizes = 3 * (3 * [5120] + 3 * [10240] + 3 * [20480])
assert len(train_batch_sizes) == 27

assert TASK_ID >= 0 and TASK_ID <= 26
learning_rate = learning_rates[TASK_ID]
sgd_minibatch_size = sgd_minibatch_sizes[TASK_ID]
train_batch_size = train_batch_sizes[TASK_ID]

assert log_std_init is not None

custom_model = RINN
custom_model_config = {
    "state_size": 4,
    "nonlin_size": 16,
    "log_std_init": log_std_init,
    "dt": dt,
    "plant": env,
    "plant_config": env_config,
    "eps": 1e-3,
}
trainer = ProjectedPPOTrainer

assert custom_model is not None
assert custom_model_config is not None
assert learning_rate is not None
assert sgd_minibatch_size is not None
assert train_batch_size is not None
assert trainer is not None


# Configure the algorithm.
config = {
    "env": env,
    "env_config": env_config,
    "model": {
        "custom_model": custom_model,
        "custom_model_config": custom_model_config,
    },
    "sgd_minibatch_size": sgd_minibatch_size,
    "train_batch_size": train_batch_size,
    "lr": learning_rate,
    "seed": SEED,
    "num_envs_per_worker": 10,
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
    trainer,
    config=config,
    stop={
        "agent_timesteps_total": 6100e3,
    },
    verbose=1,
    trial_name_creator=name_creator,
    name="FlexArm_Occas_Occas_RigidPlus",
    local_dir="ray_results",
    checkpoint_at_end=True,
    checkpoint_freq=1000,
)
