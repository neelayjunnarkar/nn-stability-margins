"""
Main file for configuring and training controllers on Savio with array job
"""

import math
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from envs import (
    FlexibleArmEnv,
    InvertedPendulumEnv,
    TimeDelayInvertedPendulumEnv,
    FlexibleArmDiskMarginEnv,
)
from models import (
    RINN,
    RNN,
    DissipativeRINN,
    DissipativeSimplestRINN,
    # DissipativeThetaRINN,
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

disk_margin_types = 2*(2*(2*["3db20deg"] + 2*["6db36deg"] + 2*["12db60deg"]) + [None, None])
assert len(disk_margin_types) == 28
disk_margin_type = disk_margin_types[TASK_ID]

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
#     # "design_model": "rigidplus",  # trs in [1, 2] seem kind of the same... Maybe use 1.2.
#     "design_model": "rigidplus_integrator",
#     "delta_alpha": 1.0,
#     "supplyrate_scale": 0.5,
#     "lagrange_multiplier": 5,
# }
dt = 0.001
env = FlexibleArmDiskMarginEnv
env_config = {
    "dt": dt,
    "seed": SEED,
    "normed": True,
    "rollout_length": int(2 / dt) - 1,
    "disturbance_model": "occasional",
    "disk_margin_type": disk_margin_type,
}

## Model Config by Task

custom_model = None
custom_model_config = None
learning_rate = None
sgd_minibatch_size = 2048
train_batch_size = 20480
trainer = None
log_std_init = np.log(1.0)

learning_rates = 14*[5*1e-5] + 14*[1e-4]
mdeltap_fixeds = 2*(6*[True] + 6*[False] + [None, None])

assert TASK_ID >= 0 and TASK_ID <= 27
assert len(learning_rates) == 28
assert len(mdeltap_fixeds) == 28

learning_rate = learning_rates[TASK_ID]
mdeltap_fixed = mdeltap_fixeds[TASK_ID]

if TASK_ID in [0, 2, 4, 6, 8, 10, 14, 16, 18, 20, 22, 24]:
    custom_model = DissipativeSimplestRINN
    custom_model_config = {
        "state_size": 2,
        "nonlin_size": 16,
        "log_std_init": log_std_init,
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
        "fix_mdeltap": mdeltap_fixed
    }
    trainer = ProjectedPPOTrainer
elif TASK_ID in [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]:
    custom_model = LTIModel
    custom_model_config = {
        "dt": dt,
        "plant": env,
        "plant_config": env_config,
        "learn": True,
        "log_std_init": log_std_init,
        "state_size": 2,
        "trs_mode": "fixed",
        "min_trs": 1.0,
        "backoff_factor": 1.05,
        "lti_controller": "dissipative_thetahat",
        "lti_controller_kwargs": {
            "trs_mode": "fixed",
            "min_trs": 1.0,
            "backoff_factor": 1.05,
        },
        "fix_mdeltap": mdeltap_fixed
    }
    trainer = ProjectedPPOTrainer
elif TASK_ID in [12, 26]:
    custom_model = RINN
    custom_model_config = {
        "state_size": 2,
        "nonlin_size": 16,
        "log_std_init": log_std_init,
        "dt": dt,
        "plant": env,
        "plant_config": env_config,
        "eps": 1e-3,
    }
    trainer = ProjectedPPOTrainer
elif TASK_ID in [13, 27]:
    custom_model = FullyConnectedNetwork
    custom_model_config = {"n_layers": 2, "size": 19}
    trainer = PPOTrainer
else:
    raise  ValueError("Invalid Task ID.")

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
        "agent_timesteps_total": 1e7,
    },
    verbose=1,
    trial_name_creator=name_creator,
    name="FlexArmDiskMargin",
    local_dir="ray_results",
    checkpoint_at_end=True,
    checkpoint_freq=1000,
)
