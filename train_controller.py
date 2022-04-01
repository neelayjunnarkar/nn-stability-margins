"""
Main file for configuring and training controllers.
"""

import os
import math
import ray
from ray import tune
import numpy as np
import torch

from envs import OtherInvertedPendulumEnv, InvertedPendulumEnv, LearnedInvertedPendulumEnv
from models import ProjRENModel, ProjRNNModel, ProjRNNOldModel
from activations import LeakyReLU, Tanh
from deq_lib.solvers import broyden, anderson # Fixed-point solvers
from trainers import ProjectedPGTrainer, ProjectedPPOTrainer


N_CPUS = 8 # laptop
# N_CPUS  = int(os.getenv('SLURM_CPUS_ON_NODE'))
n_tasks = 1
n_workers_per_task = int(math.floor(N_CPUS/n_tasks))-1-1

env = InvertedPendulumEnv
env_config = {  
    "observation": "partial",
    "normed": True,
    "factor": 1,
    # "model_params": torch.load('learned_models/inv_pend_model_B2_nonlin2.pth')
}

# Configure the algorithm.
config = {
    "env": env,
    "env_config": env_config,
    "model": {
        "custom_model": ProjRENModel, # ProjRENModel, ProjRNNModel, ProjRNNOldModel
        "custom_model_config": {
            "state_size": 2,
            "hidden_size": 4,
            "phi_cstor": Tanh, # Tanh, LeakyReLU
            "log_std_init": np.log(0.2),
            "nn_baseline_n_layers": 2,
            "nn_baseline_size": 64,
            # Projecting controller parameters
            "lmi_eps": 1e-5,
            "exp_stability_rate": 0.9,
            "plant_cstor": env,
            "plant_config": env_config,
            # REN parameters
            "solver": broyden,
            "f_thresh": 30,
            "b_thresh": 30
        }
    },
    "num_workers": n_workers_per_task,
    "framework": "torch",
    "num_gpus": 0,
    "evaluation_num_workers": 1,
    "evaluation_config": {
        "render_env": False,
        "explore": False
    },
    "evaluation_interval": 1,
    "evaluation_parallel_to_training": True
}

print('==================================')
print('Number of workers per task: ', n_workers_per_task)
print('')

print('Config: ')
print(config)
print('')

test_env = env(env_config)
print(f'Max reward per: step: {test_env.max_reward}, rollout: {test_env.max_reward*(test_env.time_max+1)}')
print('==================================')

def name_creator(trial):
    config = trial.config
    model_cfg = config['model']['custom_model_config']
    name =  f"{config['model']['custom_model'].__name__}"
    name += f"_{config['env'].__name__}"
    name += f"_phi{model_cfg['phi_cstor'].__name__}"
    name += f"_state{model_cfg['state_size']}"
    name += f"_hidden{model_cfg['hidden_size']}"
    name += f"_rho{model_cfg['exp_stability_rate']}"
    return name

ray.init()
results = tune.run(
    ProjectedPPOTrainer,
    config = config,
    stop = {
        'agent_timesteps_total': 1e3,
    },
    verbose = 1,
    trial_name_creator = name_creator,
    name = 'scratch',
    local_dir = '../ray_results',
    checkpoint_at_end = True,
    checkpoint_freq = 200,
)
