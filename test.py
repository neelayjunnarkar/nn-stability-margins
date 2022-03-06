# Import the RL algorithm (Trainer) we would like to use.
from ast import Pow
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

env = PowergridEnv
env_config = {
    "observation": "partial",
    "normed": True,
    "factor": 1,
}

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    # "env": "Taxi-v3",
    "env": env,
    "env_config": env_config,
    "model": {
        "custom_model": ProjRNNModel,
        "custom_model_config": {
            "state_size": 20, #tune.choice([2, 16, 32]), #2,
            "hidden_size": 20, #tune.choice([2, 16, 32]), # 16
            "phi_cstor": nn.Tanh,
            "A_phi": torch.tensor(0),
            "B_phi": torch.tensor(1),
            "log_std_init": np.log(0.2), # tune.uniform(np.log(0.01), np.log(2)), # np.log(0.2),
            "nn_baseline_n_layers": 2,
            "nn_baseline_size": 64,
            # Projecting controller parameters
            "lmi_eps": 1e-3,
            "exp_stability_rate": 0.98,
            "plant_cstor": env,
            "plant_config": env_config,
            # REN parameters
            "solver": broyden,
            "f_thresh": 30,
            "b_thresh": 30
        }
    },
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    # "num_envs_per_worker": 8,
    "num_workers": 2,
    "framework": "torch",
    "num_gpus": 0,
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
    "evaluation_interval": 1,
    "evaluation_parallel_to_training": True
}


ray.init()

optuna_search = OptunaSearch(metric="episode_reward_mean", mode="max")

asha_scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric='episode_reward_mean',
    mode='max',
    max_t=1000,
    grace_period=200,
    reduction_factor=3,
    brackets=1
)

def name_creator(trial):
    config = trial.config
    name = f"{config['model']['custom_model'].__name__}_{config['env'].__name__}"
    return name

results = tune.run(
    ProjectedPGTrainer,
    config = config,
    stop = {
        # "episode_reward_mean": 990,
        "training_iteration": 1000
    },
    # num_samples = 3,
    # search_alg = optuna_search,
    # scheduler = asha_scheduler,
    # fail_fast = True,
    # verbose = False
    trial_name_creator = name_creator
)

metric = "episode_reward_mean"
best_trial = results.get_best_trial(metric=metric, mode="max", scope="all")
value_best_metric = best_trial.metric_analysis[metric]["max"]
print(
    "Best trial's reward mean(over all "
    "iterations): {}".format(value_best_metric)
)

print('best config: ', results.get_best_config(metric=metric, mode="max"))

# ray.init()
# trainer = ProjectedPGTrainer(env=config["model"]["custom_model_config"]["plant_cstor"], config=config)
# trainer = ProjectedPPOTrainer(env=config["model"]["custom_model_config"]["plant_cstor"], config=config)

# for _ in range(1000):
#     print(f'\n======= Iter {_} =======')
#     trainer.train()
    

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# trainer.evaluate()
