# Import the RL algorithm (Trainer) we would like to use.
import ray
from envs.linearized_inverted_pendulum import LinearizedInvertedPendulumEnv
from models.RNN import RNNModel
from models.ProjRNN import ProjRNNModel
import numpy as np

from trainers import ProjectedPGTrainer, ProjectedPPOTrainer

env_config = {
    "observation": "partial",
    "normed": True,
    "factor": 1,
}

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    # "env": "Taxi-v3",
    "env_config": env_config,
    "model": {
        "custom_model": ProjRNNModel,
        "custom_model_config": {
            "state_size": 2,
            "hidden_size": 16,
            "log_std_init": np.log(0.2),
            "nn_baseline_n_layers": 2,
            "nn_baseline_size": 64,
            "lmi_eps": 1e-3,
            "exp_stability_rate": 0.98,
            "plant_cstor": LinearizedInvertedPendulumEnv,
            "plant_config": env_config
        }
    },
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    # "num_envs_per_worker": 8,
    "num_workers": 3,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    # "model": {
    #     "fcnet_hiddens": [64, 64],
    #     "fcnet_activation": "relu",
    # },
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
# Create our RLlib Trainer
trainer = ProjectedPGTrainer(env=config["model"]["custom_model_config"]["plant_cstor"], config=config)
# trainer = ProjectedPPOTrainer(env=config["model"]["custom_model_config"]["plant_cstor"], config=config)
# trainer = ProjectedPGTrainer(env=LinearizedInvertedPendulumEnv, config = config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.

for _ in range(1000):
    print(f'\n======= Iter {_} =======')
    trainer.train()
    

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# trainer.evaluate()
