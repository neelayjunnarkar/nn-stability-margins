"""
Main file for configuring and training controllers.
"""

import argparse
import math
import multiprocessing
import os

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer

from envs import FlexibleArmDiskMarginEnv, FlexibleArmEnv, InvertedPendulumEnv
from models import (
    RINN,
    DissipativeSimplestRINN,
    FullyConnectedNetwork,
    ImplicitModel,
    LTIModel,
)
from trainers import ProjectedPPOTrainer
from utils import ExportWeightsCallback

# =====================
# ENVIRONMENT CONFIGS
# =====================


def get_inverted_pendulum_env_crown(seed, saturate_inputs, reward_type=None):
    dt = 0.01
    env = InvertedPendulumEnv
    env_config = {
        "observation": "full",
        "normed": True,
        "dt": dt,
        "supply_rate": "l2_gain",
        "disturbance_model": "none",
        "seed": seed,
        "saturate_inputs": saturate_inputs,
    }
    if reward_type is not None:
        env_config["reward_type"] = reward_type
    return dt, env, env_config


def get_inverted_pendulum_env(
    seed,
    saturate_inputs,
    reward_type=None,
    rwd_action_scale=None,
    supply_rate="stability",
):
    dt = 0.01
    env = InvertedPendulumEnv
    env_config = {
        "observation": "partial",
        "normed": True,
        "dt": dt,
        "supply_rate": supply_rate,  # "stability" or "l2_gain",
        "disturbance_model": "occasional",
        "seed": seed,
        "saturate_inputs": saturate_inputs,
    }
    if reward_type is not None:
        env_config["reward_type"] = reward_type
    if rwd_action_scale is not None:
        env_config["rwd_action_scale"] = rwd_action_scale
    return dt, env, env_config


def get_flexible_arm_env_full(seed, saturate_inputs, l2_gain=None, reward_type=None):
    dt = 0.001  # or 0.0001
    env = FlexibleArmEnv
    env_config = {
        "observation": "full",
        "normed": True,
        "dt": dt,
        "rollout_length": int(2.5 / dt) - 1,  # 10000,
        "supply_rate": "l2_gain",
        "disturbance_model": "none",
        "disturbance_design_model": "occasional",
        "design_model": "rigid",
        "seed": seed,
        "saturate_inputs": saturate_inputs,
    }
    if l2_gain is not None:
        env_config["gamma"] = l2_gain
    if reward_type is not None:
        env_config["reward_type"] = reward_type
    return dt, env, env_config


def get_flexible_arm_env_partial(seed, saturate_inputs, l2_gain=None, reward_type=None):
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
        "design_model": "rigidplus_integrator",
        "delta_alpha": 1.0,
        "design_integrator_type": "utox2",
        "supplyrate_scale": 0.5,
        "lagrange_multiplier": 5,
        # "design_integrator_type": "utoy",
        # "supplyrate_scale": 1,
        # "lagrange_multiplier": 1000,
        "saturate_inputs": saturate_inputs,
        "seed": seed,
    }
    if l2_gain is not None:
        env_config["gamma"] = l2_gain
    if reward_type is not None:
        env_config["reward_type"] = reward_type
    return dt, env, env_config


def get_flexible_arm_disk_margin_env(seed, saturate_inputs, reward_type=None):
    dt = 0.001
    env = FlexibleArmDiskMarginEnv
    env_config = {
        "dt": dt,
        "seed": seed,
        "normed": True,
        "rollout_length": int(2 / dt) - 1,
        "disturbance_model": "occasional",
        "disk_margin_type": "12dB60deg",  # "6dB36deg",
        # "skew": 0,
        # "alpha": 0,
    }
    if reward_type is not None:
        env_config["reward_type"] = reward_type
    return dt, env, env_config


# =====================
# MODEL CONFIGS
# =====================


def get_fully_connected_model(dt, env, env_config):
    return {
        "custom_model": FullyConnectedNetwork,
        "custom_model_config": {"n_layers": 2, "size": 19},
    }


def get_implicit_model(dt, env, env_config):
    return {
        "custom_model": ImplicitModel,
        "custom_model_config": {"state_size": 16},
    }


def get_rinn_model(dt, env, env_config):
    return {
        "custom_model": RINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": 16,
            "dt": dt,
            "log_std_init": np.log(1.0),
            "plant": env,
            "plant_config": env_config,
            "eps": 1e-3,
        },
    }


def get_dissipative_simplest_rinn_model(dt, env, env_config, trs, backoff, nphi):
    return {
        "custom_model": DissipativeSimplestRINN,
        "custom_model_config": {
            "state_size": 2,
            "nonlin_size": nphi,  # 16,
            "log_std_init": np.log(1.0),
            "dt": dt,
            "plant": env,
            "plant_config": env_config,
            "eps": 1e-3,
            "mode": "thetahat",
            "trs_mode": "fixed",
            "min_trs": trs,  # 1,
            "backoff_factor": backoff,  # 1.05,
            "lti_initializer": "dissipative_thetahat",
            "lti_initializer_kwargs": {
                "trs_mode": "fixed",
                "min_trs": trs,  # 1,
                "backoff_factor": backoff,  # 1.05,
            },
            "fix_mdeltap": False,
            "Dkvw_structure": "full",  # "full" or "strict_upper_triang"
        },
    }


def get_lti_model(dt, env, env_config, trs, backoff):
    return {
        "custom_model": LTIModel,
        "custom_model_config": {
            "dt": dt,
            "plant": env,
            "plant_config": env_config,
            "learn": True,
            "log_std_init": np.log(1.0),
            "state_size": 2,
            "trs_mode": "fixed",
            "min_trs": trs,  # 1.5, # 1.44,
            "lti_controller": "dissipative_thetahat",
            "lti_controller_kwargs": {
                "trs_mode": "fixed",
                "min_trs": trs,  # 1.5 # 1.44
                "backoff_factor": backoff,  # 1.05
            },
            "fix_mdeltap": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train controller with selectable model, environment, seed, and learning rate."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["fcnn", "rinn", "drinn", "lti"],
        help="Controller model to use",
    )
    parser.add_argument(
        "--drinn_nphi", type=int, default=16, help="Number of D-RINN neurons"
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=[
            "flexible_rod_partial",
            "flexible_rod_full",
            "flexible_rod_disk_margin",
            "inverted_pendulum",
            "inverted_pendulum_crown",
        ],
        help="Environment to use",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument(
        "--trs",
        type=float,
        default=1.0,
        help="Factor for numerical conditioning in dissipative models",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.05,
        help="Suboptimal projection allowance for numerical conditioning in dissipative models",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="scratch",
        help="Name of the experiment for logging",
    )
    parser.add_argument(
        "--no_input_saturation",
        action="store_true",
        help="If set, environment will clip actions to action_space bounds (default: False)",
    )
    parser.add_argument(
        "--l2_gain",
        type=float,
        default=None,
        help="L2 gain gamma for flexible arm environments",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default=None,
        help="Reward type for environments (default or quadratic)",
    )
    parser.add_argument(
        "--rwd_action_scale",
        type=float,
        default=None,
        help="Reward scale for inverted pendulum environments",
    )
    parser.add_argument(
        "--timesteps_total",
        type=int,
        default=1e7,
        help="Total number of environment timesteps for training",
    )
    parser.add_argument(
        "--projection_period",
        type=int,
        default=1,
        help="Project every this number of steps (1 means every step)",
    )
    args = parser.parse_args()

    args.saturate_inputs = not args.no_input_saturation

    use_savio = False
    if use_savio:
        print("\n\n\nUsing Savio\n===========\n\n\n")
        N_CPUS = int(os.getenv("SLURM_CPUS_ON_NODE"))
        JOB_ID = os.getenv("SLURM_JOB_ID")
    else:
        N_CPUS = multiprocessing.cpu_count()
        JOB_ID = None
    n_tasks = 1
    n_workers_per_task = int(math.floor(N_CPUS / n_tasks)) - 2

    # Select environment based on command-line argument
    if args.env == "flexible_rod_partial":
        dt, env, env_config = get_flexible_arm_env_partial(
            args.seed, args.saturate_inputs, args.l2_gain, args.reward_type
        )
    elif args.env == "flexible_rod_full":
        dt, env, env_config = get_flexible_arm_env_full(
            args.seed, args.saturate_inputs, args.l2_gain, args.reward_type
        )
    elif args.env == "flexible_rod_disk_margin":
        dt, env, env_config = get_flexible_arm_disk_margin_env(
            args.seed, args.saturate_inputs, args.reward_type
        )
    elif args.env == "inverted_pendulum":
        dt, env, env_config = get_inverted_pendulum_env(
            args.seed, args.saturate_inputs, args.reward_type, args.rwd_action_scale
        )
    elif args.env == "inverted_pendulum_crown":
        dt, env, env_config = get_inverted_pendulum_env_crown(
            args.seed, args.saturate_inputs, args.reward_type
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

    # Select model based on command-line argument
    if args.model == "fcnn":
        model_config = get_fully_connected_model(dt, env, env_config)
    elif args.model == "rinn":
        model_config = get_rinn_model(dt, env, env_config)
    elif args.model == "drinn":
        model_config = get_dissipative_simplest_rinn_model(
            dt, env, env_config, args.trs, args.backoff, args.drinn_nphi
        )
    elif args.model == "lti":
        model_config = get_lti_model(dt, env, env_config, args.trs, args.backoff)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    config = {
        "env": env,
        "env_config": env_config,
        "model": model_config,
        "projection_period": args.projection_period,
        "sgd_minibatch_size": 2048,
        "train_batch_size": 20480,
        "lr": args.lr,
        "num_envs_per_worker": 10,
        "seed": args.seed,
        "num_workers": n_workers_per_task,
        "framework": "torch",
        "num_gpus": 0,  # 1,
        "evaluation_num_workers": 1,
        "evaluation_config": {"render_env": False, "explore": False},
        "evaluation_interval": 1,
        "evaluation_parallel_to_training": True,
        "clip_actions": False,
        "normalize_actions": args.saturate_inputs,
        "callbacks": ExportWeightsCallback,
    }

    print("==================================")
    print("Number of workers per task: ", n_workers_per_task)
    print("")
    print("Config: ")
    print(config)
    print("")

    test_env = env(env_config)
    print(
        f"Max reward per: step: {test_env.max_reward}, rollout: {test_env.max_reward * (test_env.time_max + 1)}"
    )
    print("==================================")

    def name_creator(trial):
        config = trial.config
        name = f"{config['env'].__name__}"
        name += f"_{config['model']['custom_model'].__name__}"
        if use_savio:
            name += f"_{JOB_ID}"
        name += f"_seed{args.seed}"
        return name

    # Select trainer based on model
    if model_config["custom_model"].__name__ == "FullyConnectedNetwork":
        trainer_cls = PPOTrainer
    else:
        trainer_cls = ProjectedPPOTrainer

    ray.init()
    tune.run(
        trainer_cls,
        config=config,
        stop={
            "agent_timesteps_total": args.timesteps_total,
        },
        verbose=1,
        trial_name_creator=name_creator,
        name=args.experiment_name,
        local_dir="ray_results",
        checkpoint_at_end=True,
        checkpoint_freq=100,
    )


if __name__ == "__main__":
    main()


# Configure the algorithm.
# config = {
#     "env": env,
#     "env_config": env_config,
#     "model": {
#         # "custom_model": FullyConnectedNetwork,
#         # "custom_model_config": {
#         #     "n_layers": 2,
#         #     "size": 19
#         # }
#         # "custom_model": RINN,
#         # "custom_model_config": {
#         #     "state_size": 2,
#         #     "nonlin_size": 16,
#         #     "log_std_init": np.log(1.0),
#         #     "dt": dt,
#         #     "plant": env,
#         #     "plant_config": env_config,
#         #     "eps": 1e-3,
#         # },
#         "custom_model": DissipativeSimplestRINN,
#         "custom_model_config": {
#             "state_size": 2,
#             "nonlin_size": 16,
#             "log_std_init": np.log(1.0),
#             "dt": dt,
#             "plant": env,
#             "plant_config": env_config,
#             "eps": 1e-3,
#             "mode": "thetahat",
#             "trs_mode": "fixed",
#             "min_trs": 1,
#             "backoff_factor": 1.1,
#             "lti_initializer": "dissipative_thetahat",
#             "lti_initializer_kwargs": {
#                 "trs_mode": "fixed",
#                 "min_trs": 1,
#                 "backoff_factor": 1.1,
#             },
#             "fix_mdeltap": False,
#             "Dkvw_structure": "full",  # "full" or "strict_upper_triang"
#         },
#         # "custom_model": LTIModel,
#         # "custom_model_config": {
#         #     "dt": dt,
#         #     "plant": env,
#         #     "plant_config": env_config,
#         #     "learn": True,
#         #     "log_std_init": np.log(1.0),
#         #     "state_size": 2,
#         #     "trs_mode": "fixed",
#         #     "min_trs": 1,  # 1.5, # 1.44,
#         #     "backoff_factor": 1.1,
#         #     "lti_controller": "dissipative_thetahat",
#         #     "lti_controller_kwargs": {
#         #         "trs_mode": "fixed",
#         #         "min_trs": 1,  # 1.5 # 1.44
#         #         "backoff_factor": 1.1,
#         #     },
#         #     "fix_mdeltap": False,
#         # },
#     },
#     ## Custom Policy Parameters
#     # How often to do projection. n -> every n'th gradient step. E.g., 1 -> every gradient step.
#     "projection_period": 100,
#     ## Testing changes to training parameters
#     "sgd_minibatch_size": 2048,
#     "train_batch_size": 20480,
#     "lr": 1e-4,
#     "num_envs_per_worker": 10,
#     ## End test
#     "seed": seed,
#     "num_workers": n_workers_per_task,
#     "framework": "torch",
#     "num_gpus": 0,  # 1,
#     "evaluation_num_workers": 1,
#     "evaluation_config": {"render_env": False, "explore": False},
#     "evaluation_interval": 1,
#     "evaluation_parallel_to_training": True,
#     "callbacks": ExportWeightsCallback,
# }

# print("==================================")
# print("Number of workers per task: ", n_workers_per_task)
# print("")

# print("Config: ")
# print(config)
# print("")

# test_env = env(env_config)
# print(
#     f"Max reward per: step: {test_env.max_reward}, rollout: {test_env.max_reward * (test_env.time_max + 1)}"
# )
# print("==================================")


# def name_creator(trial):
#     config = trial.config
#     name = f"{config['env'].__name__}"
#     name += f"_{config['model']['custom_model'].__name__}"
#     if use_savio:
#         name += f"_{JOB_ID}"
#     return name


# ray.init()
# results = tune.run(
#     # PPOTrainer,
#     ProjectedPPOTrainer,
#     config=config,
#     stop={
#         "agent_timesteps_total": 1e7,
#     },
#     verbose=1,
#     trial_name_creator=name_creator,
#     name="scratch",
#     local_dir="ray_results",
#     checkpoint_at_end=True,
#     checkpoint_freq=1,
# )
