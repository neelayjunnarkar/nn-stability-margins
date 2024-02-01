"""
This folder contains plant models.
"""

# from envs.cartpole import CartpoleEnv
# from envs.inverted_pendulum import InvertedPendulumEnv
# from envs.learned_inverted_pendulum import LearnedInvertedPendulumEnv
# from envs.linearized_inverted_pendulum import LinearizedInvertedPendulumEnv
# from envs.other_inverted_pendulum import OtherInvertedPendulumEnv
# from envs.pendubot import PendubotEnv
# from envs.powergrid import PowergridEnv
# from envs.vehicle import VehicleLateralEnv

from envs.inverted_pendulum import InvertedPendulumEnv
from envs.time_delay_inverted_pendulum import TimeDelayInvertedPendulumEnv
from envs.flex_arm import FlexibleArmEnv

env_map = {
    "<class 'envs.inverted_pendulum.InvertedPendulumEnv'>": InvertedPendulumEnv,
    "<class 'envs.flex_arm.FlexibleArmEnv'>": FlexibleArmEnv,
}