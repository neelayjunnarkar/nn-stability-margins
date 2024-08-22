"""
This folder contains plant models.
"""

from envs.inverted_pendulum import InvertedPendulumEnv
from envs.time_delay_inverted_pendulum import TimeDelayInvertedPendulumEnv
from envs.flex_arm import FlexibleArmEnv
from envs.disk_margin_example import DiskMarginExampleEnv
from envs.flex_arm_disk_margin import FlexibleArmDiskMarginEnv

env_map = {
    "<class 'envs.inverted_pendulum.InvertedPendulumEnv'>": InvertedPendulumEnv,
    "<class 'envs.flex_arm.FlexibleArmEnv'>": FlexibleArmEnv,
    "<class 'envs.time_delay_inverted_pendulum.TimeDelayInvertedPendulumEnv'>": TimeDelayInvertedPendulumEnv,
}
