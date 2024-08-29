"""
RLLib trainers modified to include a projection step after updating model parameters.
"""

from ray.rllib.agents import ppo, pg
from ray.rllib.utils.annotations import override


class ProjectedPGPolicy(pg.pg_torch_policy.PGTorchPolicy):
    @override(pg.pg_torch_policy.PGTorchPolicy)
    def apply_gradients(self, gradients):
        super().apply_gradients(gradients)
        self.model.project()


# class ProjectedPPOPolicy(ppo.ppo_torch_policy.PPOTorchPolicy):
#     @override(ppo.ppo_torch_policy.PPOTorchPolicy)
#     def apply_gradients(self, gradients):
#         super().apply_gradients(gradients)
#         self.model.project()


class ProjectedPPOPolicy(ppo.ppo_torch_policy.PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.projection_period = config["projection_period"]
        self.time_since_last_projection = 0

    @override(ppo.ppo_torch_policy.PPOTorchPolicy)
    def apply_gradients(self, gradients):
        super().apply_gradients(gradients)
        self.time_since_last_projection += 1
        if self.time_since_last_projection >= self.projection_period:
            self.model.project()
            self.time_since_last_projection = 0
            print("[ProjectedPPOPolicy]: DID A PROJECTION")


class ProjectedPGTrainer(pg.PGTrainer):
    @override(pg.PGTrainer)
    def get_default_policy_class(self, config):
        return ProjectedPGPolicy


class ProjectedPPOTrainer(ppo.PPOTrainer):
    @override(ppo.PPOTrainer)
    def get_default_policy_class(self, config):
        return ProjectedPPOPolicy

    @classmethod
    @override(ppo.PPOTrainer)
    def get_default_config(cls):
        config = ppo.PPOTrainer.get_default_config()
        config["projection_period"] = 1  # Always project
        return config
