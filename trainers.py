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

class ProjectedPPOPolicy(ppo.ppo_torch_policy.PPOTorchPolicy):
    @override(ppo.ppo_torch_policy.PPOTorchPolicy)
    def apply_gradients(self, gradients):
        super().apply_gradients(gradients)
        self.model.project()

class ProjectedPGTrainer(pg.PGTrainer):
    @override(pg.PGTrainer)
    def get_default_policy_class(self, config):
        return ProjectedPGPolicy

class ProjectedPPOTrainer(pg.PGTrainer):
    @override(ppo.PPOTrainer)
    def get_default_policy_class(self, config):
        return ProjectedPPOPolicy