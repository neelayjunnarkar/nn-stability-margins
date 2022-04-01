from ray.rllib.utils.annotations import override
import torch.nn as nn
from models.utils import uniform
from models.BaseRNN import BaseRNN

class RNNModel(BaseRNN):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_args
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name, **custom_args)

        #_T for transpose, _t for tilde
        self.AK_tT  = nn.Parameter(uniform(self.state_size, self.state_size))
        self.BK1_tT = nn.Parameter(uniform(self.hidden_size, self.state_size))
        self.BK2_tT = nn.Parameter(uniform(self.ob_dim, self.state_size))

        self.CK1_tT = nn.Parameter(uniform(self.state_size, self.ac_dim))
        self.DK1_tT = nn.Parameter(uniform(self.hidden_size, self.ac_dim))
        self.DK2_tT = nn.Parameter(uniform(self.ob_dim, self.ac_dim))

        self.CK2_tT = nn.Parameter(uniform(self.state_size, self.hidden_size))
        self.DK4_tT = nn.Parameter(uniform(self.ob_dim, self.hidden_size))

    @override(BaseRNN)
    def phi_t(self, state, obs):
        """Loop transformed phi"""
        # v(k) = CK2_t xi(k) + DK4_t y(k)
        # z(k) = phi_t(v(k))
        v = state @ self.CK2_tT + obs @ self.DK4_tT
        if self._scalar_bounds:
            z = self.L_phi_inv * (self.phi(v) - self.S_phi * v)
        else:
            z = self.L_phi_inv @ (self.phi(v) - self.S_phi @ v)
        return z
