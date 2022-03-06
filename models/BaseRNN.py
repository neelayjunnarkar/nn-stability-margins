import torch
import torch.nn as nn

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override

from models.utils import build_mlp

class BaseRNN(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        state_size = None,
        hidden_size = None,
        phi_cstor = nn.Tanh,
        A_phi = torch.tensor(0),
        B_phi = torch.tensor(1),
        log_std_init = None,
        nn_baseline_n_layers = 2,
        nn_baseline_size = 64
    ):
        assert 2*action_space.shape[0] == num_outputs, "Num outputs should be 2 * action dimension"
        assert state_size is not None, "state_size is None"
        assert hidden_size is not None, "hidden_size is None"
        assert log_std_init is not None, "log_std_init is None"

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.ac_dim = action_space.shape[0]
        self.ob_dim = obs_space.shape[0]
        self.state_size = state_size
        self.hidden_size = hidden_size

        # Setup activation function and loop transformation variables
        self.phi = phi_cstor()
        self.A_phi = A_phi
        self.B_phi = B_phi
        assert self.A_phi.shape == self.B_phi.shape, "A_phi and B_phi must be the same size"
        self.S_phi = (self.A_phi + self.B_phi)/2
        if self.A_phi.ndim > 0:
            self._scalar_bounds = False
            self.L_phi_inv = torch.inverse((self.B_phi - self.A_phi)/2)
        else:
            self._scalar_bounds = True
            self.L_phi_inv = 2 / (self.B_phi - self.A_phi)

        self.log_stds = nn.Parameter(log_std_init * torch.ones(self.ac_dim))

        self.value = build_mlp(input_size = self.ob_dim, output_size = 1, 
            n_layers = nn_baseline_n_layers, size = nn_baseline_size)
        
        self._last_obs = None

    def phi_t(self, state, obs):
        """Loop transformed phi"""
        assert "Must be overidden"

    @override(ModelV2)
    def get_initial_state(self):
        xi0 = torch.zeros(self.state_size)
        return [xi0]

    @override(ModelV2)
    def value_function(self):
        assert self._last_obs is not None, "must call forward() first"
        value = torch.reshape(self.value(self._last_obs), [-1])
        return value

    @override(RecurrentNetwork)
    def forward_rnn(self, obs, state, seq_lens):
        # xi(k+1) = AK_t  xi(k) + BK1_t z(k) + BK2_t y(k)
        # u(k)    = CK1_t xi(k) + DK1_t z(k) + DK2_t y(k)
        # z(k)    = phi_t(xi(k), v(k))
        assert(len(state) == 1)

        state = state[0]
        batch_size = obs.shape[0]
        time_len = obs.shape[1]
        actions = torch.zeros(batch_size, time_len, self.ac_dim)        

        for k in range(time_len):
            y = obs[:, k]
            z = self.phi_t(state, y)
            u = state @ self.CK1_tT + z @ self.DK1_tT + y @ self.DK2_tT
            state = state @ self.AK_tT + z @ self.BK1_tT + y @ self.BK2_tT
            actions[:, k] = u

        self._last_obs = obs.clone()

        log_stds_rep = self.log_stds.repeat(batch_size, time_len, 1)
        outputs = torch.cat((actions, log_stds_rep), dim=2)
        return outputs, [state]