
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
from models.utils import build_mlp, uniform
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

        #_T for transpose
        #_t for tilde
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
        

# class TorchRNNModel(RecurrentNetwork, nn.Module):
#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#         fc_size=64,
#         lstm_state_size=256,
#     ):
#         nn.Module.__init__(self)
#         super().__init__(obs_space, action_space, num_outputs, model_config, name)

#         self.obs_size = get_preprocessor(obs_space)(obs_space).size
#         self.fc_size = fc_size
#         self.lstm_state_size = lstm_state_size

#         # Build the Module from fc + LSTM + 2xfc (action + value outs).
#         self.fc1 = nn.Linear(self.obs_size, self.fc_size)
#         self.lstm = nn.LSTM(self.fc_size, self.lstm_state_size, batch_first=True)
#         self.action_branch = nn.Linear(self.lstm_state_size, num_outputs)
#         self.value_branch = nn.Linear(self.lstm_state_size, 1)
#         # Holds the current "base" output (before logits layer).
#         self._features = None

#     @override(ModelV2)
#     def get_initial_state(self):
#         # TODO: (sven): Get rid of `get_initial_state` once Trajectory  
#         #  View API is supported across all of RLlib.
#         # Place hidden states on same device as model.
#         h = [
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#             self.fc1.weight.new(1, self.lstm_state_size).zero_().squeeze(0),
#         ]
#         return h

#     @override(ModelV2)
#     def value_function(self):
#         assert self._features is not None, "must call forward() first"
#         value = torch.reshape(self.value_branch(self._features), [-1])
#         print('VALuEEEEE', value.shape)
#         return value

#     @override(RecurrentNetwork)
#     def forward_rnn(self, inputs, state, seq_lens):
#         """Feeds `inputs` (B x T x ..) through the Gru Unit.
#         Returns the resulting outputs as a sequence (B x T x ...).
#         Values are stored in self._cur_value in simple (B) shape (where B
#         contains both the B and T dims!).
#         Returns:
#             NN Outputs (B x T x ...) as sequence.
#             The state batches as a List of two items (c- and h-states).
#         """
#         x = nn.functional.relu(self.fc1(inputs))
#         self._features, [h, c] = self.lstm(
#             x, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
#         )
#         action_out = self.action_branch(self._features)
#         return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

class RNNModelOld(RecurrentNetwork, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        state_size = None,
        hidden_size = None,
        log_std_init = None,
        nn_baseline_n_layers = 2,
        nn_baseline_size = 64,
    ):
        assert 2*action_space.shape[0] == num_outputs, "Num outputs should be 2 * action dimension"

        try:
            state_size
            hidden_size
            log_std_init
        except NameError:
            assert "Input parameter was None"

        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        # config = model_config['custom_model_config']

        # plant = config['plant']()
        self.ac_dim = action_space.shape[0]
        self.ob_dim = obs_space.shape[0]
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.log_std_init = log_std_init

        #_T for transpose
        #_t for tilde
        self.AK_tT  = nn.Parameter(uniform(self.state_size, self.state_size))
        self.BK1_tT = nn.Parameter(uniform(self.hidden_size, self.state_size))
        self.BK2_tT = nn.Parameter(uniform(self.ob_dim, self.state_size))

        self.CK1_tT = nn.Parameter(uniform(self.state_size, self.ac_dim))
        self.DK1_tT = nn.Parameter(uniform(self.hidden_size, self.ac_dim))
        self.DK2_tT = nn.Parameter(uniform(self.ob_dim, self.ac_dim))

        self.CK2_tT = nn.Parameter(uniform(self.state_size, self.hidden_size))
        self.DK4_tT = nn.Parameter(uniform(self.ob_dim, self.hidden_size))

        self.phi = nn.Tanh()
        self.A_phi = torch.tensor(0)
        self.B_phi = torch.tensor(1)
        self.S_phi = (self.A_phi + self.B_phi)/2
        if self.A_phi.ndim > 0: # Assume B_phi has same shape as A_phi
            self._scalar_bounds = False
            self.L_phi_inv = torch.inverse((self.B_phi - self.A_phi)/2)
        else:
            self._scalar_bounds = True
            self.L_phi_inv = 2 / (self.B_phi - self.A_phi)

        self.log_stds = nn.Parameter(log_std_init * torch.ones(self.ac_dim))

        self.value = build_mlp(input_size = self.ob_dim, output_size = 1, 
            n_layers = nn_baseline_n_layers, size = nn_baseline_size)
        
        self._last_obs = None

    @override(ModelV2)
    def get_initial_state(self):
        xi0 = torch.zeros(self.state_size)
        return [xi0]

    @override(ModelV2)
    def value_function(self):
        assert self._last_obs is not None, "must call forward() first"
        value = torch.reshape(self.value(self._last_obs), [-1])
        return value

    def phi_t(self, v):
        if self._scalar_bounds:
            z = self.L_phi_inv * (self.phi(v) - self.S_phi * v)
        else:
            z = self.L_phi_inv @ (self.phi(v) - self.S_phi @ v)
        return z

    @override(RecurrentNetwork)
    def forward_rnn(self, obs, state, seq_lens):
        # xi(k+1) = AK_t  xi(k) + BK1_t z(k) + BK2_t y(k)
        # u(k)    = CK1_t xi(k) + DK1_t z(k) + DK2_t y(k)
        # v(k)    = CK2_t xi(k) + 0          + DK4_t y(k)
        # z(k)    = phi_t(v(k))
        assert(len(state) == 1)

        state = state[0]
        batch_size = obs.shape[0]
        time_len = obs.shape[1]
        actions = torch.zeros(batch_size, time_len, self.ac_dim)        

        for k in range(time_len):
            y = obs[:, k]
            v = state @ self.CK2_tT + y @ self.DK4_tT
            z = self.phi_t(v)
            u = state @ self.CK1_tT + z @ self.DK1_tT + y @ self.DK2_tT
            state = state @ self.AK_tT + z @ self.BK1_tT + y @ self.BK2_tT
            actions[:, k] = u

        self._last_obs = obs.clone()

        log_stds_rep = self.log_stds.repeat(batch_size, time_len, 1)
        outputs = torch.cat((actions, log_stds_rep), dim=2)
        return outputs, [state]
        

# import itertools
# from torch import nn
# from torch import optim

# import numpy as np
# import torch
# from torch import distributions
# from torch.nn.utils.rnn import pad_sequence

# from deeprl.infrastructure import pytorch_util as ptu
# from deeprl.policies.base_policy import BasePolicy
# from deeprl.policies.ren_projection import ren_project, ren_project_nonlin, LinProjector
# from deeprl.policies.rnn_projection import rnn_project

# import math

# class Transform(nn.Module):
#     def __init__(self, CK2_tT, DK4_tT, phi, A_phi, B_phi, **kwargs):
#         super().__init__(**kwargs)
#         self.CK2_tT = CK2_tT
#         self.DK4_tT = DK4_tT
#         self.phi = phi
#         self.S_phi = (A_phi + B_phi)/2
#         self.scalar_bounds = True
#         if A_phi.ndim > 0: # Assume B_phi has same shape as A_phi
#             self.scalar_bounds = False
#             self.L_phi_inv = torch.inverse((B_phi - A_phi)/2)
#         else:
#             self.L_phi_inv = 2 / (B_phi - A_phi)
#     def forward(self, xi, y):
#         v = xi @ self.CK2_tT + y @ self.DK4_tT
#         if self.scalar_bounds:
#             z = self.L_phi_inv * (self.phi(v) - self.S_phi * v)
#         else:
#             z = self.L_phi_inv @ (self.phi(v) - self.S_phi @ v)
#         return z

# class SingleStepBase(nn.Module):
#     def __init__(self, ac_dim, ob_dim, state_size, hidden_size, \
#         lmi_epsilon, exponential_stability_rate, **kwargs):
#         super().__init__(**kwargs)

#         self.ac_dim = ac_dim
#         self.ob_dim = ob_dim
#         self.state_size = state_size
#         self.hidden_size = hidden_size

#         self.lmi_epsilon = lmi_epsilon
#         self.exponential_stability_rate = exponential_stability_rate
    
#     def project(self):
#         assert("Use a subclass")

#     def forward(self, xi, y):
#         # xi(k+1) = AK_t  xi(k) + BK1_t z(k) + BK2_t y(k)
#         # u(k)    = CK1_t xi(k) + DK1_t z(k) + DK2_t y(k)
#         # v(k)    = CK2_t xi(k) + 0          + DK4_t y(k)
#         # z(k)    = phi_t(v(k))

#         z = self.transform(xi, y)
#         u      = xi @ self.CK1_tT + z @ self.DK1_tT + y @ self.DK2_tT
#         new_xi = xi @ self.AK_tT  + z @ self.BK1_tT + y @ self.BK2_tT
#         return u, new_xi

# class SingleStepNoProj(SingleStepBase):
#     def __init__(self, plant, **kwargs):
#         super().__init__(**kwargs)

#         #_T for transpose
#         #_t for tilde
#         self.AK_tT  = nn.Parameter(uniform(self.state_size, self.state_size))
#         self.BK1_tT = nn.Parameter(uniform(self.hidden_size, self.state_size))
#         self.BK2_tT = nn.Parameter(uniform(self.ob_dim, self.state_size))

#         self.CK1_tT = nn.Parameter(uniform(self.state_size, self.ac_dim))
#         self.DK1_tT = nn.Parameter(uniform(self.hidden_size, self.ac_dim))
#         self.DK2_tT = nn.Parameter(uniform(self.ob_dim, self.ac_dim))

#         self.CK2_tT = nn.Parameter(uniform(self.state_size, self.hidden_size))
#         self.DK4_tT = nn.Parameter(uniform(self.ob_dim, self.hidden_size))

#         self.phi = nn.Tanh()
#         self.A_phi = torch.tensor(0).to(ptu.device)
#         self.B_phi = torch.tensor(1).to(ptu.device)

#         self.transform = None # Transform(self.CK2_tT, self.DK4_tT, self.phi, self.A_phi, self.B_phi)

#     def project(self):
#         pass
#         if self.transform is None:
#             self.transform = Transform(self.CK2_tT, self.DK4_tT, self.phi, self.A_phi, self.B_phi)
#         else:
#             self.transform.CK2_tT = self.CK2_tT
#             self.transform.DK4_tT = self.DK4_tT



# class RNNModel(RecurrentNetwork, nn.Module):
#     """
#     Recurrent Neural Network.

#     xi:  state
#     w:   output of nonlinearity
#     y:   input to RNN (observation)
#     v:   input to nonlinearity (hidden state)
#     u:   output of RNN (action)
#     phi: nonlinearity (activation function)

#     xi(k+1) = AK  xi(k) + BK1 w(k) + BK2 y(k)
#     u(k)    = CK1 xi(k) + DK1 w(k) + DK2 y(k)
#     v(k)    = CK2 xi(k) + DK3 y(k)
#     w(k)    = phi(v(k))
#     """
#     def __init__(self,
#                 obs_space,
#                 action_space,
#                 num_outputs,
#                 model_config,
#                 name):
#         super().__init__()

#                  ac_dim, # action (output) dimension
#                  ob_dim, # observation (input) dimension
#                  state_size,
#                  hidden_size,
#                  plant,
#                  lmi_epsilon,
#                  exponential_stability_rate,
#                  projection_type = 'new',
#                  learning_rate=1e-4,
#                  training=True,
#                  nn_baseline=False,
#                  nn_baseline_n_layers=None,
#                  nn_baseline_size=None,
#                  **kwargs
#                  ):
#         super().__init__(**kwargs)

#         # init vars
#         self.ac_dim = ac_dim
#         self.ob_dim = ob_dim
#         self.state_size = state_size
#         self.hidden_size = hidden_size
#         self.learning_rate = learning_rate
#         self.training = training
#         self.nn_baseline = nn_baseline
#         self.nn_baseline_n_layers = nn_baseline_n_layers
#         self.nn_baseline_size = nn_baseline_size

#         self.projection_type = projection_type

#         print(f'Action dim: {self.ac_dim}, Observation dim: {self.ob_dim}')
#         print(f'RNN state size: {self.state_size}, RNN hidden size: {self.hidden_size}')
#         print(f'Using projection type: {projection_type}')
        
#         ## Parameters of RNN
#         SingleStepClass = SingleStepNoProj
#         self.single_step = SingleStepClass(ac_dim=self.ac_dim, ob_dim=self.ob_dim, state_size=self.state_size,
#             hidden_size=self.hidden_size, plant=plant, lmi_epsilon=lmi_epsilon, 
#             exponential_stability_rate=exponential_stability_rate)

#         self.logstd = nn.Parameter(
#             np.log(0.2) * torch.ones(self.ac_dim, dtype=torch.float32, device=ptu.device) # same initializer as Fangda/Galaxy's code
#         )

#         self.logstd.to(ptu.device)
#         self.optimizer = optim.Adam(
#             itertools.chain([self.logstd], self.single_step.parameters()),
#             self.learning_rate
#         )

#         if nn_baseline:
#             self.baseline = ptu.build_mlp(
#                 input_size=self.ob_dim,
#                 output_size=1,
#                 n_layers=self.nn_baseline_n_layers,
#                 size=self.nn_baseline_size,
#             )
#             self.baseline.to(ptu.device)
#             self.baseline_optimizer = optim.Adam(
#                 self.baseline.parameters(),
#                 self.learning_rate,
#             )
#         else:
#             self.baseline = None

#     ##################################

#     # query the policy with observation(s) to get selected action(s)
#     def get_action(self, obs: np.ndarray, xi = None, evaluation = False) -> np.ndarray:
#         if xi is None:
#             xi = torch.zeros(1, 1, self.state_size).to(ptu.device)
#         observation = ptu.from_numpy(obs).reshape((1, 1, len(obs)))
#         if evaluation == False:
#             action_distribution, xi_next = self(observation, xi)
#             action = action_distribution.sample()[0]  # don't bother with rsample
#         else:
#             action, xi_next = self(observation, xi, evaluation = True)
#             action = action[0]
#         return ptu.to_numpy(action), xi_next

#     # update/train this policy  
#     def update(self, observations, actions, **kwargs):
#         raise NotImplementedError

#     # This function outputs a distribution object representing the policy output for the particular observations.
#     def forward(self, observation: torch.FloatTensor, xi_init = None, resets = None, evaluation = False):
        
#         batch_size = observation.shape[0]
#         time_len = observation.shape[1]
#         us = torch.zeros(batch_size, time_len, self.ac_dim).to(ptu.device)

#         if xi_init is None:
#             xi_init = torch.zeros(batch_size, self.state_size).to(ptu.device)
#         xi = xi_init

#         if resets is None:
#             resets = torch.zeros(batch_size, time_len, 1).to(ptu.device)

#         for k in range(time_len):
#             y = observation[:, k]
#             u, xi = self.single_step(xi, y)
#             reset = resets[:, k]
#             xi = xi*(1 - reset) + xi_init*reset
#             us[:, k] = u

#         if evaluation:
#             return us, xi
#         else:
#             batch_mean = us

#             # Convert to distribution
#             logstd = torch.clamp(self.logstd, -10.0, 2.0)
            
#             scale_tril = torch.diag(torch.exp(logstd))

#             batch_dim = batch_mean.shape[0]
#             batch_scale_tril = scale_tril.repeat(batch_dim, time_len, 1, 1)

#             return distributions.MultivariateNormal(
#                 batch_mean,
#                 scale_tril=batch_scale_tril,
#             ), xi