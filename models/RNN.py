import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override

from activations import activations_map
from utils import build_mlp, uniform


class RNN(RecurrentNetwork, nn.Module):
    """
    A controller of the form

    xdot(t) = A  x(t) + Bw  w(t) + By  y(t)
    v(t)    = Cv x(t)            + Dvy y(t)
    u(t)    = Cu x(t) + Duw w(t) + Duy y(t)
    w(t)    = phi(v(t))

    where x is the state, v is the input to the nonlinearity phi,
    w is the output of the nonlinearity phi, y is the input,
    and u is the output.

    Train with a method that calls project after each gradient step.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        model_config = model_config["custom_model_config"]

        assert (
            2 * action_space.shape[0] == num_outputs
        ), "Num outputs should be 2 * action dimension"

        self.state_size = model_config["state_size"] if "state_size" in model_config else 16
        self.nonlin_size = model_config["nonlin_size"] if "nonlin_size" in model_config else 128
        self.input_size = obs_space.shape[0]
        self.output_size = action_space.shape[0]

        if "delta" not in model_config:
            model_config["delta"] = "tanh"
        if model_config["delta"] not in activations_map:
            raise ValueError(
                f"Activation {model_config['delta']} is not in activations.activations_map."
            )
        self.delta = activations_map[model_config["delta"]]()

        # _T for transpose
        # State x, nonlinearity Delta output w, input y,
        # nonlineary Delta input v, output u,
        # and nonlinearity Delta.
        self.A_T = nn.Parameter(uniform(self.state_size, self.state_size))
        self.Bw_T = nn.Parameter(uniform(self.nonlin_size, self.state_size))
        self.By_T = nn.Parameter(uniform(self.input_size, self.state_size))

        self.Cv_T = nn.Parameter(uniform(self.state_size, self.nonlin_size))
        self.Dvy_T = nn.Parameter(uniform(self.input_size, self.nonlin_size))

        self.Cu_T = nn.Parameter(uniform(self.state_size, self.output_size))
        self.Duw_T = nn.Parameter(uniform(self.nonlin_size, self.output_size))
        self.Duy_T = nn.Parameter(uniform(self.input_size, self.output_size))

        log_std_init = (
            model_config["log_std_init"]
            if "log_std_init" in model_config
            else -1.6094379124341003  # log(0.2)
        )
        self.log_stds = nn.Parameter(log_std_init * torch.ones(self.output_size))

        assert "dt" in model_config
        self.dt = model_config["dt"]

        self.value = build_mlp(
            input_size=obs_space.shape[0],
            output_size=1,
            n_layers=model_config["baseline_n_layers"]
            if "baseline_n_layers" in model_config
            else 2,
            size=model_config["baseline_size"] if "baseline_size" in model_config else 64,
        )
        self._cur_value = None

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        x0 = self.A_T.new_zeros(self.state_size)
        return [x0]

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def xdot(self, x, y, w=None):
        if w is None:
            v = x @ self.Cv_T + y @ self.Dvy_T
            w = self.delta(v)
        xdot = x @ self.A_T + w @ self.Bw_T + y @ self.By_T
        return xdot

    def next_state(self, x, y, w=None):
        # Compute the next state using the Runge-Kutta 4th-order method,
        # assuming y is constant over the time step.
        k1 = self.xdot(x, y, w=w)
        k2 = self.xdot(x + self.dt * k1 / 2, y)
        k3 = self.xdot(x + self.dt * k2 / 2, y)
        k4 = self.xdot(x + self.dt * k3, y)
        next_x = x + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_x

    @override(RecurrentNetwork)
    def forward_rnn(self, obs, state, seq_lens):
        assert len(state) == 1
        xkp1 = state[0]  # x[k+1], but actually used to initialize x[0]
        batch_size = obs.shape[0]
        time_len = obs.shape[1]
        actions = xkp1.new_zeros((batch_size, time_len, self.output_size))

        for k in range(time_len):
            # Set action for time k
            xk = xkp1
            yk = obs[:, k]
            vk = xk @ self.Cv_T + yk @ self.Dvy_T
            wk = self.delta(vk)
            uk = xk @ self.Cu_T + wk @ self.Duw_T + yk @ self.Duy_T
            actions[:, k] = uk
            # Compute next state
            xkp1 = self.next_state(xk, yk, w=wk)

        log_stds_rep = self.log_stds.repeat((batch_size, time_len, 1))
        outputs = torch.cat((actions, log_stds_rep), dim=2)

        self._cur_value = self.value(obs)
        self._cur_value = self._cur_value.reshape([-1])

        return outputs, [xkp1]
