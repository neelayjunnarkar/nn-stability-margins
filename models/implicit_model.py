# Defines an Implicit Neural Network for use in RLLib

import torch
import torch.nn as nn
import torchdeq
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torchdeq.norm import apply_norm, reset_norm

from activations import activations_map
from utils import build_mlp, uniform


class ImplicitModel(TorchModelV2, nn.Module):
    """
    A model of the form
    y = Cx + Du
    x = Delta(Ax + Bu)
    where u is the input, x is the state, y is the output, and
    Delta is a nonlinearity.

    Train with a method that calls project after each gradient step.

    See Implicit Deep Learning, https://epubs.siam.org/doi/abs/10.1137/20M1358517
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        model_config = model_config["custom_model_config"]

        self.input_size = obs_space.shape[0]
        self.state_size = (
            model_config["state_size"] if "state_size" in model_config else 128
        )
        self.output_size = num_outputs

        # _T for transpose

        self.A_T = nn.Parameter(uniform(self.state_size, self.state_size))
        self.B_T = nn.Parameter(uniform(self.input_size, self.state_size))
        self.C_T = nn.Parameter(uniform(self.state_size, self.output_size))
        self.D_T = nn.Parameter(uniform(self.input_size, self.output_size))

        if "delta" not in model_config:
            model_config["delta"] = "tanh"
        if model_config["delta"] not in activations_map:
            raise ValueError(
                f"Activation {model_config['delta']} is not in activations.activations_map."
            )
        self.delta = activations_map[model_config["delta"]]()

        # self.deq = torchdeq.get_deq(f_solver="broyden", f_max_iter=30, b_max_iter=30)
        self.deq = torchdeq.get_deq(
            f_solver="fixed_point_iter", f_max_iter=30, b_max_iter=30
        )

        self.value = build_mlp(
            input_size=obs_space.shape[0],
            output_size=1,
            n_layers=model_config["baseline_n_layers"]
            if "baseline_n_layers" in model_config
            else 2,
            size=model_config["baseline_size"]
            if "baseline_size" in model_config
            else 64,
        )

        apply_norm(self, filter_out=["C_T", "D_T"])

        self.project()

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        reset_norm(self)

        obs = input_dict["obs_flat"].float()
        self._last_obs = obs.reshape(obs.shape[0], -1)
        assert self._last_obs.shape[1] == self.input_size

        def delta_tilde(x):
            return self.delta(x @ self.A_T + self._last_obs @ self.B_T)

        x0 = torch.zeros(
            (self._last_obs.shape[0], self.state_size), device=self._last_obs.device
        )
        x, info = self.deq(delta_tilde, x0)
        x = x[-1]

        y = x @ self.C_T + self._last_obs @ self.D_T

        assert y.shape[0] == self._last_obs.shape[0] and y.shape[1] == self.output_size
        return y, state

    @override(TorchModelV2)
    def value_function(self):
        value = self.value(self._last_obs).squeeze(1)
        return value

    def project(self):
        # Modify parameters to ensure existence and uniqueness of solution to implicit equation.
        with torch.no_grad():
            # Row sum of A is column sum of A_T
            max_abs_row_sum = torch.linalg.matrix_norm(self.A_T, ord=1)
            if max_abs_row_sum > 0.999:
                self.A_T *= 0.99 / max_abs_row_sum
                new_max_abs_row_sum = torch.linalg.matrix_norm(self.A_T, ord=1)
                print(
                    f"Reducing max abs row sum: {max_abs_row_sum} -> {new_max_abs_row_sum}"
                )
        print(
            torch.max(
                torch.tensor(
                    [x.abs().max() for x in [self.A_T, self.B_T, self.C_T, self.D_T]]
                )
            )
        )
