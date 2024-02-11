# Defines a feed forward fully-connected neural network for use in RLLib

import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from utils import build_mlp


class FullyConnectedNetwork(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        model_config = model_config["custom_model_config"]

        self.nn = build_mlp(
            input_size=obs_space.shape[0],
            output_size=num_outputs,
            n_layers=model_config["n_layers"] if "n_layers" in model_config else 2,
            size=model_config["size"] if "size" in model_config else 128,
            activation=model_config["activation"] if "activation" in model_config else "tanh",
            output_activation=model_config["output_activation"]
            if "output_activation" in model_config
            else "identity",
        )

        self.value = build_mlp(
            input_size=obs_space.shape[0],
            output_size=1,
            n_layers=model_config["baseline_n_layers"]
            if "baseline_n_layers" in model_config
            else 2,
            size=model_config["baseline_size"] if "baseline_size" in model_config else 64,
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_obs = obs.reshape(obs.shape[0], -1)
        outputs = self.nn(self._last_obs)
        return outputs, state

    @override(TorchModelV2)
    def value_function(self):
        value = self.value(self._last_obs).squeeze(1)
        return value
