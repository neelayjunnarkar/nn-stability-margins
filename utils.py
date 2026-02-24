import torch
import torch.nn as nn
import math
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
import os

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation = 'tanh',
        output_activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


def uniform(output_size, input_size, lower_bound = None, upper_bound = None):
    if input_size == 0 or output_size == 0:
        return torch.zeros((output_size, input_size))
    if lower_bound is None:
        lower_bound = -1/math.sqrt(input_size)
        upper_bound = -lower_bound
    return (upper_bound - lower_bound)*torch.rand(output_size, input_size) + lower_bound

def from_numpy(array, device=None):
    out = torch.tensor(array.astype(np.float32))
    if device is not None:
        return out.to(device)
    else:
        return out

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()

class ExportWeightsCallback(DefaultCallbacks):
    def on_train_result(self, *, trainer, result, **kwargs):
        trial_dir = trainer.logdir
        export_path = os.path.join(trial_dir, "latest_model_state_dict.pth")

        policy = trainer.get_policy("default_policy")
        model = policy.model

        additional = {
            "P": model.P,
            "Lambda": model.Lambda,
            "MDeltapvv": model.MDeltapvv,
            "MDeltapvw": model.MDeltapvw,
            "MDeltapww": model.MDeltapww,
        }
        data = model.state_dict() | additional
        torch.save(data, export_path)

        print(f"\n--- Standalone weights saved to: {export_path} ---")
