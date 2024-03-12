"""
Activation functions.
"""

import torch.nn as nn


class ReLU(nn.ReLU):
    def __init__(self):
        super().__init__()


class LeakyReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2)


class Tanh(nn.Tanh):
    def __init__(self):
        super().__init__()


activations_map = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "tanh": Tanh,
}
