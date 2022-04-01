"""
Activation functions with sector-bound information.
"""

import torch
import torch.nn as nn



class LeakyReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2)
        self.A_phi = torch.tensor(0.2)
        self.B_phi = torch.tensor(1.0)

class Tanh(nn.Tanh):
    def __init__(self):
        super().__init__()
        self.A_phi = torch.tensor(0.0)
        self.B_phi = torch.tensor(1.0)
