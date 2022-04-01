import torch
import torch.nn as nn
from models.utils import uniform
from deq_lib.solvers import broyden

class ImplicitModel(nn.Module):
    """
    Learns a model of the form:
    F(x, u) = Ax + B1 q + B2 u
    q = Delta(C2 x + D3 q)
    with sizes
    F(x, u): state size,
    x: state size,
    q: nonlin size,
    u: action size
    """
    def __init__(
        self,
        action_size,
        state_size,
        nonlin_size,
        delta_cstor,
        solver = broyden
    ):
        nn.Module.__init__(self)

        self.action_size = action_size
        self.state_size = state_size
        self.nonlin_size = nonlin_size

        #_T for transpose

        self.A_T  = nn.Parameter(uniform(self.state_size,  self.state_size))
        self.B1_T = nn.Parameter(uniform(self.nonlin_size, self.state_size))
        self.B2_T = nn.Parameter(uniform(self.action_size, self.state_size))

        self.C2_T = nn.Parameter(uniform(self.state_size, self.nonlin_size))
        self.D3_T = nn.Parameter(uniform(self.nonlin_size, self.nonlin_size))

        self.delta = delta_cstor()
        self.solver = solver
        self.f_thresh = 30
        self.b_thresh = 30
        self.hook = None

    def forward(self, xs, us):
        assert xs.shape[1] == self.state_size
        assert us.shape[1] == self.action_size
        assert xs.shape[0] == us.shape[0]
        batch_size = xs.shape[0]
        
        x = xs.reshape(xs.shape[0], 1, xs.shape[1])
        q0 = torch.zeros(batch_size, 1, self.nonlin_size).to(xs.device)
        with torch.no_grad():
            q_star = self.solver(
                lambda q: self.delta(x @ self.C2_T + q @ self.D3_T),
                q0,
                threshold = self.f_thresh
            )['result']
            new_q_star = q_star
        
        if self.training:
            q_star.requires_grad_()
            new_q_star = self.delta(x @ self.C2_T + q_star @ self.D3_T)

            def backward_hook(grad):
                if self.hook is not None:
                    self.hook.remove()
                new_grad = self.solver(
                    lambda g: torch.autograd.grad(new_q_star, q_star, g, retain_graph = True)[0] + grad,
                    torch.zeros_like(grad).to(xs.device),
                    threshold = self.b_thresh
                )['result']
                return new_grad
            
            self.hook = new_q_star.register_hook(backward_hook)
        
        return (x @ self.A_T + new_q_star @ self.B1_T).reshape(batch_size, self.state_size) + us @ self.B2_T
