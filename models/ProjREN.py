from ray.rllib.utils.annotations import override
from models.RNN import BaseRNN
from models.theta_hat_parameterization import RENThetaHatParameterization
import torch
from deq_lib.solvers import broyden, anderson

class ProjRENModel(BaseRNN, RENThetaHatParameterization):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        lmi_eps = 1e-5,
        exp_stability_rate = 0.98,
        plant_cstor = None,
        plant_config = None,
        solver = broyden,
        f_thresh = 30,
        b_thresh = 30,
        **custom_args
    ):
        assert plant_cstor is not None, "plant_cstor parameter is None"

        BaseRNN.__init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_args)

        RENThetaHatParameterization.__init__(
            self, lmi_eps, exp_stability_rate, plant_cstor, plant_config,
            self.ac_dim, self.ob_dim, self.state_size, self.hidden_size
        )

        self.solver = solver
        self.f_thresh = f_thresh
        self.b_thresh = b_thresh

        self.hooks = []

    def base_phi_t(self, xi, z, y):
        v = xi @ self.CK2_tT + z @ self.DK3_tT + y @ self.DK4_tT
        if self._scalar_bounds:
            z_next = self.L_phi_inv * (self.phi(v) - self.S_phi * v)
        else:
            z_next = self.L_phi_inv @ (self.phi(v) - self.S_phi @ v)
        return z_next

    @override(BaseRNN)
    def phi_t(self, state, obs):
        """Loop transformed phi"""
        # v(k) = CK2_t xi(k) + DK3_t z*(k) + DK4_t y(k)
        # z*(k) = phi_t(v(k))
        
        xi = state.reshape(state.shape[0], 1, state.shape[1])
        y = obs.reshape(obs.shape[0], 1, obs.shape[1])
        batch_size = xi.shape[0]

        z0 = torch.zeros(batch_size, 1, self.hidden_size)
        with torch.no_grad():
            z_star = self.solver(lambda z: self.base_phi_t(xi, z, y), z0, threshold = self.f_thresh)['result']
            new_z_star = z_star

        if self.training:
            z_star.requires_grad_()
            new_z_star = self.base_phi_t(xi, z_star, y)

            def backward_hook(grad):
                if self.hooks:
                    self.hooks[-1].remove()
                    del self.hooks[-1]
                    # if ptu.device == 'cuda':
                    # torch.cuda.synchronize()
                
                new_grad = self.solver(
                    lambda g: torch.autograd.grad(new_z_star, z_star, g, retain_graph = True)[0] + grad,
                    torch.zeros_like(grad), threshold = self.b_thresh
                )['result']
                return new_grad
            
            self.hooks.append(new_z_star.register_hook(backward_hook))
        
        return new_z_star.reshape(new_z_star.shape[0], new_z_star.shape[2])