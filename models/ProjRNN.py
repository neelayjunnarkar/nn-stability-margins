from ray.rllib.utils.annotations import override
from models.RNN import BaseRNN
from models.theta_hat_parameterization import RNNThetaHatParameterization

class ProjRNNModel(BaseRNN, RNNThetaHatParameterization):
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
        **custom_args
    ):
        assert plant_cstor is not None, "plant_cstor parameter is None"

        BaseRNN.__init__(self, obs_space, action_space, num_outputs, model_config, name, **custom_args)

        RNNThetaHatParameterization.__init__(
            self, lmi_eps, exp_stability_rate, plant_cstor, plant_config,
            self.ac_dim, self.ob_dim, self.state_size, self.hidden_size
        )

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
