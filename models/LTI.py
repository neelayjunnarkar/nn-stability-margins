import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override

import lti_controllers
from theta_dissipativity import LTIProjector as LTIThetaProjector
from thetahat_dissipativity import LTIProjector as LTIThetahatProjector
from utils import build_mlp, from_numpy, to_numpy
from variable_structs import ControllerLTIThetaParameters


class LTIModel(RecurrentNetwork, nn.Module):
    """
    An LTI system of the following form:

    xdot(t) = A  x(t) + By  y(t)
    u(t)    = Cu x(t) + Duy y(t)
    """

    # Arguments:
    #   plant
    #   plant_config
    #   lti_controller
    #   lti_controller_kwargs: Defaults to empty dictionary.
    #   learn: Defaults to False.
    #   log_std_init: Defaults to log(2)
    #   baseline_n_layers: Defaults to 2.
    #   baseline_size: Defaults to 64.
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        *args,
        **kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        model_config = model_config["custom_model_config"]

        assert (
            2 * action_space.shape[0] == num_outputs
        ), "Num outputs should be 2 * action dimension"

        self.state_size = model_config["state_size"] if "state_size" in model_config else 16
        self.input_size = obs_space.shape[0]
        self.output_size = action_space.shape[0]

        log_std_init = (
            model_config["log_std_init"]
            if "log_std_init" in model_config
            else -1.6094379124341003  # log(0.2)
        )
        self.learn = model_config["learn"] if "learn" in model_config else False
        if self.learn:
            self.log_stds = nn.Parameter(log_std_init * torch.ones(self.output_size))
        else:
            self.log_stds = torch.zeros(self.output_size)

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

        self.eps = model_config["eps"] if "eps" in model_config else 1e-6

        assert "plant" in model_config
        assert "plant_config" in model_config
        plant = model_config["plant"](model_config["plant_config"])
        np_plant_params = plant.get_params()
        self.plant_params = np_plant_params.np_to_torch(device=self.log_stds.device)

        assert "lti_controller" in model_config
        lti_controller = model_config["lti_controller"]
        if lti_controller not in lti_controllers.controller_map:
            raise ValueError(
                f"LTI controller function {lti_controller} is not in lti_controllers.controller_map."
            )

        lti_controller_kwargs = (
            model_config["lti_controller_kwargs"] if "lti_controller_kwargs" in model_config else {}
        )
        lti_controller_kwargs["state_size"] = self.state_size
        lti_controller_kwargs["input_size"] = self.input_size
        lti_controller_kwargs["output_size"] = self.output_size
        lti_controller, info = lti_controllers.controller_map[lti_controller](
            np_plant_params, **lti_controller_kwargs
        )
        lti_controller = lti_controller.np_to_torch(device=self.log_stds.device)

        # _T for transpose
        # State x, input y, and output u.
        self.A_T = lti_controller.Ak.t()
        self.By_T = lti_controller.Bky.t()
        self.Cu_T = lti_controller.Cku.t()
        self.Duy_T = lti_controller.Dkuy.t()

        if self.learn:
            self.A_T = nn.Parameter(self.A_T)
            self.By_T = nn.Parameter(self.By_T)
            self.Cu_T = nn.Parameter(self.Cu_T)
            self.Duy_T = nn.Parameter(self.Duy_T)

        if "P" in model_config:
            self.P = from_numpy(model_config["P"])
        elif "P" in info:
            print("Using P from LTI initialization.")
            self.P = from_numpy(info["P"], device=self.A_T.device)
        else:
            self.P = torch.eye(
                self.plant_params.Ap.shape[0] + self.state_size, device=self.A_T.device
            )

        self.theta_projector = LTIThetaProjector(
            np_plant_params, self.eps, self.output_size, self.state_size, self.input_size
        )

        trs_mode = model_config["trs_mode"] if "trs_mode" in model_config else "variable"
        min_trs = model_config["min_trs"] if "min_trs" in model_config else 1.0

        self.thethat_projector = LTIThetahatProjector(
            np_plant_params,
            self.eps,
            self.output_size,
            self.state_size,
            self.input_size,
            trs_mode,
            min_trs,
        )

    def project(self):
        """Modify parameters to ensure satisfaction of dissipativity condition."""
        if not self.learn:
            return

        with torch.no_grad():
            controller_params = ControllerLTIThetaParameters(
                self.A_T.t(), self.By_T.t(), self.Cu_T.t(), self.Duy_T.t()
            )
            is_dissipative, P = self.theta_projector.is_dissipative(
                controller_params.torch_to_np(), to_numpy(self.P)
            )
            print(f"Is dissipative: {is_dissipative}")
            if is_dissipative:
                self.P = from_numpy(P, device=self.P.device)
            else:
                self.enforce_dissipativity()

    def enforce_dissipativity(self):
        """Converts current theta, P, Lambda to thetahat and projects to dissipative set."""
        assert self.learn

        theta = ControllerLTIThetaParameters(
            self.A_T.t(), self.By_T.t(), self.Cu_T.t(), self.Duy_T.t()
        )
        thetahat = theta.torch_construct_thetahat(self.P, self.plant_params)

        thetahat = thetahat.torch_to_np()
        new_thetahat = self.thethat_projector.project(thetahat)
        new_thetahat = new_thetahat.np_to_torch(device=self.A_T.device)

        _new_theta, P = new_thetahat.torch_construct_theta(self.plant_params)
        self.P = P

        new_new_theta = self.theta_projector.project(theta.torch_to_np(), to_numpy(self.P))
        new_new_theta = new_new_theta.np_to_torch(device=self.A_T.device)

        new_k = new_new_theta

        missing, unexpected = self.load_state_dict(
            {
                "A_T": new_k.Ak.t(),
                "By_T": new_k.Bky.t(),
                "Cu_T": new_k.Cku.t(),
                "Duy_T": new_k.Dkuy.t(),
            },
            strict=False,
        )
        assert unexpected == [], f"Loading unexpected key after projection: {unexpected}"
        # fmt: off
        assert missing ==  ['log_stds', 'value.0.weight', 'value.0.bias', 'value.2.weight', 'value.2.bias', 'value.4.weight', 'value.4.bias'], missing
        # fmt: on

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        x0 = self.A_T.new_zeros(self.state_size)
        return [x0]

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def xdot(self, x, y):
        xdot = x @ self.A_T + y @ self.By_T
        return xdot

    def next_state(self, x, y):
        # Compute the next state using the Runge-Kutta 4th-order method,
        # assuming y is constant over the time step.
        k1 = self.xdot(x, y)
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

        assert not torch.any(torch.isnan(xkp1))

        for k in range(time_len):
            # Set action for time k

            xk = xkp1
            yk = obs[:, k]

            uk = xk @ self.Cu_T + yk @ self.Duy_T
            actions[:, k] = uk

            # Compute next state xkp1
            # wkp10 is the initial guess for w[k+1]
            xkp1 = self.next_state(xk, yk)

        log_stds_rep = self.log_stds.repeat((batch_size, time_len, 1))
        outputs = torch.cat((actions, log_stds_rep), dim=2)

        self._cur_value = self.value(obs)
        self._cur_value = self._cur_value.reshape([-1])

        return outputs, [xkp1]
