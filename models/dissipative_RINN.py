import numpy as np
import torch
import torch.nn as nn
import torchdeq
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from torchdeq.norm import apply_norm, reset_norm

from activations import activations_map
from thetahat_dissipativity import Projector
from utils import build_mlp, from_numpy, to_numpy, uniform
from variable_structs import ControllerThetahatParameters


def print_norms(X, name):
    print(
        "{}: largest gain: {:0.3f}, 1: {:0.3f}, 2: {:0.3f}, inf: {:0.3f}, fro: {:0.3f}".format(
            name,
            torch.max(torch.abs(X)),
            torch.linalg.norm(X, 1),
            torch.linalg.norm(X, 2),
            torch.linalg.norm(X, np.inf),
            torch.linalg.norm(X, "fro"),
        )
    )


class DissipativeRINN(RecurrentNetwork, nn.Module):
    """
    A recurrent implicit neural network of the following form: TODO(Neelay)

    Train with a method that calls project after each gradient step.
    """

    # Custom config parameters:
    #   Format: parameter: default, if optional
    #   state_size: 16
    #   nonlin_size: 128
    #   delta: tanh
    #   log_std_init: log(0.2)
    #   dt
    #   baseline_n_layers: 2
    #   baseline_size: 64
    #   plant
    #   plant_config
    #   eps: 1e-6
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

        # State x, nonlinearity Delta output w, input y,
        # nonlineary Delta input v, output u,
        # and nonlinearity Delta.

        # self.deq = torchdeq.get_deq(f_solver="broyden", f_max_iter=30, b_max_iter=30)
        # self.deq = torchdeq.get_deq(f_solver="anderson", f_max_iter=30, b_max_iter=30)
        self.deq = torchdeq.get_deq(f_solver="fixed_point_iter", f_max_iter=30, b_max_iter=30)

        log_std_init = (
            model_config["log_std_init"]
            if "log_std_init" in model_config
            else -1.6094379124341003  # log(0.2)
        )
        self.log_stds = nn.Parameter(log_std_init * torch.ones(self.output_size))

        assert "dt" in model_config
        self.dt = model_config["dt"]

        # fmt: off
        n_layers = model_config["baseline_n_layers"] if "baseline_n_layers" in model_config else 2
        layer_size = model_config["baseline_size"] if "baseline_size" in model_config else 64
        # fmt: on
        self.value = build_mlp(
            input_size=obs_space.shape[0],
            output_size=1,
            n_layers=n_layers,
            size=layer_size,
        )
        self._cur_value = None

        # Define parameters
        # Those with _bar suffix indicate ones that will be used to construct the decision variables.
        # The _T suffix indicates transpose

        self.Duw_T = nn.Parameter(uniform(self.nonlin_size, self.output_size))
        self.S_bar = nn.Parameter(uniform(self.state_size, self.state_size))
        self.R_bar = nn.Parameter(uniform(self.state_size, self.state_size))
        self.Lambda_bar = nn.Parameter(torch.rand(self.nonlin_size))
        self.NA11 = nn.Parameter(uniform(self.state_size, self.state_size))
        self.NA12 = nn.Parameter(uniform(self.state_size, self.input_size))
        self.NA21 = nn.Parameter(uniform(self.output_size, self.state_size))
        self.NA22 = nn.Parameter(uniform(self.output_size, self.input_size))
        self.NB = nn.Parameter(uniform(self.state_size, self.nonlin_size))
        self.NC = nn.Parameter(uniform(self.nonlin_size, self.state_size))
        self.Dvyhat = nn.Parameter(uniform(self.nonlin_size, self.input_size))
        self.Dvwhat = nn.Parameter(uniform(self.nonlin_size, self.nonlin_size))
        # self.Dvwhat = nn.Parameter(torch.zeros((self.nonlin_size, self.nonlin_size)))

        apply_norm(self)

        self.eps = model_config["eps"] if "eps" in model_config else 1e-6

        assert "plant" in model_config
        assert "plant_config" in model_config
        plant = model_config["plant"](model_config["plant_config"])
        np_plant_params = plant.get_params()
        self.plant_params = np_plant_params.np_to_torch(device=self.Duw_T.device)
        assert self.state_size == self.plant_params.Ap.shape[0]

        trs_mode = model_config["trs_mode"] if "trs_mode" in model_config else "variable"
        min_trs = model_config["min_trs"] if "min_trs" in model_config else 1.0

        self.projector = Projector(
            np_plant_params,
            eps=self.eps,
            nonlin_size=self.nonlin_size,
            output_size=self.output_size,
            state_size=self.state_size,
            input_size=self.input_size,
            trs_mode=trs_mode,
            min_trs=min_trs,
        )

        self.oldtheta = None

        self.projected = False  # Set to true within first call to construct_thetahat
        self.project()

    def project(self):
        """Modify parameters to ensure satisfaction of dissipativity condition."""
        # Construct thetahat after training step on thetabar
        self.construct_thetahat()
        with torch.no_grad():
            # fmt: off
            controller_params = ControllerThetahatParameters(
                self.S, self.R, self.NA11, self.NA12, self.NA21, self.NA22,
                self.NB, self.NC, self.Duw_T.t(), self.Dvyhat, self.Dvwhat, self.Lambda,
            )
            # fmt: on
            is_dissipative = self.projector.is_dissipative(controller_params.torch_to_np())
            print(f"Is dissipative: {is_dissipative}")
            if not is_dissipative:
                self.enforce_dissipativity()
                self.construct_thetahat()
        # fmt: off
        thetahat = ControllerThetahatParameters(
            self.S, self.R, self.NA11, self.NA12, self.NA21, self.NA22,
            self.NB, self.NC, self.Duw_T.t(), self.Dvyhat, self.Dvwhat, self.Lambda,
        )
        # fmt: on
        theta, _P = thetahat.torch_construct_theta(self.plant_params)
        self.A_T = theta.Ak.t()
        self.Bw_T = theta.Bkw.t()
        self.By_T = theta.Bky.t()
        self.Cv_T = theta.Ckv.t()
        self.Dvw_T = theta.Dkvw.t()
        self.Dvy_T = theta.Dkvy.t()
        self.Cu_T = theta.Cku.t()
        # self.Duw_T is already a parameter = theta.Dkuw.t()
        self.Duy_T = theta.Dkuy.t()
        # self.construct_theta()

        print_norms(self.A_T.t(), "Ak   ")
        print_norms(self.Bw_T.t(), "Bkw  ")
        print_norms(self.By_T.t(), "Bky  ")
        print_norms(self.Cv_T.t(), "Ckv  ")
        print_norms(self.Dvw_T.t(), "Dkvw ")
        print_norms(self.Dvy_T.t(), "Dkvy ")
        print_norms(self.Cu_T.t(), "Cku  ")
        print_norms(self.Duw_T.t(), "Dkuw ")
        print_norms(self.Duy_T.t(), "Dkuy ")
        theta_mat = theta.matrix(type="torch")
        print_norms(theta_mat, "theta")

        if self.oldtheta is not None:
            print_norms(
                theta_mat - self.oldtheta, f"theta - oldtheta: {torch.allclose(theta_mat, self.oldtheta)}"
            )
        self.oldtheta = theta_mat.detach().clone()

    def construct_thetahat(self):
        """From the _bar parameters, construct the decision variables."""
        if self.projected:
            eps = 0.0
        else:
            eps = self.eps

        assert self.S_bar.ndim == 2
        assert self.S_bar.shape[0] == self.state_size and self.S_bar.shape[1] == self.state_size
        self.S = torch.mm(self.S_bar.t(), self.S_bar) + eps * torch.eye(
            self.state_size, device=self.S_bar.device
        )

        assert self.R_bar.ndim == 2
        assert self.R_bar.shape[0] == self.state_size and self.R_bar.shape[1] == self.state_size
        self.R = torch.mm(self.R_bar.t(), self.R_bar) + eps * torch.eye(
            self.state_size, device=self.R_bar.device
        )

        assert self.Lambda_bar.ndim == 1
        assert self.Lambda_bar.shape[0] == self.nonlin_size
        self.Lambda = self.Lambda_bar.square() + eps * torch.ones(
            self.nonlin_size, device=self.Lambda_bar.device
        )
        self.Lambda = self.Lambda.diag()

        self.projected = True

    def enforce_dissipativity(self):
        """Projects current thetahat parameters to ones that are dissipative."""
        # TODO(Neelay) Consider two levels of epsilon, one for checking dissipativity,
        # and one for enforcing dissipativity, to reduce chatter?

        # fmt: off
        controller_params = ControllerThetahatParameters(
            self.S, self.R, self.NA11, self.NA12, self.NA21, self.NA22,
            self.NB, self.NC, self.Duw_T.t(), self.Dvyhat, self.Dvwhat, self.Lambda,
        )
        # fmt: on

        np_controller_params = controller_params.torch_to_np()
        np_new_controller_params = self.projector.project(np_controller_params)
        new_k = np_new_controller_params.np_to_torch(self.Duw_T.device)

        # TODO(Neelay) Maybe don't need to initialize _bar variables, and
        # can skip directly to thetahat?

        S_bar = np.linalg.cholesky(np_new_controller_params.S).T
        R_bar = np.linalg.cholesky(np_new_controller_params.R).T
        Lambda_bar = np.sqrt(np.diag(np_new_controller_params.Lambda))

        missing, unexpected = self.load_state_dict(
            {
                "S_bar": from_numpy(S_bar, device=self.S_bar.device),
                "R_bar": from_numpy(R_bar, device=self.R_bar.device),
                "NA11": new_k.NA11,
                "NA12": new_k.NA12,
                "NA21": new_k.NA21,
                "NA22": new_k.NA22,
                "NB": new_k.NB,
                "NC": new_k.NC,
                "Duw_T": new_k.Dkuw.t(),
                "Dvyhat": new_k.Dkvyhat,
                "Dvwhat": new_k.Dkvwhat,
                "Lambda_bar": from_numpy(Lambda_bar, device=self.Lambda_bar.device),
            },
            strict=False,
        )
        assert unexpected == [], f"Loading unexpected key after projection: {unexpected}"
        assert missing == [
            "log_stds",
            "value.0.bias",
            "value.0.weight_g",
            "value.0.weight_v",
            "value.2.bias",
            "value.2.weight_g",
            "value.2.weight_v",
            "value.4.bias",
            "value.4.weight_g",
            "value.4.weight_v",
        ], missing

    # def construct_theta(self):
    #     """Construct theta, the parameters of the controller, from thetahat,
    #     the decision variables for the dissipativity condition."""

    #     # torch.linalg.solve(A, B) solves for X in AX = B, and assumes A is invertible

    #     # Construct V and U by solving VU^T = I - RS
    #     # V = self.R
    #     # U = torch.linalg.solve(
    #     #     V, torch.eye(self.state_size, device=V.device) - torch.mm(self.R, self.S)
    #     # ).t()

    #     svdU, svdSigma, svdV_T = torch.linalg.svd(torch.eye(self.state_size) - self.R @ self.S)
    #     sqrt_svdSigma = torch.diag(torch.sqrt(svdSigma))
    #     V = svdU @ sqrt_svdSigma
    #     U = svdV_T.t() @ sqrt_svdSigma

    #     print_norms(U, "U")
    #     print_norms(torch.linalg.inv(U), "Uinv")
    #     print_norms(V, "V")
    #     print_norms(torch.linalg.inv(V), "Vinv")

    #     # Construct P via P = [I, S; 0, U^T] Y^-1
    #     # Actually no need to do this

    #     # Reconstruct Dvy and Dvw from Dvyhat and Dvwhat
    #     Dvy = torch.linalg.solve(self.Lambda, self.Dvyhat)
    #     Dvw = torch.linalg.solve(self.Lambda, self.Dvwhat)

    #     # # Solve for A, By, Cu, and Duy from NA
    #     # # NA = NA1 + NA2 NA3 NA4
    #     # NA = torch.vstack(
    #     #     (torch.hstack((self.NA11, self.NA12)), torch.hstack((self.NA21, self.NA22)))
    #     # )

    #     # NA111 = torch.mm(self.S, torch.mm(self.Ap, self.R))
    #     # NA112 = V.new_zeros((self.state_size, self.Cpy.shape[0]))
    #     # NA121 = V.new_zeros((self.Bpu.shape[1], self.state_size))
    #     # NA122 = V.new_zeros((self.Bpu.shape[1], self.Cpy.shape[0]))
    #     # NA1 = torch.vstack((torch.hstack((NA111, NA112)), torch.hstack((NA121, NA122))))

    #     # NA211 = U
    #     # NA212 = torch.mm(self.S, self.Bpu)
    #     # NA221 = V.new_zeros((self.Bpu.shape[1], U.shape[1]))
    #     # NA222 = torch.eye(self.Bpu.shape[1], device=V.device)
    #     # NA2 = torch.vstack((torch.hstack((NA211, NA212)), torch.hstack((NA221, NA222))))

    #     # NA411 = V.t()
    #     # NA412 = V.new_zeros(V.shape[1], self.Cpy.shape[0])
    #     # NA421 = torch.mm(self.Cpy, self.R)
    #     # NA422 = torch.eye(self.Cpy.shape[0], device=V.device)
    #     # NA4 = torch.vstack((torch.hstack((NA411, NA412)), torch.hstack((NA421, NA422))))

    #     # NA3NA4 = torch.linalg.solve(NA2, NA - NA1)
    #     # NA3 = torch.linalg.solve(NA4.t(), NA3NA4.t()).t()

    #     # A = NA3[: self.state_size, : self.state_size]
    #     # By = NA3[: self.state_size, self.state_size :]
    #     # Cu = NA3[self.state_size :, : self.state_size]
    #     # Duy = NA3[self.state_size :, self.state_size :]

    #     Ap = self.plant_params.Ap
    #     Bpu = self.plant_params.Bpu
    #     Cpy = self.plant_params.Cpy

    #     Duy = self.NA22
    #     By = torch.linalg.solve(U, self.NA12 - self.S @ Bpu @ Duy)
    #     Cu = torch.linalg.solve(V, self.NA21.t() - self.R @ Cpy.t() @ Duy.t()).t()
    #     AVT = torch.linalg.solve(
    #         U,
    #         self.NA11
    #         - U @ By @ Cpy @ self.R
    #         - self.S @ Bpu @ (Cu @ V.t() + Duy @ Cpy @ self.R)
    #         - self.S @ Ap @ self.R,
    #     )
    #     A = torch.linalg.solve(V, AVT.t()).t()

    #     # Solve for Bw from NB
    #     # NB = NB1 + U Bw
    #     NB1 = torch.mm(self.S, torch.mm(Bpu, self.Duw_T.t()))
    #     Bw = torch.linalg.solve(U, self.NB - NB1)

    #     # Solve for Cv from NC
    #     # NC = NC1 + Lambda Cv V^T
    #     NC1 = torch.mm(self.Dvyhat, torch.mm(Cpy, self.R))
    #     CvVT = torch.linalg.solve(self.Lambda, self.NC - NC1)
    #     Cv = torch.linalg.solve(V, CvVT.t()).t()
    #     # Bring together the theta parameters here
    #     self.A_T = A.t()
    #     self.Bw_T = Bw.t()
    #     self.By_T = By.t()

    #     self.Cv_T = Cv.t()
    #     self.Dvw_T = Dvw.t()
    #     self.Dvy_T = Dvy.t()

    #     self.Cu_T = Cu.t()
    #     # self.Duw_T is already a parameter
    #     self.Duy_T = Duy.t()

    #     # print(A, self.A_T)

    #     assert self.A_T.shape == (self.state_size, self.state_size)
    #     assert self.Bw_T.shape == (self.nonlin_size, self.state_size)
    #     assert self.By_T.shape == (self.input_size, self.state_size)

    #     assert self.Cv_T.shape == (self.state_size, self.nonlin_size)
    #     assert self.Dvw_T.shape == (self.nonlin_size, self.nonlin_size)
    #     assert self.Dvy_T.shape == (self.input_size, self.nonlin_size)

    #     assert self.Cu_T.shape == (self.state_size, self.output_size)
    #     assert self.Duw_T.shape == (self.nonlin_size, self.output_size)
    #     assert self.Duy_T.shape == (self.input_size, self.output_size)

    @override(ModelV2)
    def get_initial_state(self):
        # Place hidden states on same device as model.
        x0 = self.A_T.new_zeros(self.state_size)
        return [x0]

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    def xdot(self, x, y, w=None, w0=None):
        """Computes solution for derivative of x given x and y, and possibly w or an initial guess for w."""
        # w is the solution w = Delta(v), and w0 is a guess for the solution w = Delta(v)
        if w is None:

            def delta_tilde(w):
                v = x @ self.Cv_T + w @ self.Dvw_T + y @ self.Dvy_T
                return self.delta(v)

            reuse = True
            if w0 is None:
                assert len(x.shape) == 2
                w0 = x.new_zeros((x.shape[0], self.nonlin_size))
                reuse = False

            solver_kwargs = {"f_max_iter": 5} if reuse else {}
            w, info = self.deq(delta_tilde, w0, solver_kwargs=solver_kwargs)
            assert len(w) > 0
            w = w[-1]

        xdot = x @ self.A_T + w @ self.Bw_T + y @ self.By_T
        return xdot, w

    def next_state(self, x, y, w=None):
        """Compute the next state using the Runge-Kutta 4th-order method,
        assuming y is constant over the time step.
        Optionally takes the solution for w at the time of the current state."""
        k1, w1 = self.xdot(x, y, w=w)
        k2, w2 = self.xdot(x + self.dt * k1 / 2, y, w0=w1)
        k3, w3 = self.xdot(x + self.dt * k2 / 2, y, w0=w2)
        k4, w4 = self.xdot(x + self.dt * k3, y, w0=w3)
        next_x = x + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_x, w4

    @override(RecurrentNetwork)
    def forward_rnn(self, obs, state, seq_lens):
        reset_norm(self)

        assert len(state) == 1
        xkp1 = state[0]  # x[k+1], but actually used to initialize x[0]
        batch_size = obs.shape[0]
        time_len = obs.shape[1]
        actions = xkp1.new_zeros((batch_size, time_len, self.output_size))

        wkp10 = xkp1.new_zeros((batch_size, self.nonlin_size))
        reuse = False

        assert not torch.any(torch.isnan(obs))
        assert not torch.any(torch.isnan(xkp1)), xkp1

        for k in range(time_len):
            # Set action for time k

            xk = xkp1
            yk = obs[:, k]

            assert not torch.any(torch.isnan(xk)), f"At time {k}, xk has nans"
            assert not torch.any(torch.isnan(yk)), f"At time {k}, yk has nans"

            # Solve for wk
            wk0 = wkp10
            assert not torch.any(torch.isnan(wk0)), f"At time {k}, wk0 has nans"

            def delta_tilde(w):
                v = xk @ self.Cv_T + w @ self.Dvw_T + yk @ self.Dvy_T
                return self.delta(v)

            solver_kwargs = {"f_max_iter": 5} if reuse else {}
            wk, info = self.deq(delta_tilde, wk0, solver_kwargs=solver_kwargs)
            assert len(wk) > 0
            wk = wk[-1]

            if torch.any(torch.isnan(wk)):
                # Retry with different initialization
                reuse = False
                wk0 = xkp1.new_zeros((batch_size, self.nonlin_size))
                assert not torch.any(torch.isnan(wk0)), f"At time {k}, wk0 has nans"
                wk, info = self.deq(delta_tilde, wk0)
                assert len(wk) > 0
                wk = wk[-1]
            assert not torch.any(torch.isnan(wk)), f"At time {k}, wk has nans: {wk}, {info}"

            uk = xk @ self.Cu_T + wk @ self.Duw_T + yk @ self.Duy_T
            actions[:, k] = uk

            assert not torch.any(torch.isnan(uk)), f"At time {k}, uk has nans"

            # Compute next state xkp1
            # wkp10 is the initial guess for w[k+1]
            xkp1, wkp10 = self.next_state(xk, yk, w=wk)
            reuse = True

        assert not torch.any(torch.isnan(actions))
        assert not torch.any(torch.isnan(self.log_stds))

        log_stds_rep = self.log_stds.repeat((batch_size, time_len, 1))
        outputs = torch.cat((actions, log_stds_rep), dim=2)

        self._cur_value = self.value(obs)
        self._cur_value = self._cur_value.reshape([-1])

        return outputs, [xkp1]
