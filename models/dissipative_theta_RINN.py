import numpy as np
import torch
import torch.nn as nn
import torchdeq
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from torchdeq.norm import apply_norm, reset_norm

from activations import activations_map
from theta_dissipativity import Projector as ThetaProjector
from thetahat_dissipativity import Projector as ThetahatProjector
from utils import build_mlp, from_numpy, to_numpy, uniform
from variable_structs import ControllerThetahatParameters, ControllerThetaParameters


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


class DissipativeThetaRINN(RecurrentNetwork, nn.Module):
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

        # _T for transpose
        # State x, nonlinearity Delta output w, input y,
        # nonlineary Delta input v, output u,
        # and nonlinearity Delta.
        self.A_T = nn.Parameter(uniform(self.state_size, self.state_size))
        self.Bw_T = nn.Parameter(uniform(self.nonlin_size, self.state_size))
        self.By_T = nn.Parameter(uniform(self.input_size, self.state_size))

        self.Cv_T = nn.Parameter(uniform(self.state_size, self.nonlin_size))
        # self.Dvw_T = nn.Parameter(uniform(self.nonlin_size, self.nonlin_size))
        self.Dvw_T = nn.Parameter(torch.zeros((self.nonlin_size, self.nonlin_size)))
        self.Dvy_T = nn.Parameter(uniform(self.input_size, self.nonlin_size))

        self.Cu_T = nn.Parameter(uniform(self.state_size, self.output_size))
        self.Duw_T = nn.Parameter(uniform(self.nonlin_size, self.output_size))
        self.Duy_T = nn.Parameter(uniform(self.input_size, self.output_size))

        apply_norm(self, filter_out=["A_T", "Bw_T", "By_T", "Cu_T", "Duw_T", "Duy_T"])

        self.eps = model_config["eps"] if "eps" in model_config else 1e-6

        assert "plant" in model_config
        assert "plant_config" in model_config
        plant = model_config["plant"](model_config["plant_config"])
        np_plant_params = plant.get_params()
        self.plant_params = np_plant_params.np_to_torch(device=self.A_T.device)
        assert self.state_size == self.plant_params.Ap.shape[0]

        trs_mode = model_config["trs_mode"] if "trs_mode" in model_config else "variable"
        min_trs = model_config["min_trs"] if "min_trs" in model_config else 1.0

        self.theta_projector = ThetaProjector(
            np_plant_params,
            eps=self.eps,
            nonlin_size=self.nonlin_size,
            output_size=self.output_size,
            state_size=self.state_size,
            input_size=self.input_size,
        )

        self.thetahat_projector = ThetahatProjector(
            np_plant_params,
            eps=self.eps,
            nonlin_size=self.nonlin_size,
            output_size=self.output_size,
            state_size=self.state_size,
            input_size=self.input_size,
            trs_mode=trs_mode,
            min_trs=min_trs,
        )

        if "P" in model_config:
            self.P = from_numpy(model_config["P"], device=self.A_T.device)
        else:
            self.P = torch.eye(self.plant_params.Ap.shape[0] + self.state_size)
        if "Lambda" in model_config:
            self.Lambda = from_numpy(model_config["Lambda"], device=self.A_T.device)
        else:
            self.Lambda = torch.eye(self.nonlin_size)

        # Counts the number of times project is called.
        # It is assumed that project is called everytime the model parameters are updated.
        self.update_count = 0

        # The number of updates to wait before doing the first projection.
        self.project_delay = (
            model_config["project_delay"] if "project_delay" in model_config else 10
        )
        # The number of updates to wait after a projection before doing the next one.
        self.project_spacing = (
            model_config["project_spacing"] if "project_spacing" in model_config else 5
        )

        self.oldtheta = None

        self.project()

    def project(self):
        """Modify parameters to ensure existence and uniqueness of solution to implicit equation."""
        with torch.no_grad():
            # fmt: off
            controller_params = ControllerThetaParameters(
                self.A_T.t(), self.Bw_T.t(), self.By_T.t(),
                self.Cv_T.t(), self.Dvw_T.t(), self.Dvy_T.t(),
                self.Cu_T.t(), self.Duw_T.t(), self.Duy_T.t(),
                self.Lambda
            )
            # fmt: on
            is_dissipative, newP, newLambda = self.theta_projector.is_dissipative(
                controller_params.torch_to_np(), to_numpy(self.P)
            )
            print(f"Is dissipative: {is_dissipative}")
            if is_dissipative:
                self.P = from_numpy(newP)
                self.P = 0.5 * (self.P + self.P.t())
                self.Lambda = from_numpy(newLambda)
                self.Lambda = 0.5 * (self.Lambda + self.Lambda.t())
            else:
                first_delay_complete = self.update_count == self.project_delay
                delay_interval_complete = (
                    self.update_count > self.project_delay
                    and (self.update_count - self.project_delay) % self.project_spacing == 0
                )

                if first_delay_complete or delay_interval_complete:
                    print("Projection")
                    self.enforce_dissipativity()
                else:
                    print("Not Projection (just inf norm normalization)")
                    # Modify parameters to ensure existence and uniqueness of solution to implicit equation.
                    # Row sum of Dvw is column sum of Dvw_T
                    max_abs_row_sum = torch.linalg.matrix_norm(self.Dvw_T, ord=1)
                    if max_abs_row_sum > 0.999:
                        self.Dvw_T = nn.Parameter(self.Dvw_T * 0.99 / max_abs_row_sum)
                        new_max_abs_row_sum = torch.linalg.matrix_norm(self.Dvw_T, ord=1)
                        print(
                            f"Reducing max abs row sum: {max_abs_row_sum} -> {new_max_abs_row_sum}"
                        )

        self.update_count += 1

        # fmt: off
        theta = torch.vstack((
            torch.hstack((self.A_T.t(),  self.Bw_T.t(), self.By_T.t())),
            torch.hstack((self.Cv_T.t(), self.Dvw_T.t(), self.Dvy_T.t())),
            torch.hstack((self.Cu_T.t(), self.Duw_T.t(), self.Duy_T.t()))
        ))
        # fmt: on
        print_norms(self.A_T.t(), "Ak   ")
        print_norms(self.Bw_T.t(), "Bkw  ")
        print_norms(self.By_T.t(), "Bky  ")
        print_norms(self.Cv_T.t(), "Ckv  ")
        print_norms(self.Dvw_T.t(), "Dkvw ")
        print_norms(self.Dvy_T.t(), "Dkvy ")
        print_norms(self.Cu_T.t(), "Cku  ")
        print_norms(self.Duw_T.t(), "Dkuw ")
        print_norms(self.Duy_T.t(), "Dkuy ")
        print_norms(theta, "theta")
        if self.oldtheta is not None:
            print_norms(
                theta - self.oldtheta,
                f"theta - oldtheta: {torch.allclose(theta, self.oldtheta)}",
            )
        self.oldtheta = theta.detach().clone()

    def enforce_dissipativity(self):
        """Checks if current thetahat parameters correspond to a controller
        which makes the closed loop dissipative. If yes, no change is made.
        If no, the module (_bar, etc.) parameters are modified such that they do.

        Returns whether the module parameters have been modified."""

        controller_params = self.construct_thetahat()
        np_controller_params = controller_params.torch_to_np()

        np_new_controller_params = self.thetahat_projector.project(np_controller_params)
        new_thetahat = np_new_controller_params.np_to_torch(self.A_T.device)

        new_theta, P = self.construct_theta(new_thetahat)

        self.P = P
        self.Lambda = new_theta.Lambda

        missing, unexpected = self.load_state_dict(
            {
                "A_T": new_theta.Ak.t(),
                "Bw_T": new_theta.Bkw.t(),
                "By_T": new_theta.Bky.t(),
                "Cv_T": new_theta.Ckv.t(),
                "Dvw_T": new_theta.Dkvw.t(),
                "Dvy_T": new_theta.Dkvy.t(),
                "Cu_T": new_theta.Cku.t(),
                "Duw_T": new_theta.Dkuw.t(),
                "Duy_T": new_theta.Dkuy.t(),
            },
            strict=False,
        )
        assert unexpected == [], f"Loading unexpected key after projection: {unexpected}"
        # fmt: off
        assert missing == [
            "log_stds", "value.0.bias", "value.0.weight_g", "value.0.weight_v",
            "value.2.bias", "value.2.weight_g", "value.2.weight_v",
            "value.4.bias", "value.4.weight_g", "value.4.weight_v",
        ], missing
        # fmt: on

    def construct_thetahat(self):
        """Constructs thetahat from RINN parameters, P, and Lambda."""
        Duw = self.Duw_T.t()

        S = self.P[: self.state_size, : self.state_size]
        S = 0.5 * (S + S.t())
        U = self.P[: self.state_size, self.state_size :]
        # if (not torch.allclose(S, S.t())) and (S - S.t()).abs().max() < self.eps:
        #     print(
        #         f"S: {torch.allclose(S, S.t())}, {(S - S.t()).abs().max()}, {torch.linalg.eigvalsh(S).min()}"
        #     )
        #     S = (S + S.t()) / 2.0
        #     print(
        #         f"S: {torch.allclose(S, S.t())}, {(S - S.t()).abs().max()}, {torch.linalg.eigvalsh(S).min()}"
        #     )
        assert torch.allclose(S, S.t()), "S is not symmetric"
        assert (
            torch.linalg.eigvalsh(S).min() > self.eps
        ), f"S min eigval is {torch.linalg.eigvalsh(S).min()}"

        # Pinv = self.P.inverse()
        # R = Pinv[:self.state_size, :self.state_size]
        # V = Pinv[:self.state_size, self.state_size:]
        # if (not torch.allclose(R, R.t())) and  (R - R.t()).abs().max() < self.eps:
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        #     R = (R + R.t())/2.0
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        # assert torch.allclose(R, R.t()), "R is not symmetric"
        # assert torch.linalg.eigvalsh(R).min() > self.eps, f"P min eigval is {torch.linalg.eigvalsh(R).min()}"
        # Solve P Y = Y2 for Y
        # fmt: off
        Y2 = torch.vstack((
            torch.hstack((torch.eye(S.shape[0], device=S.device), S)),
            torch.hstack((S.new_zeros((U.t().shape[0], S.shape[0])), U.t()))
        ))
        # fmt: on
        Y = torch.linalg.solve(self.P, Y2)
        R = Y[: self.state_size, : self.state_size]
        R = 0.5 * (R + R.t())
        V = Y[self.state_size :, : self.state_size].t()
        # fmt: off
        # if (not torch.allclose(R, R.t())) and  (R - R.t()).abs().max() < self.eps:
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        #     R = (R + R.t())/2.0
        #     print(f"R: {torch.allclose(R, R.t())}, {(R - R.t()).abs().max()}, {torch.linalg.eigvalsh(R).min()}")
        # fmt: on
        assert torch.allclose(R, R.t()), "R is not symmetric"
        assert (
            torch.linalg.eigvalsh(R).min() > self.eps
        ), f"P min eigval is {torch.linalg.eigvalsh(R).min()}"

        Lambda = self.Lambda
        Lambda = 0.5 * (Lambda + Lambda.t())

        # NA = NA1 + NA2 NA3 NA4

        Ap = self.plant_params.Ap
        Bpu = self.plant_params.Bpu
        Cpy = self.plant_params.Cpy

        NA111 = torch.mm(S, torch.mm(Ap, R))
        NA112 = V.new_zeros((self.state_size, Cpy.shape[0]))
        NA121 = V.new_zeros((Bpu.shape[1], self.state_size))
        NA122 = V.new_zeros((Bpu.shape[1], Cpy.shape[0]))
        NA1 = torch.vstack((torch.hstack((NA111, NA112)), torch.hstack((NA121, NA122))))

        NA211 = U
        NA212 = torch.mm(S, Bpu)
        NA221 = V.new_zeros((Bpu.shape[1], U.shape[1]))
        NA222 = torch.eye(Bpu.shape[1], device=V.device)
        NA2 = torch.vstack((torch.hstack((NA211, NA212)), torch.hstack((NA221, NA222))))

        # fmt: off
        NA3 = torch.vstack((
            torch.hstack((self.A_T.t(), self.By_T.t())), 
            torch.hstack((self.Cu_T.t(), self.Duy_T.t()))
        ))
        # fmt: on

        NA411 = V.t()
        NA412 = V.new_zeros(V.shape[1], Cpy.shape[0])
        NA421 = torch.mm(Cpy, R)
        NA422 = torch.eye(Cpy.shape[0], device=V.device)
        NA4 = torch.vstack((torch.hstack((NA411, NA412)), torch.hstack((NA421, NA422))))

        NA = NA1 + torch.mm(NA2, torch.mm(NA3, NA4))
        NA11 = NA[: self.state_size, : self.state_size]
        NA12 = NA[: self.state_size, self.state_size :]
        NA21 = NA[self.state_size :, : self.state_size]
        NA22 = NA[self.state_size :, self.state_size :]

        NB = torch.mm(S, torch.mm(Bpu, self.Duw_T.t())) + torch.mm(U, self.Bw_T.t())
        # fmt: off
        NC = torch.mm(Lambda, torch.mm(self.Dvy_T.t(), torch.mm(Cpy, R))) \
            + torch.mm(Lambda, torch.mm(self.Cv_T.t(), V.t()))
        # fmt: on

        Dvyhat = torch.mm(Lambda, self.Dvy_T.t())
        Dvwhat = torch.mm(Lambda, self.Dvw_T.t())

        return ControllerThetahatParameters(
            S, R, NA11, NA12, NA21, NA22, NB, NC, Duw, Dvyhat, Dvwhat, Lambda
        )

    def construct_theta(self, thetahat: ControllerThetahatParameters):
        """Construct theta, the parameters of the controller, from thetahat,
        the decision variables for the dissipativity condition."""

        S = thetahat.S
        R = thetahat.R
        NA11 = thetahat.NA11
        NA12 = thetahat.NA12
        NA21 = thetahat.NA21
        NA22 = thetahat.NA22
        NB = thetahat.NB
        NC = thetahat.NC
        Duw_T = thetahat.Dkuw.t()
        Dvyhat = thetahat.Dkvyhat
        Dvwhat = thetahat.Dkvwhat
        Lambda = thetahat.Lambda

        # torch.linalg.solve(A, B) solves for X in AX = B, and assumes A is invertible

        # Construct V and U by solving VU^T = I - RS
        # V = R
        # U = torch.linalg.solve(
        #     V, torch.eye(self.state_size, device=V.device) - torch.mm(R, S)
        # ).t()

        svdU, svdSigma, svdV_T = torch.linalg.svd(
            torch.eye(self.state_size, device=R.device) - torch.mm(R, S)
        )
        sqrt_svdSigma = svdSigma.sqrt().diag()
        V = torch.mm(svdU, sqrt_svdSigma)
        U = torch.mm(svdV_T.t(), sqrt_svdSigma)

        try:
            V.inverse()
        except Exception as e:
            assert False, f"V not invertible: {e.message}"
        try:
            U.inverse()
        except Exception as e:
            assert False, f"U not invertible: {e.message}"

        # Construct P via P = [I, S; 0, U^T] Y^-1
        # fmt: off
        Y = torch.vstack((
            torch.hstack((R, torch.eye(R.shape[0], device=R.device))),
            torch.hstack((V.t(), V.new_zeros((V.t().shape[0], R.shape[0]))))
        ))
        Y2 = torch.vstack((
            torch.hstack((torch.eye(S.shape[0], device=S.device), S)),
            torch.hstack((V.new_zeros((U.t().shape[0], S.shape[0])), U.t()))
        ))
        # fmt: on
        P = torch.linalg.solve(Y.t(), Y2.t())
        P = 0.5 * (P + P.t())
        # if (not torch.allclose(P, P.t())) and (P - P.t()).abs().max() < self.eps:
        #     print(
        #         f"P: {torch.allclose(P, P.t())}, {(P - P.t()).abs().max()}, {torch.linalg.eigvalsh(P).min()}"
        #     )
        #     P = (P + P.t()) / 2.0
        #     print(
        #         f"P: {torch.allclose(P, P.t())}, {(P- P.t()).abs().max()}, {torch.linalg.eigvalsh(P).min()}"
        #     )
        assert torch.allclose(
            P, P.t()
        ), f"P is not even symmetric: max abs error: {(P - P.t()).abs().max()}"
        assert (
            torch.linalg.eigvalsh(P).min() > 0
        ), f"P min eigval is {torch.linalg.eigvalsh(P).min()}"

        # Reconstruct Dvy and Dvw from Dvyhat and Dvwhat
        Dvy = torch.linalg.solve(Lambda, Dvyhat)
        Dvw = torch.linalg.solve(Lambda, Dvwhat)

        # Solve for A, By, Cu, and Duy from NA
        # NA = NA1 + NA2 NA3 NA4
        # NA = torch.vstack((torch.hstack((NA11, NA12)), torch.hstack((NA21, NA22))))

        # NA111 = torch.mm(S, torch.mm(self.Ap, R))
        # NA112 = V.new_zeros((self.state_size, self.Cpy.shape[0]))
        # NA121 = V.new_zeros((self.Bpu.shape[1], self.state_size))
        # NA122 = V.new_zeros((self.Bpu.shape[1], self.Cpy.shape[0]))
        # NA1 = torch.vstack((torch.hstack((NA111, NA112)), torch.hstack((NA121, NA122))))

        # NA211 = U
        # NA212 = torch.mm(S, self.Bpu)
        # NA221 = V.new_zeros((self.Bpu.shape[1], U.shape[1]))
        # NA222 = torch.eye(self.Bpu.shape[1], device=V.device)
        # NA2 = torch.vstack((torch.hstack((NA211, NA212)), torch.hstack((NA221, NA222))))

        # NA411 = V.t()
        # NA412 = V.new_zeros(V.shape[1], self.Cpy.shape[0])
        # NA421 = torch.mm(self.Cpy, R)
        # NA422 = torch.eye(self.Cpy.shape[0], device=V.device)
        # NA4 = torch.vstack((torch.hstack((NA411, NA412)), torch.hstack((NA421, NA422))))

        # NA3NA4 = torch.linalg.solve(NA2, NA - NA1)
        # NA3 = torch.linalg.solve(NA4.t(), NA3NA4.t()).t()

        # A = NA3[: self.state_size, : self.state_size]
        # By = NA3[: self.state_size, self.state_size :]
        # Cu = NA3[self.state_size :, : self.state_size]
        # Duy = NA3[self.state_size :, self.state_size :]

        Ap = self.plant_params.Ap
        Bpu = self.plant_params.Bpu
        Cpy = self.plant_params.Cpy

        Duy = NA22
        By = torch.linalg.solve(U, NA12 - S @ Bpu @ Duy)
        Cu = torch.linalg.solve(V, NA21.t() - R @ Cpy.t() @ Duy.t()).t()
        AVT = torch.linalg.solve(
            U,
            NA11 - U @ By @ Cpy @ R - S @ Bpu @ (Cu @ V.t() + Duy @ Cpy @ R) - S @ Ap @ R,
        )
        A = torch.linalg.solve(V, AVT.t()).t()

        # Solve for Bw from NB
        # NB = NB1 + U Bw
        NB1 = torch.mm(S, torch.mm(Bpu, Duw_T.t()))
        Bw = torch.linalg.solve(U, NB - NB1)

        # Solve for Cv from NC
        # NC = NC1 + Lambda Cv V^T
        NC1 = torch.mm(Dvyhat, torch.mm(Cpy, R))
        CvVT = torch.linalg.solve(Lambda, NC - NC1)
        Cv = torch.linalg.solve(V, CvVT.t()).t()

        # Bring together the theta parameters here
        controller_params = ControllerThetaParameters(
            A, Bw, By, Cv, Dvw, Dvy, Cu, Duw_T.t(), Duy, Lambda
        )
        return controller_params, P

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
 