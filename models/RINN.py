import torch
import torch.nn as nn
import torchdeq
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from torchdeq.norm import apply_norm, reset_norm

from activations import activations_map
from utils import build_mlp, uniform


class RINN(RecurrentNetwork, nn.Module):
    """
    A recurrent implicit neural network of the following form: TODO(Neelay)

    Train with a method that calls project after each gradient step.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        model_config = model_config["custom_model_config"]

        assert (
            2 * action_space.shape[0] == num_outputs
        ), "Num outputs should be 2 * action dimension"

        self.state_size = (
            model_config["state_size"] if "state_size" in model_config else 16
        )
        self.nonlin_size = (
            model_config["nonlin_size"] if "nonlin_size" in model_config else 128
        )
        self.input_size = obs_space.shape[0]
        self.output_size = action_space.shape[0]

        if "delta" not in model_config:
            model_config["delta"] = "tanh"
        if model_config["delta"] not in activations_map:
            raise ValueError(
                f"Activation {model_config['delta']} is not in activations.activations_map."
            )
        self.delta = activations_map[model_config["delta"]]()

        # _T for transpose
        # State x, nonlinearity Delta output w, input y,
        # nonlineary Delta input v, output u,
        # and nonlinearity Delta.
        self.A_T = nn.Parameter(uniform(self.state_size, self.state_size))
        self.Bw_T = nn.Parameter(uniform(self.nonlin_size, self.state_size))
        self.By_T = nn.Parameter(uniform(self.input_size, self.state_size))

        self.Cv_T = nn.Parameter(uniform(self.state_size, self.nonlin_size))
        self.Dvw_T = nn.Parameter(uniform(self.nonlin_size, self.nonlin_size))
        self.Dvy_T = nn.Parameter(uniform(self.input_size, self.nonlin_size))

        self.Cu_T = nn.Parameter(uniform(self.state_size, self.output_size))
        self.Duw_T = nn.Parameter(uniform(self.nonlin_size, self.output_size))
        self.Duy_T = nn.Parameter(uniform(self.input_size, self.output_size))

        # self.deq = torchdeq.get_deq(f_solver="broyden", f_max_iter=30, b_max_iter=30)
        self.deq = torchdeq.get_deq(
            f_solver="fixed_point_iter", f_max_iter=30, b_max_iter=30
        )

        log_std_init = (
            model_config["log_std_init"]
            if "log_std_init" in model_config
            else -1.6094379124341003  # log(0.2)
        )
        self.log_stds = nn.Parameter(log_std_init * torch.ones(self.output_size))

        assert "dt" in model_config
        self.dt = model_config["dt"]

        self.value = build_mlp(
            input_size=obs_space.shape[0],
            output_size=1,
            n_layers=model_config["baseline_n_layers"]
            if "baseline_n_layers" in model_config
            else 2,
            size=model_config["baseline_size"]
            if "baseline_size" in model_config
            else 64,
        )
        self._cur_value = None

        apply_norm(self, filter_out=["A_T", "Bw_T", "By_T", "Cu_T", "Duw_T", "Duy_T"])

        self.project()

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
        # Compute the next state using the Runge-Kutta 4th-order method,
        # assuming y is constant over the time step.
        k1, w = self.xdot(x, y, w=w)
        k2, w = self.xdot(x + self.dt * k1 / 2, y, w0=w)
        k3, w = self.xdot(x + self.dt * k2 / 2, y, w0=w)
        k4, w = self.xdot(x + self.dt * k3, y, w0=w)
        next_x = x + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_x, w

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

        assert not torch.any(torch.isnan(xkp1))

        for k in range(time_len):
            # Set action for time k

            xk = xkp1
            yk = obs[:, k]

            # Solve for wk
            wk0 = wkp10

            def delta_tilde(w):
                v = xk @ self.Cv_T + w @ self.Dvw_T + yk @ self.Dvy_T
                return self.delta(v)

            solver_kwargs = {"f_max_iter": 5} if reuse else {}
            wk, info = self.deq(delta_tilde, wk0, solver_kwargs=solver_kwargs)
            assert len(wk) > 0
            wk = wk[-1]

            uk = xk @ self.Cu_T + wk @ self.Duw_T + yk @ self.Duy_T
            actions[:, k] = uk

            # Compute next state xkp1
            # wkp10 is the initial guess for w[k+1]
            xkp1, wkp10 = self.next_state(xk, yk, w=wk)
            reuse = True

        log_stds_rep = self.log_stds.repeat((batch_size, time_len, 1))
        outputs = torch.cat((actions, log_stds_rep), dim=2)

        self._cur_value = self.value(obs)
        self._cur_value = self._cur_value.reshape([-1])

        return outputs, [xkp1]

    def project(self):
        # Modify parameters to ensure existence and uniqueness of solution to implicit equation.
        with torch.no_grad():
            # Row sum of Dvw is column sum of Dvw_T
            max_abs_row_sum = torch.linalg.matrix_norm(self.Dvw_T, ord=1)
            if max_abs_row_sum > 0.999:
                self.Dvw_T = nn.Parameter(self.Dvw_T * 0.99 / max_abs_row_sum)
                new_max_abs_row_sum = torch.linalg.matrix_norm(self.Dvw_T, ord=1)
                print(
                    f"Reducing max abs row sum: {max_abs_row_sum} -> {new_max_abs_row_sum}"
                )
            # max_gain = 1.0
            # missing, unexpected = self.load_state_dict(
            #     {
            #         "A_T": torch.clamp(self.A_T, min=-max_gain, max=max_gain),
            #         "Bw_T": torch.clamp(self.Bw_T, min=-max_gain, max=max_gain),
            #         "By_T": torch.clamp(self.By_T, min=-max_gain, max=max_gain),
            #         "Cv_T": torch.clamp(self.Cv_T, min=-max_gain, max=max_gain),
            #         "Dvw_T": torch.clamp(self.Dvw_T, min=-max_gain, max=max_gain),
            #         "Dvy_T": torch.clamp(self.Dvy_T, min=-max_gain, max=max_gain),
            #         "Cu_T": torch.clamp(self.Cu_T, min=-max_gain, max=max_gain),
            #         "Duw_T": torch.clamp(self.Duw_T, min=-max_gain, max=max_gain),
            #         "Duy_T": torch.clamp(self.Duy_T, min=-max_gain, max=max_gain),
            #     },
            #     strict=False,
            # )

        print(torch.max(torch.tensor([x.abs().max() for x in [self.A_T, self.Bw_T, self.By_T, 
                                                            self.Cv_T, self.Dvw_T, self.Dvy_T, 
                                                            self.Cu_T, self.Duw_T, self.Duy_T]])))