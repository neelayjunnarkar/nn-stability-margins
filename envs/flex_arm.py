import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from variable_structs import PlantParameters


class FlexibleArmEnv(gym.Env):
    """
    Flexible arm on a cart model.
    """

    def __init__(self, env_config):
        assert "observation" in env_config
        self.observation = env_config["observation"]

        assert "normed" in env_config
        normed = env_config["normed"]

        assert "disturbance_model" in env_config
        self.disturbance_model = env_config["disturbance_model"]

        assert "design_model" in env_config
        design_model = env_config["design_model"]

        if "disturbance_design_model" in env_config:
            self.disturbance_design_model = env_config["disturbance_design_model"]
        else:
            self.disturbance_design_model = self.disturbance_model

        assert "supply_rate" in env_config
        self.supply_rate = env_config["supply_rate"]

        if "dt" in env_config:  # Discretization time (s)
            self.dt = env_config["dt"]
        else:
            self.dt = 0.0001

        if "rollout_length" in env_config:
            self.time_max = env_config["rollout_length"]
        else:
            self.time_max = 1000  # 0.1 seconds with dt = 0.0001

        # Used to scale robustness in rigidplus, with MDeltap = [alpha, 0; 0, -1]
        # Only used in rigidplus
        if "delta_alpha" in env_config:
            delta_alpha = env_config["delta_alpha"]
        else:
            delta_alpha = 1.0

        # Used to scale supply rate in rigidplus
        # Only used in rigidplus
        if "supplyrate_scale" in env_config:
            supplyrate_scale = env_config["supplyrate_scale"]
        else:
            supplyrate_scale = 1.0

        self.mb = 1  # Mass of base (Kg)
        self.mt = 0.1  # Mass of tip (Kg)
        self.L = 1  # Length of link (m)
        self.rho = 0.1  # Mass density of Link (N/m)

        self.r = 1e-2  # Radius of rod cross-section (m)
        self.E = 200e9  # Young's modulus for steel, GPa
        self.I = (np.pi / 4) * self.r**4  # Area second moment of inertia (m^4)
        self.EI = self.E * self.I
        self.flexdamp = 0.9
        self.Mr = self.mb + self.mt + self.rho * self.L  # Total mass (Kg)

        self.max_force = 20  # 200
        self.max_x = 1.5
        self.max_h = 0.66 * self.L
        self.max_xdot = 25
        self.max_hdot = 200

        # State xp is (theta, thetadot) where theta is angle from vertical,
        # with theta=0 being inverted, and thetadot is derivative of theta.
        #
        # Continuous-time parameters for model:
        # xpdot(t) = Ap  xp(t) + Bpw  wp(t) + Bpd d(t)  + Bpu u(t)
        # vp(t)    = Cpv xp(t) + Dpvw wp(t) + Dpvd d(t) + Dpvu u(t)
        # e(t)     = Cpe xp(t) + Dpew wp(t) + Dped d(t) + Dpeu u(t)
        # y(t)     = Cpy xp(t) + Dpyw wp(t) + Dpyd d(t) + 0
        # wp(t)       = Deltap(vp(t))
        #
        # where xp is state, wp is output of uncertainty Deltap, d is disturbance,
        # u is control input, vp is input to uncertainty Deltap, e is performance output,
        # y is measurement output, and Deltap is the uncertainty.
        # Suffix p indicates plant parameter.

        # States are (x, h, xdot, hdot)
        # x is position of base of rod on cart
        # h is horizontal deviation of tip of rod from the base of the rod
        # fmt: off

        self.true_model = self._build_true_model(self.observation, design_model, self.disturbance_model, self.supply_rate)

        if design_model == "flexible":
            self.design_model = self._build_true_model(self.observation, "flexible", self.disturbance_design_model, self.supply_rate)
        elif design_model == "rigid":
            self.design_model = self._build_rigid_design_model(self.observation, self.disturbance_design_model, self.supply_rate)
        elif design_model == "rigidplus":
            self.design_model = self._build_rigidplus_design_model(self.observation, self.disturbance_design_model, self.supply_rate, delta_alpha, supplyrate_scale)
        else:
            raise ValueError(f"Unexpected design model: {design_model}.")

        self.Ap = self.true_model.Ap
        self.Bpw = self.true_model.Bpw
        self.Bpd = self.true_model.Bpd
        self.Bpu = self.true_model.Bpu
        self.Cpv = self.true_model.Cpv
        self.Dpvw = self.true_model.Dpvw
        self.Dpvd = self.true_model.Dpvd
        self.Dpvu = self.true_model.Dpvu
        self.Cpe = self.true_model.Cpe
        self.Dpew = self.true_model.Dpew
        self.Dped = self.true_model.Dped
        self.Dpeu = self.true_model.Dpeu
        self.Cpy = self.true_model.Cpy
        self.Dpyw = self.true_model.Dpyw
        self.Dpyd = self.true_model.Dpyd
        self.MDeltapvv = self.true_model.MDeltapvv
        self.MDeltapvw = self.true_model.MDeltapvw
        self.MDeltapww = self.true_model.MDeltapww
        self.Xdd = self.true_model.Xdd
        self.Xde = self.true_model.Xde
        self.Xee = self.true_model.Xee
        # Make sure dimensions of parameters match up.
        self.nx = self.Ap.shape[0]
        self.nw = self.Bpw.shape[1]
        self.nd = self.Bpd.shape[1]
        self.nu = self.Bpu.shape[1]
        self.nv = self.Cpv.shape[0]
        self.ne = self.Cpe.shape[0]
        self.ny = self.Cpy.shape[0]
        self.check_parameter_sizes()

        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(self.nu,),
            dtype=np.float32,
        )
        x_max = np.array([self.max_x, self.max_h, self.max_xdot, self.max_hdot], dtype=np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

        if normed:
            self.Cpy = self.Cpy / self.state_space.high
        ob_max = self.Cpy @ x_max
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)


        self.state_size = self.nx
        # TODO(Neelay): this nonlin_size parameter likely only works when nv = nw. Fix.
        self.nonlin_size = self.nv

        # self.max_reward = 2.0
        self.max_reward = 1.0

        self.seed(env_config["seed"] if "seed" in env_config else None)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def Deltap(self, vp):
        wp = np.zeros((self.nw,), dtype=np.float32)
        return wp

    def xdot(self, state, d, u):
        # In this case, vp doesnt depend on wp so can calculate both easily.
        vp = self.Cpv @ state + self.Dpvd @ d + self.Dpvu @ u
        wp = self.Deltap(vp)
        xdot = self.Ap @ state + self.Bpw @ wp + self.Bpd @ d + self.Bpu @ u
        return xdot

    def next_state(self, state, d, u):
        # Runge-Kutta 4th order
        # Assume d and u are constant
        # More accurate than the 0th order hold
        k1 = self.xdot(state, d, u)
        k2 = self.xdot(state + self.dt * k1 / 2, d, u)
        k3 = self.xdot(state + self.dt * k2 / 2, d, u)
        k4 = self.xdot(state + self.dt * k3, d, u)
        state = state + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return state

    def step(self, u, fail_on_state_space=True, fail_on_time_limit=True):
        u = np.clip(u, -self.max_force, self.max_force)

        if self.disturbance_model == "none":
            d = np.zeros((self.nd,), dtype=np.float32)
        elif self.disturbance_model == "occasional":
            # Have disturbance ocurr (in expectation) twice a second.
            p = 2 * self.dt
            rand = self.np_random.uniform()
            if rand < p / 2:
                d = 1 * self.action_space.low.astype(np.float32)
            elif rand < p:
                d = 1 * self.action_space.high.astype(np.float32)
            else:
                d = np.zeros((self.nd,), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected disturbance model: {self.disturbance_model}.")

        self.state = self.next_state(self.state, d, u)

        # Reward for small state and small control
        reward_state = np.exp(-(np.linalg.norm(self.state) ** 2))
        reward_control = np.exp(-(np.linalg.norm(u) ** 2))
        reward = reward_control

        # reward_state = np.linalg.norm(self.state_space.high)**2 - np.linalg.norm(self.state)**2
        # reward_state = 0.5 * reward_state / (np.linalg.norm(self.state_space.high)**2)
        # reward_control = np.linalg.norm(self.action_space.high)**2 - np.linalg.norm(u)**2
        # reward_control = 0.5 * reward_control / (np.linalg.norm(self.action_space.high)**2)
        # reward = reward_state + reward_control
        # print(
        #     f"State norm: {np.linalg.norm(self.state)}, control norm: {np.linalg.norm(u)}, reward state: {reward_state}, reward control: {reward_control}"
        # )
        # reward = reward_control

        terminated = False
        if fail_on_time_limit and self.time >= self.time_max:
            # print(f"Exceeded time limit {self.time_max}")
            terminated = True
        if fail_on_state_space and self.state not in self.state_space:
            # print(f"Failed state space constraint: {self.state} not it {self.state_space}")
            terminated = True

        self.time += 1

        return self.get_obs(), reward, terminated, {}

    def reset(self, state=None):
        if state is None:
            high = np.array(
                [
                    0.66 * self.max_x,
                    0.66 * self.max_h,
                    0.01 * self.max_xdot,
                    0.01 * self.max_hdot,
                ],
                dtype=np.float32,
            )
            self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        else:
            self.state = state
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return self.Cpy @ self.state  # TODO(Neelay): generalize

    def get_params(self):
        return self.design_model

    def _build_true_model(self, observation, design_model, disturbance_model, supply_rate):
        nx = 4  # Plant state size
        nw = 1  # Output size of uncertainty Deltap
        nd = None  # Disturbance size. Defined based on disturbance model.
        nu = 1  # Control input size
        nv = 1  # Input size to uncertainty Deltap
        ne = None  # Performance output size. Defined based on supply_rate.
        ny = None  # Measurement output size. Defined based on observation.
        M = np.array(
            [
                [self.Mr, self.mt + self.rho * self.L / 3],
                [self.mt + self.rho * self.L / 3, self.mt + self.rho * self.L / 5],
            ],
            dtype=np.float32,
        )
        K = np.array([[0, 0], [0, 4 * self.EI / (self.L**3)]], dtype=np.float32)
        B = np.diag([0, self.flexdamp]).astype(np.float32)

        # fmt: off
        Ap = np.bmat([
                [np.zeros((2, 2)), np.eye(2)],
                [-np.linalg.solve(M, K), -np.linalg.solve(M, B)],
        ])
        # fmt: on
        Ap = np.asarray(Ap).astype(np.float32)
        Bpw = np.zeros((nx, nw), dtype=np.float32)
        Bpu = np.vstack([np.zeros((2, 1)), np.linalg.solve(M, np.array([[1], [0]]))])
        Bpu = np.asarray(Bpu).astype(np.float32)

        Cpv = np.zeros((nv, nx), dtype=np.float32)
        Dpvw = np.zeros((nv, nw), dtype=np.float32)
        Dpvu = np.zeros((nv, nu), dtype=np.float32)

        assert design_model in ["flexible", "rigid", "rigidplus"]
        if observation == "full" and design_model == "flexible":
            # Observe full state
            ny = nx
            Cpy = np.eye(nx, dtype=np.float32)
        elif observation == "full" and (design_model == "rigid" or design_model == "rigidplus"):
            # Observe "full state" of rigid model, with sensors at top of pendulum (gives x + h, xdot + hdot)
            ny = 2
            Cpy = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
        elif observation == "partial" and design_model == "flexible":
            # Observe x and h
            ny = 2
            Cpy = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        elif observation == "partial" and (design_model == "rigid" or design_model == "rigidplus"):
            # Observe x + h
            ny = 1
            Cpy = np.array([[1, 1, 0, 0]], dtype=np.float32)
        else:
            raise ValueError(f"observation {observation} must be one of 'partial', 'full'")
        Dpyw = np.zeros((ny, nw), dtype=np.float32)
        # Dpyu is always 0

        if disturbance_model == "none":
            nd = 1
            Bpd = np.zeros((nx, nd), dtype=np.float32)
            Dpvd = np.zeros((nv, nd), dtype=np.float32)
            Dpyd = np.zeros((ny, nd), dtype=np.float32)
        elif disturbance_model == "occasional":
            nd = nu
            Bpd = Bpu.copy()
            Dpvd = np.zeros((nv, nd), dtype=np.float32)
            Dpyd = np.zeros((ny, nd), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected disturbance model: {disturbance_model}.")

        if supply_rate == "stability":
            ne = nx
            Cpe = np.eye(nx, dtype=np.float32)
            Dpew = np.zeros((ne, nw), dtype=np.float32)
            Dped = np.zeros((ne, nd), dtype=np.float32)
            Dpeu = np.zeros((ne, nu), dtype=np.float32)

            Xdd = np.zeros((nd, nd), dtype=np.float32)
            Xde = np.zeros((nd, ne), dtype=np.float32)
            Xee = np.zeros((ne, ne), dtype=np.float32)
        elif supply_rate == "l2_gain":
            ne = nx
            Cpe = np.eye(nx, dtype=np.float32)
            Dpew = np.zeros((ne, nw), dtype=np.float32)
            Dped = np.zeros((ne, nd), dtype=np.float32)
            Dpeu = np.zeros((ne, nu), dtype=np.float32)

            gamma = 0.99
            Xdd = gamma * np.eye(nd, dtype=np.float32)
            Xde = np.zeros((nd, ne), dtype=np.float32)
            Xee = -np.eye(ne, dtype=np.float32)
        else:
            raise ValueError(f"Supply rate {supply_rate} must be one of: 'stability', 'l2_gain'.")

        MDeltapvv = np.zeros((nv, nv), dtype=np.float32)
        MDeltapvw = np.zeros((nv, nw), dtype=np.float32)
        MDeltapww = np.zeros((nw, nw), dtype=np.float32)

        # fmt: off
        return PlantParameters(
            Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu, Cpe, Dpew, Dped, Dpeu, Cpy, Dpyw, Dpyd,
            MDeltapvv, MDeltapvw, MDeltapww, Xdd, Xde, Xee
        )
        # fmt: on

    def _build_rigid_design_model(self, observation, disturbance_model, supply_rate):
        nx = 2
        nw = 1  # Output size of uncertainty Deltap
        nd = None  # Disturbance size. Defined based on disturbance_model.
        nu = 1  # Control input size
        nv = 1  # Input size to uncertainty Deltap
        ne = None  # Performance output size. Defined based on supply_rate.
        ny = None  # Measurement output size. Defined based on observation.

        Ap = np.array([[0, 1], [0, 0]], dtype=np.float32)
        Bpw = np.zeros((nx, nw), dtype=np.float32)
        Bpu = np.array([[0], [1.0 / self.Mr]], dtype=np.float32)

        Cpv = np.zeros((nv, nx), dtype=np.float32)
        Dpvw = np.zeros((nv, nw), dtype=np.float32)
        Dpvu = np.zeros((nv, nu), dtype=np.float32)

        if observation == "full":  # Observe position and velocity
            ny = 2
            Cpy = np.eye(2, dtype=np.float32)
        elif observation == "partial":  # Observe only position
            ny = 1
            Cpy = np.array([[1, 0]], dtype=np.float32)
        else:
            raise ValueError(f"observation {observation} must be one of 'partial', 'full'")
        Dpyw = np.zeros((ny, nw), dtype=np.float32)
        # Dpyu is always 0

        if disturbance_model == "none":
            nd = 1
            Bpd = np.zeros((nx, nd), dtype=np.float32)
            Dpvd = np.zeros((nv, nd), dtype=np.float32)
            Dpyd = np.zeros((ny, nd), dtype=np.float32)
        elif disturbance_model == "occasional":
            nd = nu
            Bpd = Bpu.copy()
            Dpvd = np.zeros((nv, nd), dtype=np.float32)
            Dpyd = np.zeros((ny, nd), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected disturbance model: {disturbance_model}.")

        if supply_rate == "stability":
            ne = nx
            Cpe = np.eye(nx, dtype=np.float32)
            Dpew = np.zeros((ne, nw), dtype=np.float32)
            Dped = np.zeros((ne, nd), dtype=np.float32)
            Dpeu = np.zeros((ne, nu), dtype=np.float32)

            Xdd = np.zeros((nd, nd), dtype=np.float32)
            Xde = np.zeros((nd, ne), dtype=np.float32)
            Xee = np.zeros((ne, ne), dtype=np.float32)
        elif supply_rate == "l2_gain":
            ne = nx
            Cpe = np.eye(nx, dtype=np.float32)
            Dpew = np.zeros((ne, nw), dtype=np.float32)
            Dped = np.zeros((ne, nd), dtype=np.float32)
            Dpeu = np.zeros((ne, nu), dtype=np.float32)

            gamma = 0.99
            Xdd = gamma**2 * np.eye(nd, dtype=np.float32)
            Xde = np.zeros((nd, ne), dtype=np.float32)
            Xee = -np.eye(ne, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected supply rate: {supply_rate}")

        MDeltapvv = np.zeros((nv, nv), dtype=np.float32)
        MDeltapvw = np.zeros((nv, nw), dtype=np.float32)
        MDeltapww = np.zeros((nw, nw), dtype=np.float32)

        # fmt: off
        return PlantParameters(
            Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu, Cpe, Dpew, Dped, Dpeu, Cpy, Dpyw, Dpyd,
            MDeltapvv, MDeltapvw, MDeltapww, Xdd, Xde, Xee
        )
        # fmt: on

    def _build_rigidplus_design_model(self, observation, disturbance_model, supply_rate, delta_alpha, supplyrate_scale):
        """
        Build rigid model with additive uncertainty covering the difference between the rigid and flexible models.

             d  |-> Delta -> W -|
             |  |               |
             v  |               v
        u -> + ---------> Gr -> + -> y        
        """

        assert observation == "partial", "This model only supports partial observation"
        assert disturbance_model == "occasional", "This model only supports disturbance"
        assert supply_rate == "l2_gain", "This model only supports a supply rate for L2 gain"

        # Combination of Gr and W:

        # # Using balanced realization of interconnection with
        # #        3.336e-10 s^2 + 2.489e-05 s + 0.9412
        # # W(s) = ------------------------------------
        # #              s^2 + 8.567 s + 5.974e04
        # Ap = np.array(
        #     [[0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, -4.2236, 244.4], [0, 0, -244.4, -4.331]],
        #     dtype=np.float32,
        # )
        # Bpw = np.array([[0], [0], [-0.04402], [-0.04373]], dtype=np.float32)
        # Bpd = np.array([[0], [1.667], [0], [0]], dtype=np.float32)
        # Bpu = np.array([[0], [1.667], [0], [0]], dtype=np.float32)

        # Cpv = np.zeros((1, 4), dtype=np.float32)
        # Dpvw = np.zeros((1, 1), dtype=np.float32)
        # Dpvd = np.ones((1, 1), dtype=np.float32)
        # Dpvu = np.ones((1, 1), dtype=np.float32)

        # Cpe = np.array([[1, 0, 0, 0], [0, 0.5, 0, 0]], dtype=np.float32)
        # Dpew = np.zeros((2, 1), dtype=np.float32)
        # Dped = np.zeros((2, 1), dtype=np.float32)
        # Dpeu = np.zeros((2, 1), dtype=np.float32)

        # Cpy = np.array([[1, 0, -0.04402, 0.04373]], dtype=np.float32)
        # Dpyw = np.array([[3.336e-10]], dtype=np.float32)
        # Dpyd = np.zeros((1, 1), dtype=np.float32)

        # Using balanced realization of interconnection with
        #                0.9412
        # W(s) = ------------------------
        #        s^2 + 8.567 s + 5.974e04
        Ap = np.array(
            [[0, 0.5, 0, 0], [0, 0, 0, 0], [0, 0, -4.208, 244.4], [0, 0, -244.4, -4.359]],
            dtype=np.float32,
        )
        Bpw = np.array([[0], [0], [-0.04388], [-0.04388]], dtype=np.float32)
        Bpd = np.array([[0], [1.667], [0], [0]], dtype=np.float32)
        Bpu = np.array([[0], [1.667], [0], [0]], dtype=np.float32)

        Cpv = np.zeros((1, 4), dtype=np.float32)
        Dpvw = np.zeros((1, 1), dtype=np.float32)
        Dpvd = np.ones((1, 1), dtype=np.float32)
        Dpvu = np.ones((1, 1), dtype=np.float32)

        Cpe = np.array([[1, 0, 0, 0], [0, 0.5, 0, 0]], dtype=np.float32)
        Dpew = np.zeros((2, 1), dtype=np.float32)
        Dped = np.zeros((2, 1), dtype=np.float32)
        Dpeu = np.zeros((2, 1), dtype=np.float32)

        Cpy = np.array([[1, 0, -0.04388, 0.04388]], dtype=np.float32)
        Dpyw = np.zeros((1, 1), dtype=np.float32)
        Dpyd = np.zeros((1, 1), dtype=np.float32)

        # # Using unbalanced realization of interconnection with
        # #                0.9412
        # # W(s) = ------------------------
        # #        s^2 + 8.567 s + 5.974e04
        # Ap = np.array(
        #     [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, -8.567, -233.4], [0, 0, 256, 0]],
        #     dtype=np.float32,
        # )
        # Bpw = np.array([[0], [0], [0.0625], [0]], dtype=np.float32)
        # Bpd = np.array([[0], [0.8333], [0], [0]], dtype=np.float32)
        # Bpu = np.array([[0], [0.8333], [0], [0]], dtype=np.float32)

        # Cpv = np.zeros((1, 4), dtype=np.float32)
        # Dpvw = np.zeros((1, 1), dtype=np.float32)
        # Dpvd = np.ones((1, 1), dtype=np.float32)
        # Dpvu = np.ones((1, 1), dtype=np.float32)

        # Cpe = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        # Dpew = np.zeros((2, 1), dtype=np.float32)
        # Dped = np.zeros((2, 1), dtype=np.float32)
        # Dpeu = np.zeros((2, 1), dtype=np.float32)

        # Cpy = np.array([[1, 0, 0, 0.05883]], dtype=np.float32)
        # Dpyw = np.zeros((1, 1), dtype=np.float32)
        # Dpyd = np.zeros((1, 1), dtype=np.float32)

        assert delta_alpha >= 0.0 and delta_alpha <= 1.0
        mdelta_scale = 1
        MDeltapvv = mdelta_scale * delta_alpha * np.array([[1]], dtype=np.float32)
        MDeltapvw = np.array([[0]], dtype=np.float32)
        MDeltapww = mdelta_scale * np.array([[-1]], dtype=np.float32)

        gamma = 0.99 # L2 gain
        alpha = supplyrate_scale # 1.6 # Scale supply rate for better numerical results
        Xdd = alpha * gamma**2 * np.eye(1, dtype=np.float32)
        Xde = np.zeros((1, 2), dtype=np.float32)
        Xee = alpha * -np.eye(2, dtype=np.float32)

    
        # fmt: off
        return PlantParameters(
            Ap, Bpw, Bpd, Bpu, Cpv, Dpvw, Dpvd, Dpvu, Cpe, Dpew, Dped, Dpeu, Cpy, Dpyw, Dpyd,
            MDeltapvv, MDeltapvw, MDeltapww, Xdd, Xde, Xee
        )
        # fmt: on

    def check_parameter_sizes(self):
        assert (
            self.Ap.shape[0] == self.nx
            and self.Ap.shape[0] == self.Bpw.shape[0] == self.Bpd.shape[0] == self.Bpu.shape[0]
        ), f"{self.Ap.shape}, {self.Bpw.shape}, {self.Bpd.shape}, {self.Bpu.shape}"
        assert (
            self.Cpv.shape[0] == self.nv
            and self.Cpv.shape[0] == self.Dpvw.shape[0] == self.Dpvd.shape[0] == self.Dpvu.shape[0]
        ), f"{self.Cpv.shape}, {self.Dpvw.shape}, {self.Dpvd.shape}, {self.Dpvu.shape}"
        assert (
            self.Cpe.shape[0] == self.ne
            and self.Cpe.shape[0] == self.Dpew.shape[0] == self.Dped.shape[0] == self.Dpeu.shape[0]
        ), f"{self.Cpe.shape}, {self.Dpew.shape}, {self.Dped.shape}, {self.Dpeu.shape}"
        assert (
            self.Cpy.shape[0] == self.ny
            and self.Cpy.shape[0] == self.Dpyw.shape[0] == self.Dpyd.shape[0]
        ), f"{self.Cpy.shape}, {self.Dpyw.shape}, {self.Dpyd.shape}"
        assert (
            self.Ap.shape[1] == self.nx
            and self.Ap.shape[1] == self.Cpv.shape[1] == self.Cpe.shape[1] == self.Cpy.shape[1]
        ), f"{self.Ap.shape}, {self.Cpv.shape}, {self.Cpe.shape}, {self.Cpy.shape}"
        assert (
            self.Bpw.shape[1] == self.nw
            and self.Bpw.shape[1] == self.Dpvw.shape[1] == self.Dpew.shape[1] == self.Dpyw.shape[1]
        ), f"{self.Bpw.shape}, {self.Dpvw.shape}, {self.Dpew.shape}, {self.Dpyw.shape}"
        assert (
            self.Bpd.shape[1] == self.nd
            and self.Bpd.shape[1] == self.Dpvd.shape[1] == self.Dped.shape[1] == self.Dpyd.shape[1]
        ), f"{self.Bpd.shape}, {self.Dvpd.shape}, {self.Dped.shape}, {self.Dpyd.shape}"
        assert (
            self.Bpu.shape[1] == self.nu
            and self.Bpu.shape[1] == self.Dpvu.shape[1] == self.Dpeu.shape[1]
        ), f"{self.Bpu.shape}, {self.Dpvu.shape}, {self.Dpeu.shape}"
