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
        assert "factor" in env_config
        factor = env_config["factor"]

        assert "observation" in env_config
        self.observation = env_config["observation"]

        assert "normed" in env_config
        normed = env_config["normed"]

        assert "disturbance_model" in env_config
        self.disturbance_model = env_config["disturbance_model"]

        assert "design_model" in env_config
        self.design_model = env_config["design_model"]

        self.factor = factor
        self.viewer = None

        if "dt" in env_config:  # Discretization time (s)
            self.dt = env_config["dt"]
        else:
            self.dt = 0.01

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
        M = np.array(
            [
                [self.Mr, self.mt + self.rho * self.L / 3],
                [self.mt + self.rho * self.L / 3, self.mt + self.rho * self.L / 5],
            ],
            dtype=np.float32,
        )
        K = np.array([[0, 0], [0, 4 * self.EI / (self.L**3)]], dtype=np.float32)
        B = np.diag([0, self.flexdamp]).astype(np.float32)

        self.max_force = 200
        self.max_x = 1.5
        self.max_h = 0.66 * self.L
        self.max_xdot = 25
        self.max_hdot = 200
        self.time_max = 1000  # 10 seconds with dt = 0.01

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

        self.nx = 4  # Plant state size
        self.nw = 1  # Output size of uncertainty Deltap
        self.nd = None  # Disturbance size. Defined based on supply_rate.
        self.nu = 1  # Control input size
        self.nv = 1  # Input size to uncertainty Deltap
        self.ne = None  # Performance output size. Defined based on supply_rate.
        self.ny = None  # Measurement output size. Defined later.

        # States are (x, h, xdot, hdot)
        # x is position of base of rod on cart
        # h is horizontal deviation of tip of rod from the base of the rod
        # fmt: off
        self.Ap = np.bmat([
                [np.zeros((2, 2)), np.eye(2)],
                [-np.linalg.solve(M, K), -np.linalg.solve(M, B)],
        ])
        self.Ap = np.asarray(self.Ap).astype(np.float32)
        self.Bpw = np.zeros((self.nx, self.nw), dtype=np.float32)
        self.Bpu = np.vstack([
            np.zeros((2, 1)),
            np.linalg.solve(M, np.array([[1], [0]]))
        ])
        self.Bpu = np.asarray(self.Bpu).astype(np.float32)
        self.Cpv = np.zeros((self.nv, self.nx), dtype=np.float32)
        self.Dpvw = np.zeros((self.nv, self.nw), dtype=np.float32)
        self.Dpvu = np.zeros((self.nv, self.nu), dtype=np.float32)
        # fmt: on

        if self.observation == "full":  # Observe full state
            if self.design_model == "flexible":
                self.ny = self.nx
                self.Cpy = np.eye(self.nx, dtype=np.float32)
            elif self.design_model == "rigid":
                self.ny = 2
                self.Cpy = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.float32)
            else:
                raise ValueError(f"Design model: {self.design_model} unexpected")
        elif self.observation == "partial":  # Observe sum of x and h
            self.ny = 1
            self.Cpy = np.array([[1, 1, 0, 0]], dtype=np.float32)
        else:
            raise ValueError(
                f"observation {self.observation} must be one of 'partial', 'full', 'rigid_full'"
            )
        self.Dpyw = np.zeros((self.ny, self.nw), dtype=np.float32)
        # Dpyu is always 0

        assert "supply_rate" in env_config
        self.supply_rate = env_config["supply_rate"]
        if self.supply_rate == "stability":
            print(
                "Plant using stability construction for disturbance, performance output, and supply rate."
            )
            self.nd = 1
            self.ne = self.nx

            alpha = 0

            self.Bpd = np.zeros((self.nx, self.nd), dtype=np.float32)
            self.Dpvd = np.zeros((self.nv, self.nd), dtype=np.float32)
            self.Dpyd = np.zeros((self.ny, self.nd), dtype=np.float32)

            self.Cpe = np.eye(self.nx, dtype=np.float32)
            self.Dpew = np.zeros((self.ne, self.nw), dtype=np.float32)
            self.Dped = np.zeros((self.ne, self.nd), dtype=np.float32)
            self.Dpeu = np.zeros((self.ne, self.nu), dtype=np.float32)

            self.Xdd = 0 * np.eye(self.nd, dtype=np.float32)
            self.Xde = np.zeros((self.nd, self.ne), dtype=np.float32)
            self.Xee = -alpha * np.eye(self.ne, dtype=np.float32)
        elif self.supply_rate == "l2_gain":
            print(
                "Plant using L2 gain construction for disturbance, performance output, and supply rate."
            )
            # Supply rate for L2 gain of 0.99 from disturbance into input to output being the state
            self.nd = self.nu
            self.ne = self.nx

            self.Bpd = self.Bpu.copy()
            self.Dpvd = np.zeros((self.nv, self.nd), dtype=np.float32)
            self.Dpyd = np.zeros((self.ny, self.nd), dtype=np.float32)

            self.Cpe = np.eye(self.nx, dtype=np.float32)
            self.Dpew = np.zeros((self.ne, self.nw), dtype=np.float32)
            self.Dped = np.zeros((self.ne, self.nd), dtype=np.float32)
            self.Dpeu = np.zeros((self.ne, self.nu), dtype=np.float32)

            gamma = 0.99
            self.Xdd = gamma * np.eye(self.nd, dtype=np.float32)
            self.Xde = np.zeros((self.nd, self.ne), dtype=np.float32)
            self.Xee = -np.eye(self.ne, dtype=np.float32)
        else:
            raise ValueError(
                f"Supply rate {self.supply_rate} must be one of: 'stability', 'l2_gain'."
            )

        # Make sure dimensions of parameters match up.
        self.check_parameter_sizes()

        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(self.nu,),
            dtype=np.float32,
        )
        x_max = np.array([self.max_x, self.max_h, self.max_xdot, self.max_hdot], dtype=np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)
        # if self.observation == "full":
        #     ob_max = self.Cpy @ x_max
        # else:
        #     ob_max = np.array([self.max_x + self.max_h], dtype=np.float32)
        ob_max = self.Cpy @ x_max
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)

        if normed:
            self.Cpy = self.Cpy / self.state_space.high

        self.state_size = self.nx
        self.nonlin_size = (
            self.nv
        )  # TODO(Neelay): this nonlin_size parameter likely only works when nv = nw. Fix.

        self.MDeltapvv = np.zeros((self.nv, self.nv), dtype=np.float32)
        self.MDeltapvw = np.zeros((self.nv, self.nw), dtype=np.float32)
        self.MDeltapww = np.zeros((self.nw, self.nw), dtype=np.float32)

        self.max_reward = 1

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
                # print("D low")
                d = 2 * self.action_space.low.astype(np.float32)
            elif rand < p:
                # print("D high")
                d = 2 * self.action_space.high.astype(np.float32)
            else:
                d = np.zeros((self.nd,), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected disturbance model: {self.disturbance_model}.")

        self.state = self.next_state(self.state, d, u)

        # Reward for small state and small control
        reward_state = np.exp(-(np.linalg.norm(self.state) ** 2))
        reward_control = np.exp(-(np.linalg.norm(u) ** 2))
        reward = reward_control # reward_state + reward_control

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
        if self.design_model == "flexible":
            # fmt: off
            return PlantParameters(
                self.Ap, self.Bpw, self.Bpd, self.Bpu, 
                self.Cpv, self.Dpvw, self.Dpvd, self.Dpvu,
                self.Cpe, self.Dpew, self.Dped, self.Dpeu,
                self.Cpy, self.Dpyw, self.Dpyd,
                self.MDeltapvv, self.MDeltapvw, self.MDeltapww,
                self.Xdd, self.Xde, self.Xee
            )
            # fmt: on
            # return {
            #     "Ap": self.Ap,
            #     "Bpw": self.Bpw,
            #     "Bpd": self.Bpd,
            #     "Bpu": self.Bpu,
            #     "Cpv": self.Cpv,
            #     "Dpvw": self.Dpvw,
            #     "Dpvd": self.Dpvd,
            #     "Dpvu": self.Dpvu,
            #     "Cpe": self.Cpe,
            #     "Dpew": self.Dpew,
            #     "Dped": self.Dped,
            #     "Dpeu": self.Dpeu,
            #     "Cpy": self.Cpy,
            #     "Dpyw": self.Dpyw,
            #     "Dpyd": self.Dpyd,
            #     "MDeltapvv": self.MDeltapvv,
            #     "MDeltapvw": self.MDeltapvw,
            #     "MDeltapww": self.MDeltapww,
            #     "Xdd": self.Xdd,
            #     "Xde": self.Xde,
            #     "Xee": self.Xee,
            # }
        elif self.design_model == "rigid":
            # Can copy some because dimensions are same as with flexible model
            nx = 2
            Ap = np.array([[0, 1], [0, 0]], dtype=np.float32)
            Bpu = np.array([[0], [1.0 / self.Mr]], dtype=np.float32)
            if self.supply_rate == "stability":
                Bpd = np.zeros((nx, self.nd), dtype=np.float32)
            elif self.supply_rate == "l2_gain":
                Bpd = Bpu.copy()
            else:
                raise ValueError(f"Unexpected supply rate: {self.supply_rate}")
            if self.observation == "full":
                Cpy = np.eye(2, dtype=np.float32)
            elif self.observation == "partial":
                Cpy = np.array([[1, 0]], dtype=np.float32)
            else:
                raise ValueError(f"Unexpected observavion: {self.observation}")
            Bpw = np.zeros((nx, self.nw), dtype=np.float32)
            Cpv = np.zeros((self.nv, nx), dtype=np.float32)
            Cpe = np.eye(nx, dtype=np.float32)
            Dpew = np.zeros((nx, self.nw), dtype=np.float32)
            Dped = np.zeros((nx, self.nd), dtype=np.float32)
            Dpeu = np.zeros((nx, self.nu), dtype=np.float32)
            # fmt: off
            return PlantParameters(
                Ap, Bpw, Bpd, Bpu,
                Cpv, self.Dpvw, self.Dpvd, self.Dpvu,
                Cpe, Dpew, Dped, Dpeu,
                Cpy, self.Dpyw, self.Dpyd,
                self.MDeltapvv, self.MDeltapvw, self.MDeltapww,
                self.Xdd, self.Xde, self.Xee
            )
        # fmt: on
        # return {
        #     "Ap": Ap,
        #     "Bpw": np.zeros((nx, self.nw), dtype=np.float32),
        #     "Bpd": Bpd,
        #     "Bpu": Bpu,
        #     "Cpv": np.zeros((self.nv, nx), dtype=np.float32),
        #     "Dpvw": self.Dpvw,
        #     "Dpvd": self.Dpvd,
        #     "Dpvu": self.Dpvu,
        #     "Cpe": np.eye(nx, dtype=np.float32),
        #     "Dpew": np.zeros((nx, self.nw), dtype=np.float32),
        #     "Dped": np.zeros((nx, self.nd), dtype=np.float32),
        #     "Dpeu": np.zeros((nx, self.nu), dtype=np.float32),
        #     "Cpy": Cpy,
        #     "Dpyw": self.Dpyw,
        #     "Dpyd": self.Dpyd,
        #     "MDeltapvv": self.MDeltapvv,
        #     "MDeltapvw": self.MDeltapvw,
        #     "MDeltapww": self.MDeltapww,
        #     "Xdd": self.Xdd,
        #     "Xde": self.Xde,
        #     "Xee": self.Xee,
        # }
        else:
            raise ValueError(f"Unknown design model: {self.design_model}")

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
