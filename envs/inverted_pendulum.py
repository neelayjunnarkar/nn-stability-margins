import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from variable_structs import PlantParameters


class InvertedPendulumEnv(gym.Env):
    """
    Nonlinear inverted pendulum with Deltap(v) = sin(v)
    """

    def __init__(self, env_config):
        assert "observation" in env_config
        observation = env_config["observation"]

        assert "normed" in env_config
        normed = env_config["normed"]

        assert "disturbance_model" in env_config
        self.disturbance_model = env_config["disturbance_model"]

        self.g = 9.8  # gravity (m/s^2)
        self.m = 0.15  # mass (Kg)
        self.l = 0.5  # length of pendulum (m)
        self.mu = 0.05  # coefficient of friction (Nms/rad)
        if "dt" in env_config:  # Discretization time (s)
            self.dt = env_config["dt"]
        else:
            self.dt = 0.01
        self.max_torque = 2
        self.max_speed = 8.0
        self.max_pos = np.pi
        self.time_max = 200

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

        self.nx = 2  # Plant state size
        self.nw = 1  # Output size of uncertainty Deltap
        self.nd = None  # Disturbance size. Defined based on supply_rate.
        self.nu = 1  # Control input size
        self.nv = 1  # Input size to uncertainty Deltap
        self.ne = None  # Performance output size. Defined based on supply_rate.
        self.ny = None  # Measurement output size. Defined later.

        self.Ap = np.array([[0, 1], [0, -self.mu / (self.m * self.l**2)]], dtype=np.float32)
        self.Bpw = np.array([[0], [self.g / self.l]], dtype=np.float32)
        self.Bpu = np.array([[0], [1 / (self.m * self.l**2)]], dtype=np.float32)

        self.Cpv = np.array([[1, 0]], dtype=np.float32)
        self.Dpvw = np.zeros((self.nv, self.nw), dtype=np.float32)
        self.Dpvu = np.zeros((self.nv, self.nu), dtype=np.float32)

        if observation == "full":  # Observe full state
            self.ny = self.nx
            self.Cpy = np.eye(self.nx, dtype=np.float32)
        elif observation == "partial":  # Observe only angle from vertical
            self.ny = 1
            self.Cpy = np.array([[1, 0]], dtype=np.float32)
        else:
            raise ValueError(f"observation {observation} must be one of 'partial', 'full'")
        self.Dpyw = np.zeros((self.ny, self.nw), dtype=np.float32)
        # Dpyu is always 0

        assert "supply_rate" in env_config
        if env_config["supply_rate"] == "stability":
            print(
                "Plant using stability construction for disturbance, performance output, and supply rate."
            )
            # This has issues because of the -Xde Ded - Ded^T Xde^T - Xde < 0 condition.
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
        elif env_config["supply_rate"] == "l2_gain":
            print(
                "Plant using L2 gain construction for disturbance, performance output, and supply rate."
            )
            # Supply rate for L2 gain of 0.99 from disturbance to output being the state
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
            alpha = 0.7
            self.Xdd = alpha * gamma**2 * np.eye(self.nd, dtype=np.float32)
            self.Xde = np.zeros((self.nd, self.ne), dtype=np.float32)
            self.Xee = -alpha * np.eye(self.ne, dtype=np.float32)
        else:
            raise ValueError(
                f"Supply rate {env_config['supply_rate']} must be one of: 'stability'."
            )

        # Make sure dimensions of parameters match up.
        self.check_parameter_sizes()

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(self.nu,),
            dtype=np.float32,
        )
        x_max = np.array([self.max_pos, self.max_speed], dtype=np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)
        if observation == "full":
            ob_max = x_max
        else:
            ob_max = x_max[0:1]
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)

        if normed:
            self.Cpy = self.Cpy / self.observation_space.high

        self.state_size = self.nx
        self.nonlin_size = (
            self.nv
        )  # TODO(Neelay): this nonlin_size parameter likely only works when nv = nw. Fix.

        # Sector bounds on Delta (in this case Delta = sin)
        # Sin is sector-bounded [0, 1] from [-pi, pi], [2/pi, 1] from [-pi/2, pi/2], and sector-bounded about [-0.2173, 1] in general.
        self.C_Delta = 0
        self.D_Delta = 1

        self.MDeltapvv = np.array([[-2 * self.C_Delta * self.D_Delta]], dtype=np.float32)
        self.MDeltapvw = np.array([[self.C_Delta + self.D_Delta]], dtype=np.float32)
        self.MDeltapww = np.array([[-2]], dtype=np.float32)

        self.max_reward = 1  # 2.1

        self.seed(env_config["seed"] if "seed" in env_config else None)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def Deltap(self, vp):
        wp = np.sin(vp)
        return wp

    def xdot(self, state, d, u):
        # In this case, vp doesnt depend on wp so can calculate both easily.
        vp = self.Cpv @ state + self.Dpvd @ d + self.Dpvu @ u
        wp = self.Deltap(vp)
        xdot = self.Ap @ state + self.Bpw @ wp + self.Bpd @ d + self.Bpu @ u
        return xdot

    def next_state(self, state, d, u):
        # # 0th order hold on derivative
        # xdot = self.xdot(state, d, u)
        # state = state + self.dt * xdot

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
        u = np.clip(u, -self.max_torque, self.max_torque)

        if self.disturbance_model == "none":
            d = np.zeros((self.nd,), dtype=np.float32)
        elif self.disturbance_model == "occasional":
            # Have disturbance ocurr (in expectation) thrice a second.
            p = 3 * self.dt
            rand = self.np_random.uniform()
            if rand < p / 2:
                d = 3 * self.action_space.low.astype(np.float32)
            elif rand < p:
                d = 3 * self.action_space.high.astype(np.float32)
            else:
                d = np.zeros((self.nd,), dtype=np.float32)
        else:
            raise ValueError(f"Unexpected disturbance model: {self.disturbance_model}.")

        self.state = self.next_state(self.state, d, u)

        # Reward for small angle, small angular velocity, and small control
        theta, thetadot = self.state
        reward_state = np.exp(-(theta**2)) + np.exp(-(thetadot**2))
        reward_control = np.exp(-(u[0] ** 2))
        # reward = reward_state + 0.1*reward_control
        reward = reward_control

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
            # high = np.array([0.6 * self.max_pos, 0.1 * self.max_speed], dtype=np.float32) * self.factor
            high = (
                np.array([0.6 * self.max_pos, 0.25 * self.max_speed], dtype=np.float32)
                # * self.factor
            )
            self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        else:
            self.state = state
        self.states = [self.state]
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return self.Cpy @ self.state

    def get_params(self):
        return PlantParameters(
            self.Ap,
            self.Bpw,
            self.Bpd,
            self.Bpu,
            self.Cpv,
            self.Dpvw,
            self.Dpvd,
            self.Dpvu,
            self.Cpe,
            self.Dpew,
            self.Dped,
            self.Dpeu,
            self.Cpy,
            self.Dpyw,
            self.Dpyd,
            self.MDeltapvv,
            self.MDeltapvw,
            self.MDeltapww,
            self.Xdd,
            self.Xde,
            self.Xee,
        )

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
