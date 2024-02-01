import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from variable_structs import PlantParameters
import queue


class TimeDelayInvertedPendulumEnv(gym.Env):
    """
    Nonlinear inverted pendulum with the input delayed.
    """

    def __init__(self, env_config):
        assert "observation" in env_config
        self.observation = env_config["observation"]

        assert "normed" in env_config
        self.normed = env_config["normed"]

        self.g = 9.8  # gravity (m/s^2)
        self.m = 0.15  # mass (Kg)
        self.l = 0.5  # length of pendulum (m)
        self.mu = 0.05  # coefficient of friction (Nms/rad)
        self.max_torque = 2
        self.max_speed = 8.0
        self.max_pos = np.pi

        if "dt" in env_config:  # Discretization time (s)
            self.dt = env_config["dt"]
        else:
            self.dt = 0.01
        self.time_max = 200

        # Actual simulated time delay is time_delay_steps*dt seconds
        if "time_delay_steps" in env_config:
            self.time_delay_steps = env_config["time_delay_steps"]
        else:
            self.time_delay_steps = 10
        
        if "design_time_delay" in env_config:
            self.design_time_delay = env_config["design_time_delay"]
        else:
            self.design_time_delay = 0.2 # seconds

        self.max_reward = 2

        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32,
        )
        x_max = np.array([self.max_pos, self.max_speed], dtype=np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)
        if self.observation == "full":
            ob_max = x_max
        else:
            ob_max = x_max[0:1]
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)

        self.design_model = self._build_design_model(self.observation)

        self.input_buffer = queue.Queue(maxsize=self.time_delay_steps + 1)

        self.seed(env_config["seed"] if "seed" in env_config else None)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def xdot(self, state, _d, _u):
        assert len(state) == 2

        # Access delayed input
        if self.input_buffer.full():
            Du = self.input_buffer.get()
        else:
            Du = np.zeros(self.action_space.shape)

        x1 = state[0]
        x2 = state[1]
        x1dot = x2
        x2dot = (
            -(self.mu / (self.m * self.l**2)) * x2
            + (self.g / self.l) * np.sin(x1)
            + (1 / (self.m * self.l**2)) * Du
        )

        xdot = np.zeros_like(state)
        assert len(xdot) == 2
        xdot[0] = x1dot
        xdot[1] = x2dot
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
        self.input_buffer.put(u)

        d = np.zeros((1,), dtype=np.float32)
        self.state = self.next_state(self.state, d, u)

        # Reward for small angle, small angular velocity, and small control
        theta, thetadot = self.state
        reward_state = np.exp(-(theta**2)) + np.exp(-(thetadot**2))
        reward_control = np.exp(-(u[0] ** 2))
        reward = reward_state + reward_control
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
            high = np.array([0.6 * self.max_pos, 0.25 * self.max_speed], dtype=np.float32)
            self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        else:
            self.state = state
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        if self.observation == "full":
            y = self.state # angular position and angular velocity
        elif self.observation == "partial":
            y = self.state[0] # Just angular position
        else:
            raise ValueError(f"Unexpected observation: {self.observation}.")
        if self.normed:
            y = y/self.observation_space.high
        return y

    def get_params(self):
        return self.design_model

    def _build_design_model(self, observation):
        nx = 2
        nu = 1
        if observation == "full":
            ny = nx
        elif observation == "partial":
            ny = 1
        else:
            raise ValueError(f"Unexpected observation: {self.observation}.")
        nv = 2
        nw = 2
        nd = 1
        ne = nx

        # First, build plant corresponding to inverted pendulum
        Ap0 = np.array(
            [[0, 1], [self.g / (2 * self.l), -(self.mu / (self.m * self.l**2))]], dtype=np.float32
        )
        Bpw0 = np.array(
            [[0, 0], [self.g / (2 * self.l), 1 / (self.m * self.l**2)]], dtype=np.float32
        )
        Bpd0 = np.zeros((nx, nd), dtype=np.float32)
        Bpu0 = np.array([[0], [1 / (self.m * self.l**2)]], dtype=np.float32)

        Cpv0 = np.array([[1, 0], [0, 0]], dtype=np.float32)
        Dpvw0 = np.zeros((nv, nw), dtype=np.float32)
        Dpvd0 = np.zeros((nv, nd), dtype=np.float32)
        Dpvu0 = np.array([[0], [1]], dtype=np.float32)

        Cpe0 = np.eye(nx, dtype=np.float32)
        Dpew0 = np.zeros((ne, nw), dtype=np.float32)
        Dped0 = np.zeros((ne, nd), dtype=np.float32)
        Dpeu0 = np.zeros((ne, nu), dtype=np.float32)

        if observation == "full":
            Cpy0 = np.eye(nx, dtype=np.float32)
        elif observation == "partial":
            Cpy0 = np.array([[1, 0]], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected observation: {self.observation}.")
        Dpyw0 = np.zeros((ny, nw), dtype=np.float32)
        Dpyd0 = np.zeros((ny, nd), dtype=np.float32)
        # Dpyu0 is always 0

        # Second, extend and transform with filters

        # State space form of phi transfer function (eq. 6) from "An Overview of Integral Quadratic Constraints for
        # Delayed Nonlinear and Parameter-Varying Systems" by Pfifer and Seiler
        time_delay = self.design_time_delay
        assert time_delay >= self.time_delay_steps * self.dt
        Apsi1 = np.array([[0, 1], [-7.1 / (time_delay**2), -4.5 / time_delay]], dtype=np.float32)
        Bpsi1 = np.array([[0, 0], [0, 1]], dtype=np.float32)
        Cpsi1 = np.array(
            [[0, 0], [(-14.2 + 2e-6) / (time_delay**2), -2 / time_delay]], dtype=np.float32
        )
        Dpsi1 = np.array([[1, 0], [0, 2]], dtype=np.float32)
        # The transformation "T" in this case is just the identity since Psi2 = I

        # fmt: off
        Ap = np.bmat([
            [Ap0, np.zeros((Ap0.shape[0], Apsi1.shape[1]))],
            [Bpsi1 @ Cpv0, Apsi1],
        ])
        # fmt: on
        Ap = np.asarray(Ap).astype(np.float32)
        Bpw = np.bmat([[Bpw0], [Bpsi1 @ Dpvw0]])
        Bpw = np.asarray(Bpw).astype(np.float32)
        Bpd = np.bmat([[Bpd0], [Bpsi1 @ Dpvd0]])
        Bpd = np.asarray(Bpd).astype(np.float32)
        Bpu = np.bmat([[Bpu0], [Bpsi1 @ Dpvu0]])
        Bpu = np.asarray(Bpu).astype(np.float32)

        Cpv = np.bmat([[Dpsi1 @ Cpv0, Cpsi1]])
        Cpv = np.asarray(Cpv).astype(np.float32)
        Dpvw = Dpsi1 @ Dpvw0
        Dpvd = Dpsi1 @ Dpvd0
        Dpvu = Dpsi1 @ Dpvu0

        Cpe = np.bmat([[Cpe0, np.zeros((Cpe0.shape[0], Apsi1.shape[0]))]])
        Cpe = np.asarray(Cpe).astype(np.float32)
        Dpew = Dpew0
        Dped = Dped0
        Dpeu = Dpeu0

        Cpy = np.bmat([[Cpy0, np.zeros((Cpy0.shape[0], Apsi1.shape[0]))]])
        Cpy = np.asarray(Cpy).astype(np.float32)
        Dpyw = Dpyw0
        Dpyd = Dpyd0

        MDeltapvv = np.eye(nv, dtype=np.float32)
        MDeltapvw = np.zeros((nv, nw), dtype=np.float32)
        MDeltapww = -np.eye(nw, dtype=np.float32)

        Xdd = 0 * np.eye(nd, dtype=np.float32)
        Xde = np.zeros((nd, ne), dtype=np.float32)
        Xee = 0 * np.eye(ne, dtype=np.float32)

        # fmt: off
        return PlantParameters(
            Ap, Bpw, Bpd, Bpu,
            Cpv, Dpvw, Dpvd, Dpvu,
            Cpe, Dpew, Dped, Dpeu,
            Cpy, Dpyw, Dpyd,
            MDeltapvv, MDeltapvw, MDeltapww,
            Xdd, Xde, Xee
        )
        # fmt: on
