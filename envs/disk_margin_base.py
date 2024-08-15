import gym
import numpy as np
from gym.utils import seeding
from variable_structs import PlantParameters
import cvxpy as cp

class DiskMarginBaseEnv(gym.Env):
    """
    A base class for implenting LTI plants with disk margin uncertainty at the input.
    """

    def __init__(
        self, env_config, Ap, Bpd, Bpu, Cpe, Dped, Dpeu, Cpy, Dpyd, skew, alpha
    ):
        if "dt" in env_config:  # Discretization time (s).
            self.dt = env_config["dt"]
        else:
            self.dt = 0.01

        self.nx = Ap.shape[0]  # Plant state size.
        assert (
            Ap.shape[1] == self.nx
            and Bpd.shape[0] == self.nx
            and Bpu.shape[0] == self.nx
        )
        self.nw = Bpu.shape[1]  # Output size of uncertainty Delta.
        assert Dpeu.shape[1] == self.nw
        self.nd = Bpd.shape[1]  # Disturbance size.
        assert Dped.shape[1] == self.nd and Dpyd.shape[1] == self.nd
        self.nu = self.nw  # Control input size.
        self.nv = self.nw  # Input size to uncertainty Delta.
        self.ne = Cpe.shape[0]  # Performance output size.
        assert Dped.shape[0] == self.ne and Dpeu.shape[0] == self.ne
        self.ny = Cpy.shape[0]  # Measurement output size.
        assert Dpyd.shape[0] == self.ny

        self.Ap = Ap
        self.Bpw = Bpu
        self.Bpd = Bpd
        self.Bpu = Bpu

        self.Cpv = np.zeros((self.nv, self.nx), dtype=np.float32)
        self.Dpvw = (1 + skew) / 2 * np.eye(self.nv, dtype=np.float32)
        self.Dpvd = np.zeros((self.nv, self.nd), dtype=np.float32)
        self.Dpvu = np.eye(self.nv, dtype=np.float32)

        self.Cpe = Cpe
        self.Dpew = Dpeu
        self.Dped = Dped
        self.Dpeu = Dpeu

        self.Cpy = Cpy
        self.Dpyw = np.zeros((self.ny, self.nw), dtype=np.float32)
        self.Dpyd = Dpyd

        self.state_size = self.nx
        self.MDeltapvv = alpha**2 * np.eye(self.nv, dtype=np.float32)
        self.MDeltapvw = np.zeros((self.nv, self.nw), dtype=np.float32)
        self.MDeltapww = -np.eye(self.nw, dtype=np.float32)

        def plant_uncertainty_constraints(eps):
            Lambda = cp.Variable((self.nv, self.nv), diag=True)
            MDeltapvv = alpha**2 * Lambda
            MDeltapvw = np.zeros((self.nv, self.nw), dtype=np.float32)
            MDeltapww = -Lambda
            variables = [Lambda]
            constraints = [Lambda >> 0]
            return (MDeltapvv, MDeltapvw, MDeltapww, variables, constraints)
        self.plant_uncertainty_constraints = plant_uncertainty_constraints

        self.seed(env_config["seed"] if "seed" in env_config else None)

        # For subclass to setup:
        #   self.action_space, self.state_space, self.observation_space
        #   Optionally normalize Cpy and Dpyd
        #   self.time_max
        #   self.max_reward
        #   function compute_reward(state, action) to real
        #   function random_initial_state() to intial state
        #   self.Xdd, self.Xde, self.Xee

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def xdot(self, state, d, u):
        # Compute derivative of nominal model
        # The disk margin is only for the theoretical bit
        xdot = self.Ap @ state + self.Bpd @ d + self.Bpu @ u
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
        u = np.clip(u, self.action_space.low, self.action_space.high)
        d = np.zeros(self.nd, dtype=np.float32)
        self.state = self.next_state(self.state, d, u)
        reward = self.compute_reward(self.state, u)

        terminated = False
        if fail_on_time_limit and self.time >= self.time_max:
            terminated = True
        if fail_on_state_space and self.state not in self.state_space:
            terminated = True

        self.time += 1

        return self.get_obs(), reward, terminated, {}

    def reset(self, state=None):
        if state is None:
            self.state = self.random_initial_state()
        else:
            self.state = state
        self.states = [self.state]
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        # TODO: setting disturbance to 0, making it only for guarantees
        # return self.Cpy @ self.state + self.Dpyd @ d
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