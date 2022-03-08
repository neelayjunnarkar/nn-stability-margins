import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class PendubotEnv(gym.Env):

    def __init__(self, env_config):
        factor      = env_config['factor']
        observation = env_config['observation']
        normed      = env_config['normed']

        self.factor = factor
        self.viewer = None
        self.dt = 0.01 # sampling time

        # maximum control input
        self.control_scale = 1
        self.max_control = 1.0 * self.control_scale * 2
        self.soft_max_control = self.max_control / 4

        self.time_max = 30

        # Pendubot dynamics from Section V-A in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9345365
        # x1-theta1, x2-theta1dot, x3-theta2, x4-theta2dot
        # continuous-time dynamics xdot = Ac*x + Bc*u
        Ac = np.array([[0, 1, 0, 0], [67.38, 0, -24.83, 0], [0, 0, 0, 1], [-69.53, 0, 105.32, 0]], dtype=np.float32)
        Bc = np.array([[0], [44.87], [0], [-85.09]], dtype=np.float32) / self.control_scale

        # discrete-time system
        self.AG = Ac * self.dt + np.eye(4, dtype=np.float32)
        self.BG = Bc * self.dt * factor

        if observation == 'full':
            self.CG = np.eye(4, dtype=np.float32)
        elif observation == 'partial':
            self.obs_scale = 1
            self.CG = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ], dtype=np.float32) * self.obs_scale
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        self.action_space = spaces.Box(low=-self.max_control, high=self.max_control, shape=(self.nu,), dtype=np.float32)

        # x max limits
        ts = 1
        self.x1lim = 1.0 * factor*ts
        self.x2lim = 2.0 * factor*ts * 10
        self.x3lim = 1.0 * factor*ts
        self.x4lim = 4.0 * factor*ts * 5
        x_max = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim], dtype=np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = np.array([self.x1lim, self.x3lim], dtype=np.float32) * self.obs_scale
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)
        
        if normed:
            self.CG = self.CG / self.observation_space.high[:, np.newaxis]

        self.state_size = self.nx

    def step(self,u):
        x1, x2, x3, x4 = self.state
        u = np.clip(u, -self.max_control, self.max_control)
        # costs = 1/self.factor**2 * (1.0 * x1**2 + 0.05 * x2**2 + 1.0 * x3**2 + 0.05 * x4**2 + 0.2 * (u / self.control_scale * self.factor)**2) - 5.0
        costs = 10*(np.max([0, np.abs(u[0]*self.factor)-self.soft_max_control]) - self.max_control*self.factor)
        
        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        if self.time >= self.time_max or self.state not in self.state_space:
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.array([0.05, 0.1, 0.05, 0.1], dtype=np.float32) * self.factor
        self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return self.CG @ self.state

    def is_nonlin(self):
        return False