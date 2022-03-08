import gym
from gym import spaces
import numpy as np

class CartpoleEnv(gym.Env):
    def __init__(self, env_config):
        factor      = env_config['factor']
        observation = env_config['observation']
        normed      = env_config['normed']

        self.factor = factor
        self.viewer = None
        self.dt = 0.02 # Sampling time.
        self.max_control = 2 # Max control input.
        self.soft_max_control = self.max_control/4
        self.time_max = 30

        # Discrete time model
        # x(k+1) = AG x(k) + BG u(k)
        # y(k)   = CG x(k)
         
        self.AG = np.array([
            [1.0, -0.001,  0.02, 0.0],
            [0.0,  1.005,  0.0,  0.02],
            [0.0, -0.079,  1.0, -0.001],
            [0.0,  0.550,  0.0,  1.005]
        ], dtype=np.float32)

        self.BG = np.array([
            [0.0], [0.0], [0.04], [-0.04]
        ], dtype=np.float32) * factor

        if observation == 'full':
            self.CG = np.eye(4, dtype=np.float32)
        elif observation == 'partial':
            self.CG = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32)
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        self.action_space = spaces.Box(low=-self.max_control, high=self.max_control, shape=(self.nu,), dtype=np.float32)

        # x max limits
        ts = 1 # testing scale
        self.x1lim = 1.0 * factor * ts
        self.x2lim = np.pi/2 * factor * ts
        self.x3lim = 5.0 * factor * ts
        self.x4lim = 2.0 * np.pi * factor * ts
        x_max = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim], dtype=np.float32)
        
        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = np.array([self.x1lim, self.x2lim], dtype=np.float32)
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)

        if normed:
            self.CG = self.CG / self.observation_space.high[:, np.newaxis]

        self.state_size = self.nx
    
    def step(self, u):
        x1, x2, x3, x4 = self.state
        u = np.clip(u, -self.max_control, self.max_control)
        # costs = 1/self.factor**2 * (1.0 * x1**2 + 1.0 * x2**2 + 0.04 * x3**2 + 0.1 * x4**2 + 0.2 * (u * self.factor)**2) - 5.0
        costs = 10*(np.max([0, np.abs(u[0]*self.factor)-self.soft_max_control]) - self.max_control*self.factor)

        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        if self.time >= self.time_max or self.state not in self.state_space:
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self, *, seed = None, options = None):
        high = np.array([0.05, 0.05, 0.25, 0.15], dtype=np.float32)
        self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        self.time = 0

        return self.get_obs()
    
    def get_obs(self):
        return self.CG @ self.state

    def is_nonlin(self):
        return False