import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class CartpoleEnv(gym.Env):
    def __init__(self, factor = 1, observation = 'full', normed = False):
        self.factor = factor
        self.viewer = None
        self.dt = 0.02 # Sampling time.
        self.max_control = 2 # Max control input.

        # Discrete time model
        # x(k+1) = AG x(k) + BG u(k)
        # y(k)   = CG x(k)
         
        self.AG = np.array([
            [1.0, -0.001,  0.02, 0.0],
            [0.0,  1.005,  0.0,  0.02],
            [0.0, -0.079,  1.0, -0.001],
            [0.0,  0.550,  0.0,  1.005]
        ])

        self.BG = np.array([
            [0.0], [0.0], [0.04], [-0.04]
        ]) * factor

        if observation == 'full':
            self.CG = np.eye(4)
        elif observation == 'partial':
            self.CG = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_control, high=self.max_control, shape=(self.nu,))

        # x max limits
        ts = 1 # testing scale
        self.x1lim = 1.0 * factor * ts
        self.x2lim = np.pi/2 * factor * ts
        self.x3lim = 5.0 * factor * ts
        self.x4lim = 2.0 * np.pi * factor * ts
        x_max = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])
        
        self.state_space = spaces.Box(low=-x_max, high=x_max)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = np.array([self.x1lim, self.x2lim])
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max)

        if normed:
            self.CG = self.CG / self.observation_space.high[:, np.newaxis]

        self.state_size = self.nx

        self.seed()

    def in_state_space(self):
        return self.state.shape == self.state_space.shape \
            and np.all(self.state >= self.state_space.low) \
            and np.all(self.state <= self.state_space.high)
    
    def step(self, u):
        x1, x2, x3, x4 = self.state
        u = np.clip(u, -self.max_control, self.max_control)
        costs = 1/self.factor**2 * (1.0 * x1**2 + 1.0 * x2**2 + 0.04 * x3**2 + 0.1 * x4**2 + 0.2 * (u * self.factor)**2) # - 5.0

        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        # if self.time > 200 or not self.in_state_space():
        if not self.in_state_space():
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self, *, seed = None, options = None):
        high = np.array([0.05, 0.05, 0.25, 0.15])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.time = 0

        return self.get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def get_obs(self):
        return self.CG @ self.state

    def is_nonlin(self):
        return False