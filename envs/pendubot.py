import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class PendubotEnv(gym.Env):

    def __init__(self, factor = 1, observation = 'full', normed = False):

        self.factor = factor
        self.viewer = None
        self.dt = 0.01 # sampling time

        # maximum control input
        self.control_scale = 1
        self.max_control = 1.0 * self.control_scale * 2

        # Pendubot dynamics from Section V-A in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9345365
        # x1-theta1, x2-theta1dot, x3-theta2, x4-theta2dot
        # continuous-time dynamics xdot = Ac*x + Bc*u
        Ac = np.array([[0, 1, 0, 0], [67.38, 0, -24.83, 0], [0, 0, 0, 1], [-69.53, 0, 105.32, 0]])
        Bc = np.array([[0], [44.87], [0], [-85.09]]) / self.control_scale

        # discrete-time system
        self.AG = Ac * self.dt + np.eye(4)
        self.BG = Bc * self.dt * factor

        if observation == 'full':
            self.CG = np.eye(4)
        elif observation == 'partial':
            self.obs_scale = 1
            self.CG = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ]) * self.obs_scale
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_control, high=self.max_control, shape=(self.nu,))

        # x max limits
        ts = 1
        self.x1lim = 1.0 * factor*ts
        self.x2lim = 2.0 * factor*ts * 10
        self.x3lim = 1.0 * factor*ts
        self.x4lim = 4.0 * factor*ts * 5
        x_max = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])
        self.state_space = spaces.Box(low=-x_max, high=x_max)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = np.array([self.x1lim, self.x3lim]) * self.obs_scale
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max)
        
        if normed:
            self.CG = self.CG / self.observation_space.high[:, np.newaxis]

        self.state_size = self.nx

        self.seed()


    def in_state_space(self):
        return self.state.shape == self.state_space.shape \
            and np.all(self.state >= self.state_space.low) \
            and np.all(self.state <= self.state_space.high)


    def step(self,u):
        x1, x2, x3, x4 = self.state
        u = np.clip(u, -self.max_control, self.max_control)
        costs = 1/self.factor**2 * (1.0 * x1**2 + 0.05 * x2**2 + 1.0 * x3**2 + 0.05 * x4**2 + 0.2 * (u / self.control_scale * self.factor)**2) # - 5.0

        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        if not self.in_state_space(): # or self.time > 200
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}


    def reset(self):
        high = np.array([0.05, 0.1, 0.05, 0.1]) * self.factor / 1
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