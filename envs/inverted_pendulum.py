import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

class InvertedPendulumEnv(gym.Env):

    def __init__(self, factor = 1, observation = 'full', normed = False):

        self.factor = factor
        self.viewer = None
        self.g = 10.0
        self.m = 0.15
        self.l = 0.5
        self.mu = 0.05
        self.dt = 0.02
        self.max_torque = 2 #
        self.max_speed = 8.0 * factor
        self.max_pos = np.pi/2 * factor # pi/2 = 1.571 # 1.5 * factor

        # This uses a different splitting of variables than Galaxy's
        self.AG = np.array([
            [1, self.dt],
            [0, 1-self.mu/(self.m*self.l**2)*self.dt]
        ])
        # self.g/self.l*self.dt
        self.BG1 = np.array([[0], [-self.g*self.dt/self.l]]) #* factor
        self.BG2 = np.array([[0], [self.dt/(self.m*self.l**2)]]) * factor

        self.nx = self.AG.shape[0]
        self.nu = self.BG2.shape[1]

        if observation == 'full':
            self.CG1 = np.eye(self.nx)
        elif observation == 'partial':
            self.CG1 = np.array([[1, 0]])
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')
        
        self.CG2 = np.array([[1, 0]])
        self.DG3 = np.array([[0]])

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(self.nu,))
        # observations are the two states
        x_max = np.array([self.max_pos, self.max_speed])
        self.state_space = spaces.Box(low=-x_max, high=x_max)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = x_max[0:1]
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max)
        
        if normed:
            self.CG1 = self.CG1 / self.observation_space.high

        self.state_size = self.nx
        self.nonlin_size = 1

        # Sector bounds on Delta (in this case Delta = sin)
        # Sin is sector-bounded [0, 1] from [-pi, pi], [2/pi, 1] from [-pi/2, pi/2], and sector-bounded about [-0.2173, 1] in general.
        self.C_Delta = 2/math.pi # 0
        self.D_Delta = 1

        self.seed()

    def in_state_space(self):
        return self.state.shape == self.state_space.shape \
            and np.all(self.state >= self.state_space.low) \
            and np.all(self.state <= self.state_space.high)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state

        u = np.clip(u, -self.max_torque, self.max_torque)
        # costs = 1/self.factor**2*(th**2 + .1*thdot**2 + 1*((u*self.factor)**2)) # - 1
        costs = 1*th**2 + 0.1**thdot**2 + 0.01*(u*self.factor)**2 + 5*np.max([0, np.abs(u*self.factor)-0.5])
        
        self.state = self.AG @ self.state + self.BG1 @ np.sin(self.CG2 @ self.state) + self.BG2 @ (u*self.factor)

        terminated = False
        if not self.in_state_space():
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        # high = np.array([np.pi/30, np.pi/20]) * self.factor
        high = np.array([0.8*np.pi/2, np.pi/20]) * self.factor
        self.state = self.np_random.uniform(low=-high, high=high)
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.CG1 @ self.state

    def is_nonlin(self):
        return True