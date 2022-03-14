import gym
from gym import spaces
import numpy as np

class LinearizedInvertedPendulumEnv(gym.Env):

    def __init__(self, env_config):
        factor      = env_config['factor']
        observation = env_config['observation']
        normed      = env_config['normed']

        self.factor = factor
        self.viewer = None
        self.g = 10.0
        self.m = 0.15
        self.l = 0.5
        self.mu = 0.05
        self.dt = 0.02
        self.max_torque = 2 #
        self.soft_max_torque = self.max_torque / 4
        self.max_speed = 8.0 * factor
        self.max_pos = 1.5 * factor
        self.time_max = 30

        self.AG = np.array([
            [1, self.dt],
            [self.g/self.l*self.dt, 1-self.mu/(self.m*self.l**2)*self.dt]
        ], dtype=np.float32)
        self.BG = np.array([[0], [self.dt/(self.m*self.l**2)]], dtype=np.float32) * factor

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        if observation == 'full':
            self.CG = np.eye(self.nx, dtype=np.float32)
        elif observation == 'partial':
            self.CG = np.array([[1, 0]], dtype=np.float32)
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(self.nu,), dtype=np.float32)
        # observations are the two states
        x_max = np.array([self.max_pos, self.max_speed], dtype=np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, shape=x_max.shape, dtype=np.float32)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = x_max[0:1]
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, shape=ob_max.shape, dtype=np.float32)
        
        if normed:
            self.CG = self.CG / self.observation_space.high

        self.state_size = self.nx

    def step(self, u):
        th, thdot = self.state

        u = np.clip(u, -self.max_torque, self.max_torque)
        # costs = 1/self.factor**2*(th**2 + .1*thdot**2 + 1*((u*self.factor)**2)) - 5
        # costs = 10*(np.max([0, np.abs(u[0]*self.factor)-self.soft_max_torque]) - self.max_torque*self.factor)

        Ju = 5*np.max([0, np.abs(u[0]*self.factor)-self.soft_max_torque])
        max_Ju = 5*(self.max_torque-self.soft_max_torque)
        Js = 1*th**2 + 0.1*thdot**2 + 0.01*(u[0]*self.factor)**2
        max_Js = 1*self.max_pos**2 + 0.1*self.max_speed**2 + 0.01*self.max_torque**2
        costs = Ju + Js - max_Ju - max_Js

        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        if self.time >= self.time_max or self.state not in self.state_space:
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.array([np.pi/30, np.pi/20], dtype=np.float32) * self.factor
        
        self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return  self.CG @ self.state

    def is_nonlin(self):
        return False
