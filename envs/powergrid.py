import gym
from gym import spaces
import numpy as np

class PowergridEnv(gym.Env):

    def __init__(self, env_config):
        factor      = env_config['factor']
        observation = env_config['observation']
        normed      = env_config['normed']

        self.factor = factor
        self.viewer = None
        self.dt = 0.2 # sampling time

        # maximum control input
        self.control_scale = 1
        self.max_control = 1.0 * self.control_scale * 2
        self.soft_max_control = self.max_control / 4

        self.time_max = 30

        # powergrid dynamics in Section IV-B of https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9296215
        M = np.diag([4.0000, 3.0000, 2.5000, 4.0000, 2.0000, 3.5000, 3.0000, 2.5000, 2.0000, 6.0000]).astype(np.float32)
        L = np.array([
            [33.1348,   -2.5425,   -2.8928,   -2.0816,   -0.9371,   -2.1775,   -1.2132,  -10.3953,   -3.1871,   -7.7078],
            [-2.5425,   26.4495,  -10.5043,   -1.6747,   -0.7539,   -1.7518,   -0.9760,   -1.3576,   -0.8933,   -5.9955],
            [-2.8928,  -10.5043,   28.2182,   -2.3262,   -1.0471,   -2.4333,   -1.3557,   -1.5863,   -1.1153,   -4.9572],
            [-2.0816,   -1.6747,   -2.3262,   34.5327,  -16.1015,   -5.1567,   -2.8730,   -1.4638,   -1.5682,   -1.2871],
            [-0.9371,   -0.7539,   -1.0471,  -16.1015,   24.3984,   -2.3213,   -1.2933,   -0.6589,   -0.7059,   -0.5794],
            [-2.1775,   -1.7518,   -2.4333,   -5.1567,   -2.3213,   32.5848,  -14.2263,   -1.5312,   -1.6404,   -1.3463],
            [-1.2132,   -0.9760,   -1.3557,   -2.8730,   -1.2933,  -14.2263,   24.4547,   -0.8531,   -0.9139,   -0.7501],
            [-10.3953,   -1.3576,   -1.5863,   -1.4638,   -0.6589,   -1.5312,   -0.8531,   25.7938,   -4.4344,   -3.5132],
            [-3.1871,   -0.8933,   -1.1153,   -1.5682,   -0.7059,   -1.6404,   -0.9139,   -4.4344,   15.7360,   -1.2775],
            [-7.7078,   -5.9955,   -4.9572,   -1.2871,   -0.5794,   -1.3463,   -0.7501,   -3.5132,   -1.2775,   27.4140],
        ], dtype=np.float32)
        D = np.diag([5.0000, 4.0000, 4.0000, 6.0000, 3.5000, 3.0000, 7.5000, 4.0000, 6.5000, 5.0000]).astype(np.float32)
        # continuous-time dynamics xdot = Ac*x + Bc*u
        Ac = np.block([
            [np.zeros((10, 10), dtype=np.float32), np.eye(10, dtype=np.float32)],
            [-np.linalg.inv(M) @ L, -np.linalg.inv(M) @ D],
        ]).astype(np.float32)
        Bc = np.block([[np.zeros((10, 10))], [np.linalg.inv(M)]]).astype(np.float32) / self.control_scale

        # discrete-time system
        self.AG = Ac * self.dt + np.eye(20, dtype=np.float32)
        self.BG = Bc * self.dt * factor

        self.nx = self.AG.shape[0]
        self.nu = self.BG.shape[1]

        if observation == 'full':
            self.CG = np.eye(self.nx, dtype=np.float32)
        elif observation == 'partial':
            self.obs_scale = 1
            self.CG = np.block([
                np.eye(10, dtype=np.float32), np.zeros((10, 10), dtype=np.float32)
            ]).astype(np.float32) * self.obs_scale
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.action_space = spaces.Box(low=-self.max_control, high=self.max_control, shape=(self.nu,), dtype=np.float32)

        self.thetalim = 0.5 * factor * np.ones(10, dtype=np.float32)
        self.omegalim = 1.2 * factor * np.ones(10, dtype=np.float32)
        x_max = np.block([self.thetalim, self.omegalim]).astype(np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, shape=x_max.shape, dtype=np.float32)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = self.thetalim * self.obs_scale
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, shape=ob_max.shape, dtype=np.float32)

        if normed:
            self.CG = self.CG / self.observation_space.high[:, np.newaxis]

        self.state_size = self.nx
        

    def step(self,u):
        u = np.clip(u, -self.max_control, self.max_control)
        # costs = 1/self.factor**2 * (np.linalg.norm(self.state, 2)**2 + 0.2 * np.linalg.norm(u / self.control_scale * self.factor, 2)**2) - 5.0
        costs = 10*(np.max([0, np.linalg.norm(u*self.factor, 2)-self.soft_max_control]) - self.max_control*self.factor)
        
        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        if self.time >= self.time_max or self.state not in self.state_space:
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.block([self.thetalim / 10, self.omegalim / 10]).astype(np.float32) * self.factor
        self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return self.CG @ self.state

    def is_nonlin(self):
        return False
