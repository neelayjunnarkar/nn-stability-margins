import gym
from gym import spaces
import numpy as np

class InvertedPendulumEnv(gym.Env):
    """
    Nonlinear inverted pendulum with Delta(v) = sin(v)
    """

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
        if 'dt' in env_config:
            self.dt = env_config['dt']
        else:
            self.dt = 0.02
        self.max_torque = 2
        self.max_speed = 8.0 * factor
        self.max_pos = np.pi * factor 
        self.time_max = 200
        self.max_buff_size = 10

        self.AG = np.array([
            [1, self.dt],
            [0, 1-(self.dt*self.mu)/(self.m*self.l**2)]
        ], dtype=np.float32)
        
        self.BG1 = np.array([[0], [(self.g*self.dt)/self.l]], dtype=np.float32) #* factor
        self.BG2 = np.array([[0], [self.dt/(self.m*self.l**2)]], dtype=np.float32) * factor

        self.nx = self.AG.shape[0]
        self.nu = self.BG2.shape[1]

        if observation == 'full':
            self.CG1 = np.eye(self.nx, dtype=np.float32)
        elif observation == 'partial':
            self.CG1 = np.array([[1, 0]], dtype=np.float32)
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')
        
        self.CG2 = np.array([[1, 0]], dtype=np.float32)
        self.DG3 = np.array([[0]], dtype=np.float32)  

        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(self.nu,), dtype=np.float32)
        # observations are the two states
        x_max = np.array([self.max_pos, self.max_speed], dtype=np.float32)
        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = x_max[0:1]
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)
        
        if normed:
            self.CG1 = self.CG1 / self.observation_space.high

        self.state_size = self.nx
        self.nonlin_size = 1

        # Sector bounds on Delta (in this case Delta = sin)
        # Sin is sector-bounded [0, 1] from [-pi, pi], [2/pi, 1] from [-pi/2, pi/2], and sector-bounded about [-0.2173, 1] in general.
        # self.C_Delta = 2/math.pi
        # self.C_Delta = -0.2173
        self.C_Delta = 0
        self.D_Delta = 1

        # self.max_reward = self.max_torque**2 # For Ju reward
        self.max_reward = 5 # For Js reward

    def step(self, u, fail_on_state_space = True, fail_on_time_limit = True):
        th, thdot = self.state
        prev_th, prev_thdot = np.mean(self.states, axis = 0)

        u = np.clip(u, -self.max_torque, self.max_torque)
        u *= self.factor

        Js = 1*th**2 + 0.1*thdot**2 + 0.01*(u[0])**2
        max_Js = 5
        costs = Js - max_Js

        # Ju = u[0]**2
        # max_Ju = self.max_torque**2
        # costs = Ju - max_Ju
        
        self.states.append(self.state)
        if len(self.states) > self.max_buff_size:
            del self.states[0]
        self.state = self.AG @ self.state + self.BG1 @ np.sin(self.CG2 @ self.state) + self.BG2 @ (u*self.factor)
        
        terminated = False
        if (fail_on_time_limit and self.time >= self.time_max) or (fail_on_state_space and self.state not in self.state_space):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self, state = None):
        if state is None:
            # high = np.array([0.6 * self.max_pos, 0.1 * self.max_speed], dtype=np.float32) * self.factor
            high = np.array([0.6 * self.max_pos, 0.25 * self.max_speed], dtype=np.float32) * self.factor
            self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        else:
            self.state = state
        self.states = [self.state]
        self.time = 0

        return self.get_obs()

    def get_obs(self):
        return self.CG1 @ self.state

    def is_nonlin(self):
        return True

    def get_params(self):
        return [self.AG, self.BG1, self.BG2, self.CG1, self.CG2, self.DG3, self.C_Delta, self.D_Delta]
