import gym
from gym import spaces
import numpy as np

class OtherInvertedPendulumEnv(gym.Env):

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
        self.max_torque = 2 #
        self.soft_max_torque = 0.5
        self.max_speed = 8.0 * factor
        self.max_pos = np.pi * factor # pi/2 = 1.571 # 1.5 * factor
        self.time_max = 200 # 30 # 200
        self.max_buff_size = 10

        self.AG = np.array([
            [1, self.dt],
            [(self.g*self.dt)/self.l, 1-(self.dt*self.mu)/(self.m*self.l**2)]
        ], dtype=np.float32)
        
        self.BG1 = np.array([[0], [-(self.g*self.dt)/self.l]], dtype=np.float32) #* factor
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
        
        # Apparently the old projection method becomes super infeasible when I norm the output
        # if normed:
        #     self.CG1 = self.CG1 / self.observation_space.high

        self.state_size = self.nx
        self.nonlin_size = 1

        # Sector bounds on Delta (in this case Delta = theta - sin(theta))
        self.Delta = lambda v: v - np.sin(v)
        # Delta is sector-bounded [0, 1] from [-pi, pi], [0, 1.2173] in general
        self.C_Delta = 0
        # self.D_Delta = 1
        # self.D_Delta = 1.2173
        self.D_Delta = 0.41 # -1.4 to 1.4

        # self.max_reward = 3
        self.max_reward = self.max_torque**2 
        # self.max_reward = 5# 1*self.max_pos**2 + 0.1*self.max_speed**2 + 0.01*self.max_torque**2

        # Set up IQC filter for the old RNN projection method
        self.Dpsi1 = np.array([[self.D_Delta],
                               [-self.C_Delta]])
        self.Dpsi2 = np.array([[-1],
                               [1]])
        # M matrix for IQC
        self.M = np.array([[0, 1],
                           [1, 0]])

        # dynamics of the extended system of G and Psi
        self.Ae = self.AG
        self.Be1 = self.BG1
        self.Be2 = self.BG2
        self.Ce1 = self.Dpsi1 @ self.CG2
        self.De1 = self.Dpsi1 @ self.DG3 + self.Dpsi2
        self.Ce2 = self.CG1
        self.npsi = 0
        self.nr = self.Dpsi1.shape[0]
        self.nxe = self.Ae.shape[0]
        self.nq = self.BG1.shape[1]

    def step(self, u, fail_on_state_space = True, fail_on_time_limit = True):
        th, thdot = self.state
        prev_th, prev_thdot = np.mean(self.states, axis = 0)

        u = np.clip(u, -self.max_torque, self.max_torque)
        u *= self.factor
        # costs = 1/self.factor**2*(th**2 + .1*thdot**2 + 1*((u*self.factor)**2)) - 5
        # costs = 1*th**2 + 0.1*thdot**2 + 0.01*(u*self.factor)**2 + 5*np.max([0, np.abs(u*self.factor)-0.5])
        # Ju = 5*np.max([0, np.abs(u[0]*self.factor)-self.soft_max_torque])
        # max_Ju = 5*(self.max_torque-self.soft_max_torque)
        # Js = 1*th**2 + 0.1*thdot**2 + 0.01*(u[0]*self.factor)**2
        # max_Js = 5# 1*self.max_pos**2 + 0.1*self.max_speed**2 + 0.01*self.max_torque**2
        # costs = Js - max_Js
        # costs = Ju + Js - max_Ju - max_Js
        # Jaway = (th-self.max_pos)**2 + 0.1*(self.max_speed-thdot)**2 + 0.01*(u[0]*self.factor-self.max_torque)**2
        # max_Jaway = 4*max_Js
        # Jalt = 0.1*(thdot-thdot_old)**2 #+ 0.01*(u[0]*self.factor)**2
        # max_Jalt = 0.1*(2*self.max_speed)**2 #+ 0.01*self.max_torque**2
        Ju = u[0]**2
        max_Ju = self.max_torque**2
        costs = Ju - max_Ju
        # J = 0
        # J += ((th-prev_th)/(2*self.max_pos))**2
        # J += ((thdot-prev_thdot)/(2*self.max_speed))**2
        # J += (u[0]/self.max_torque)**2
        # max_J = 3
        # costs = J - max_J
        
        self.states.append(self.state)
        if len(self.states) > self.max_buff_size:
            del self.states[0]
        self.state = self.AG @ self.state + self.BG1 @ self.Delta(self.CG2 @ self.state) + self.BG2 @ u
        
        terminated = False
        if (fail_on_time_limit and self.time >= self.time_max) or (fail_on_state_space and self.state not in self.state_space):
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self, state = None):
        if state is None:
            # high = np.array([np.pi/30, np.pi/20], dtype=np.float32) * self.factor
            high = np.array([0.3 * self.max_pos, 0.1 * self.max_speed], dtype=np.float32) * self.factor
            # high = np.array([0.5 * self.max_pos, 0.25 * self.max_speed], dtype=np.float32) * self.factor
            # high = np.array([0.6 * self.max_pos, 0.25 * self.max_speed], dtype=np.float32) * self.factor
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