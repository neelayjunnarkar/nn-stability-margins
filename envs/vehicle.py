import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class VehicleLateralEnv(gym.Env):

    def __init__(self, env_config):
        factor      = env_config['factor']
        observation = env_config['observation']
        normed      = env_config['normed']

        self.factor = factor
        self.viewer = None

        # Nominal speed of the vehicle travels at.
        self.U = 28.0; # m/s
        # Front cornering stiffness for one wheel.
        self.Ca1 = -61595.0; # unit: Newtons/rad
        # Rear cornering stiffness for one wheel.
        self.Ca3 = -52095.0; # unit: Newtons/rad

        # Front cornering stiffness for two wheels.
        self.Caf = self.Ca1*2.0; # unit: Newtons/rad
        # Rear cornering stiffness for two wheels.
        self.Car = self.Ca3*2.0; # unit: Newtons/rad

        # Vehicle mass
        self.m = 1670.0; # kg
        # Moment of inertia
        self.Iz = 2100.0; # kg/m^2

        # Distance from vehicle CG to front axle
        self.a = 0.99; # m
        # Distance from vehicle CG to rear axle
        self.b = 1.7; # m

        self.g = 10.0
        self.dt = 0.02

        # maximum control input
        self.max_steering = np.pi/6 #* factor
        self.soft_max_steering = self.max_steering / 16

        self.time_max = 50

        # continuous-time model
        Ac = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, (self.Caf+self.Car)/(self.m*self.U), -(self.Caf+self.Car)/self.m, (self.a*self.Caf-self.b*self.Car)/(self.m*self.U)],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, (self.a*self.Caf-self.b*self.Car)/(self.Iz*self.U), -(self.a*self.Caf-self.b*self.Car)/self.Iz, (self.a**2*self.Caf+self.b**2*self.Car)/(self.Iz*self.U)]
        ], dtype=np.float32)
        Bc = np.array([
            [0.0], [-self.Caf/self.m], [0.0], [-self.a*self.Caf/self.Iz]
        ], dtype=np.float32) * factor

        self.nx = Ac.shape[0]
        self.nu = Bc.shape[1]

        # discrete-time model for control synthesis
        self.AG = np.eye(self.nx, dtype=np.float32) + Ac * self.dt
        self.BG = Bc * self.dt

        if observation == 'full':
            self.CG = np.eye(4, dtype=np.float32)
        elif observation == 'partial':
            self.CG = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ], dtype=np.float32)
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.action_space = spaces.Box(low=-self.max_steering, high=self.max_steering, shape=(self.nu,), dtype=np.float32)

        # xmax limits
        self.x1lim = 10.0 * factor
        self.x2lim = 5.0 * factor
        self.x3lim = 1.0 * factor
        self.x4lim = 5.0 * factor  
        x_max = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim], dtype=np.float32)

        self.state_space = spaces.Box(low=-x_max, high=x_max, dtype=np.float32)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = np.array([self.x1lim, self.x3lim], dtype=np.float32)
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max, dtype=np.float32)

        if normed:
            self.CG = self.CG / self.observation_space.high[:, np.newaxis]

        self.state_size = self.nx

        self.max_Js = 1*self.x1lim**2 + 0.1*self.x2lim**2 + 1*self.x3lim**2 + 0.1*self.x4lim**2 + 0.01*self.max_steering**2
        self.max_reward = self.max_Js

    def step(self, u):
        e, edot, etheta, ethetadot = self.state

        u = np.clip(u, -self.max_steering, self.max_steering)
        # costs = 0.01 * e**2 + 1/25.0 * edot**2 + etheta**2 + 1/25.0 * ethetadot**2 + 2.0/(np.pi/6.0)**2 * (u*self.factor)**2 - 5.0
        # costs = 10*(np.max([0, np.abs(u[0]*self.factor)-self.soft_max_steering]) - self.max_steering*self.factor)

        # Ju = 5*np.max([0, np.abs(u[0]*self.factor)-self.soft_max_steering])
        # max_Ju = 5*(self.max_steering - self.soft_max_steering)
        Js = 1*e**2 + 0.1*edot**2 + 1*etheta**2 + 0.1*ethetadot**2 + 0.01*(u[0]*self.factor)**2
        
        # costs = Ju + Js - (max_Ju + max_Js)/10
        costs = Js - self.max_Js

        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        if self.time >= self.time_max or self.state not in self.state_space:
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        # high = np.array([1.0, 0.5, 0.1, 0.5], dtype=np.float32) * self.factor
        high = 0.5*self.state_space.high
        self.state = self.np_random.uniform(low=-high, high=high).astype(np.float32)
        self.init_state = self.state.copy()
        self.time = 0

        return self.get_obs()
    
    def get_obs(self):
        return self.CG @ self.state

    def is_nonlin(self):
        return False