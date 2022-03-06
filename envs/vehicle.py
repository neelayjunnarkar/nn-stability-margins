import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class VehicleLateralEnv(gym.Env):

    def __init__(self, factor = 1, observation = 'full', normed = False):

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

        # continuous-time model
        Ac = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, (self.Caf+self.Car)/(self.m*self.U), -(self.Caf+self.Car)/self.m, (self.a*self.Caf-self.b*self.Car)/(self.m*self.U)],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, (self.a*self.Caf-self.b*self.Car)/(self.Iz*self.U), -(self.a*self.Caf-self.b*self.Car)/self.Iz, (self.a**2*self.Caf+self.b**2*self.Car)/(self.Iz*self.U)]
        ])
        Bc = np.array([
            [0.0], [-self.Caf/self.m], [0.0], [-self.a*self.Caf/self.Iz]
        ]) * factor

        self.nx = Ac.shape[0]
        self.nu = Bc.shape[1]

        # discrete-time model for control synthesis
        self.AG = np.eye(self.nx) + Ac * self.dt
        self.BG = Bc * self.dt

        if observation == 'full':
            self.CG = np.eye(4)
        elif observation == 'partial':
            self.CG = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ])
        else:
            assert(f'observation must be one of \'partial\', \'full\', but was: {observation}')

        self.time = 0

        self.action_space = spaces.Box(low=-self.max_steering, high=self.max_steering, shape=(self.nu,))

        # xmax limits
        self.x1lim = 10.0 * factor
        self.x2lim = 5.0 * factor
        self.x3lim = 1.0 * factor
        self.x4lim = 5.0 * factor  
        x_max = np.array([self.x1lim, self.x2lim, self.x3lim, self.x4lim])

        self.state_space = spaces.Box(low=-x_max, high=x_max)

        if observation == 'full':
            ob_max = x_max
        else:
            ob_max = np.array([self.x1lim, self.x3lim])
        self.observation_space = spaces.Box(low=-ob_max, high=ob_max) # spaces.

        if normed:
            self.CG = self.CG / self.observation_space.high[:, np.newaxis]

        self.state_size = self.nx
        self.seed()

    def in_state_space(self):
        state = self.state
        return state.shape == self.state_space.shape \
            and np.all(state >= self.state_space.low) \
            and np.all(state <= self.state_space.high)

    def step(self, u):
        e, edot, etheta, ethetadot = self.state

        u = np.clip(u, -self.max_steering, self.max_steering)
        costs = 0.01 * e**2 + 1/25.0 * edot**2 + etheta**2 + 1/25.0 * ethetadot**2 + 2.0/(np.pi/6.0)**2 * (u*self.factor)**2 # - 5.0
        # costs = 100*e**2 + 10*edot**2 + 100*etheta**2 + 10*ethetadot**2 + u**2 + 
        # costs = (e - self.init_state[0])**2 - 5
        # costs = np.array([costs])

        self.state = self.AG @ self.state + self.BG @ u

        terminated = False
        if not self.in_state_space():
            terminated = True

        self.time += 1

        return self.get_obs(), -costs, terminated, {}

    def reset(self):
        high = np.array([1.0, 0.5, 0.1, 0.5]) * self.factor
        self.state = self.np_random.uniform(low=-high, high=high)
        self.init_state = self.state.copy()
        self.time = 0

        return self.get_obs()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self):
        return self.CG @ self.state

    def is_nonlin(self):
        return False