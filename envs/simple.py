import gym
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self, env_config):
        self.delta = 0.02
        high = 1
        self.action_space = gym.spaces.Box(low=-high, high=high, shape=(1,), dtype=np.float32)
        self.state_space = gym.spaces.Box(low=-high, high=high, shape=(1,), dtype=np.float32)
        self.observation_space = self.state_space
    
    def reset(self, seed = None):
        super().reset(seed=seed)
        self.state = self.state_space.sample()
        self.num_steps = 0
        return self.state

    def step(self, u):
        self.num_steps += 1
        self.state = self.state + (-self.state + u)*self.delta
        cost = self.state**2 + u**2
        reward = -cost[0]
        done = (self.state not in self.state_space) or (self.num_steps >= 200)
        
        return self.state, reward, done, {}
