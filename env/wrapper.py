from .fogcom.environment import *
import numpy as np

class EnvWrapper:
    def __init__(self, config={}):
        self.config = config
        self.env = Environment(config)

    def reset(self):
        state = self.env.reset()
        return state

    def seed(self, seed):
        self.env.seed(seed)
        
    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, _ = self.env.step(action.ravel())
        return next_state, reward, terminal

    def set_random_seed(self, seed):
        self.env.seed(seed)

    def render(self):
        # frame = self.env.render(mode='rgb_array')
        # return frame
        pass

    def close(self):
        # self.env.close()
        pass

    def get_action_space(self):
        return self.env.action_space

    def normalise_state(self, state):
        return state

    def normalise_reward(self, reward):
        return reward
    