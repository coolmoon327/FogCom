from .fogcom.environment import *
import numpy as np

class EnvWrapper:
    def __init__(self, config={}):
        self.tag = -1   # set from outside
        self.config = config
        self.env = Environment(config)
        self.env_name = self.env.env_name
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.if_discrete = self.env.if_discrete

    def reset(self):
        state = self.env.reset()
        return state

    def seed(self, seed):
        self.env.seed(seed)
        
    def get_random_action(self):
        action = self.env.action_space.sample()
        return action

    def step(self, action):
        next_state, reward, terminal, info_dict = self.env.step(action.ravel())
        next_state = next_state.reshape(self.state_dim)
        return next_state, float(reward), terminal, info_dict

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
    