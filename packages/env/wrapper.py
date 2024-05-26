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
        next_state = self.normalise_state(next_state)
        next_state = next_state.reshape(self.state_dim)
        return next_state, float(reward), terminal, info_dict

    def step_with_inner_policy(self, policy_id: int):
        next_state, reward, terminal, info_dict = self.env.step_with_inner_policy(policy_id)
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
        if self.config['no_tag_mode']:
            # print(state)
            tag_len = self.config['follower_strategies_num']
            # 6 ~ tag_len+5
            for i in range(6, tag_len+6):
                state[i] = 1.
            # print(state)
        return state

    def normalise_reward(self, reward):
        return reward
    