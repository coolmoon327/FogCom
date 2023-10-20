import numpy as np
from ..env.wrapper import EnvWrapper


def test_env(config):
    print('\n| test_env.py')
    env = EnvWrapper(config)
    assert isinstance(env.env_name, str)
    assert isinstance(env.state_dim, int)
    assert isinstance(env.action_dim, int)
    assert isinstance(env.if_discrete, bool)

    state = env.reset()
    assert state.shape == (env.state_dim,)

    action = np.random.uniform(0, 1, size=env.action_dim)
    state, reward, done, info_dict = env.step(action)
    assert isinstance(state, np.ndarray)
    assert state.shape == (env.state_dim,)
    assert isinstance(state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info_dict, dict) or (info_dict is None)
    
    print('\n| End testing.')

