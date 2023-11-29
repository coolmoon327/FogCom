from packages.utils.utils import read_config
from packages.test.test_env import *
from packages.alg.PPO import test as test_ppo
from packages.alg.eval import test as test_others

if __name__ == "__main__":
    config = read_config('config.yml')
    
    # test_env(config)
    # test_ppo(config, 100)
    test_others(config)
    
    
