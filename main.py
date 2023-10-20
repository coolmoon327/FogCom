from packages.utils.utils import read_config
from packages.utils.logger import Logger
from packages.test.test_env import *
from packages.alg.PPO import train_ppo_for_fogcom

if __name__ == "__main__":
    config = read_config('config.yml')
    test_env(config)
    train_ppo_for_fogcom(config)
