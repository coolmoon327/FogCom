from packages.utils.utils import read_config
from packages.utils.logger import Logger
from packages.test.test_env import *

if __name__ == "__main__":
    config = read_config('config.yml')
    test_env(config)
