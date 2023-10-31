from packages.utils.utils import read_config
from packages.utils.logger import Logger
from packages.test.test_env import *
from packages.alg.PPO import train_ppo_for_fogcom
import torch.multiprocessing as mp

if __name__ == "__main__":
    config = read_config('config.yml')
    
    mp.set_start_method('spawn')
    manager = mp.Manager()
    result_list = manager.list()
    lock = manager.Lock()
    threads_num = 10
    
    # test_env(config)
    train_ppo_for_fogcom(config, threads_num, result_list, lock)
