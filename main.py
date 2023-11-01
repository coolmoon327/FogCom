from packages.utils.utils import read_config
from packages.utils.logger import Logger
from packages.test.test_env import *
from packages.alg.PPO import train_ppo_for_fogcom
import torch.multiprocessing as mp
import faulthandler

if __name__ == "__main__":
    faulthandler.enable()
    
    config = read_config('config.yml')
    
    mp.set_start_method('spawn')
    manager = mp.Manager()
    result_list = manager.list()
    lock = manager.Lock()
    threads_num = 10 # 若要调试, 此处设置为 0
    
    # test_env(config)
    train_ppo_for_fogcom(config, threads_num, result_list, lock)
