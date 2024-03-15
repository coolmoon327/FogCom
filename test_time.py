from packages.utils.utils import read_config
from packages.alg.PPO import train_ppo_for_fogcom
from packages.env.fogcom.environment import Environment
from packages.env.fogcom.node import Node
from packages.env.fogcom.task import Task
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import os

def scheduled_after_PING_serial(config):
    env = Environment(config)

    env.next_task()
    
    ping_time = 0.  # 默认使用 1MB 的包进行测速
    start_time = time.time()

    task: Task = env.new_tasks[env.task_index-1]
    provider: Node = task.provider()
    candidates = config['vm_database'][task.sid].get_servers()
    
    for storage in candidates:
        bw_fd, lt_fd = config['link_check'].check(provider, storage)
        ping_time += lt_fd + 1/bw_fd
    
    env.leader.search_candidates(task)

    end_time = time.time()

    total_time = (end_time - start_time) * 1000 + ping_time

    return total_time

def scheduled_after_PING_parallel(config):
    env = Environment(config)
    env.next_task()
    
    ping_time = 0.  # 默认使用 1MB 的包进行测速
    start_time = time.time()

    task: Task = env.new_tasks[env.task_index-1]
    provider: Node = task.provider()
    candidates = config['vm_database'][task.sid].get_servers()
    
    for storage in candidates:
        bw_fd, lt_fd = config['link_check'].check(provider, storage)
        ping_time = max(lt_fd + 1/bw_fd, ping_time)
    
    env.leader.search_candidates(task)

    end_time = time.time()

    total_time = (end_time - start_time) * 1000 + ping_time

    return total_time

def PPO(config):
    env = Environment(config)
    env.next_task()

    PPO_time = 1.27 # 从 eval 中获得的均值
    start_time = time.time()

    task: Task = env.new_tasks[env.task_index-1]
    
    env.leader.search_candidates(task)

    end_time = time.time()

    total_time = (end_time - start_time) * 1000 + PPO_time

    return total_time

if __name__ == "__main__":
    config = read_config('config.yml')
    np.random.seed(config['seed'])

    # print(scheduled_after_PING_serial(config))
    # print(scheduled_after_PING_parallel(config))
    # print(PPO(config))

    ping_s = []
    ping_p = []
    ppo = []

    sticks = range(10, 301, 10)

    for i in sticks:
        config["N_m"] = i
        psj = []
        ppj = []
        poj = []
        for j in range(10):
            psj.append(scheduled_after_PING_serial(copy.deepcopy(config)))
            ppj.append(scheduled_after_PING_parallel(copy.deepcopy(config)))
            poj.append(PPO(copy.deepcopy(config)))
        ping_s.append(np.mean(psj))
        ping_p.append(np.mean(ppj))
        ppo.append(np.mean(poj))
    
    print(ping_s[-1], ping_p[-1], ppo[-1])

    # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    # 设置图形尺寸和标题
    plt.figure(figsize=(12, 8))
    # plt.title("Results Comparison")

    # 绘制基准线和 PPO 奖励曲线
    plt.plot(sticks, ppo, color='blue', linewidth=2, label="PPO")
    plt.plot(sticks, ping_s, color='orange', linewidth=2, label="Serial Ping")
    plt.plot(sticks, ping_p, color='pink', linewidth=2, label="Parallel Ping")

    # plt.ylim(0, 1000)
    # plt.xlim(0, 1000)
    # plt.yscale('log') 

    # 添加图例和坐标轴标签
    plt.legend(loc='upper left', fontsize=25)
    plt.xlabel("Edge Server Number", fontsize=35, weight='bold')
    plt.ylabel("Time (ms)", fontsize=35, weight='bold')

    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    # 保存图像到指定位置
    plt.savefig("./results/time.png")
    plt.close()