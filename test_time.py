from packages.utils.utils import read_config
from packages.alg.PPO import train_ppo_for_fogcom
from packages.env.fogcom.environment import Environment
from packages.env.fogcom.node import Node
from packages.env.fogcom.task import Task
from packages.alg.PPO import *
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import os
import torch

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

def PPO(config, agent):
    env = Environment(config)
    state = env.next_task()

    # PPO_time = 1.27 # 从 eval 中获得的均值
    PPO_time = 0.
    start_time = time.time()

    task: Task = env.new_tasks[env.task_index-1]
    candidates = config['vm_database'][task.sid].get_servers()
    
    device = next(agent.act.parameters()).device
    tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    agent.act.eval()
    tensor_action = agent.act(tensor_state)

    env.leader.search_candidates(task)
    env.leader.inform_candidates(task, candidates)

    end_time = time.time()

    total_time = (end_time - start_time) * 1000 + PPO_time

    return total_time

if __name__ == "__main__":
    config = read_config('config.yml')
    # np.random.seed(config['seed'])

    act_grad_file = './results/act_grad.pth'
    act_model = torch.load(act_grad_file)
    args = set_args(config)
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    # agent.act.load_state_dict(act_model)
    PPO(copy.deepcopy(config),agent)    # 用来消除加载延迟

    # config["N_m"] = 1000000
    # print(PPO(copy.deepcopy(config),agent))
    # exit()

    ping_s = []
    ping_p = []
    ppo = []

    sticks = range(4, 101, 3)

    for i in sticks:
        config["N_m"] = i
        psj = []
        ppj = []
        poj = []
        for j in range(30):
            psj.append(scheduled_after_PING_serial(copy.deepcopy(config)))
            ppj.append(scheduled_after_PING_parallel(copy.deepcopy(config)))
            poj.append(PPO(copy.deepcopy(config),agent))
        ping_s.append(np.mean(psj))
        ping_p.append(np.mean(ppj))
        ppo.append(np.mean(poj))
    
    print(ping_s[-1], ping_p[-1], ppo[-1])

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

    # 设置图形尺寸和标题
    plt.figure(figsize=(12, 8))
    # plt.title("Results Comparison")

    # 绘制基准线和 PPO 奖励曲线
    plt.plot(sticks, ppo, color='blue', linewidth=3, label="PPO 筛选")
    plt.plot(sticks, ping_s, color='orange', linewidth=3, label="最优选择+串行探测")
    plt.plot(sticks, ping_p, color='pink', linewidth=3, label="最优选择+并行探测")

    # plt.ylim(0, 1000)
    # plt.xlim(0, 1000)
    # plt.yscale('log') 

    # 添加图例和坐标轴标签
    plt.legend(loc='upper left', fontsize=25)
    plt.xlabel("服务器数量", fontsize=35, weight='bold')
    plt.ylabel("调度完成时间 (ms)", fontsize=35, weight='bold')

    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)

    # 保存图像到指定位置
    plt.savefig("./results/time.png")
    plt.close()