import numpy as np
import time
from .node import Node
from .task import *

class LinkCheck(object):
    def __init__(self):
        self.database = {}  # key: f"{a.id},{b.id}" or f"{b.id},{a.id}"; value: {"bw": float, "lt": float}.
        
    def check(self, a: Node, b: Node):
        """Get the link state between node a and b.

        Args:
            a (Node)
            b (Node)

        Returns:
            [bandwidth (MBps), latency (s)]
        """
        
        key1 = f"{a.id},{b.id}"
        key2 = f"{b.id},{a.id}"
        if key1 in self.database.keys():
            bw = self.database[key1]["bw"]
            lt = self.database[key1]["lt"]
        else:
            # no exist, create a new link
            bw = np.random.randint(1, 100) / 10.    # [0.1, 10.] MBps
            lt = np.random.randint(1, 100) * 1e-3   # [1, 100] * 1e-3 s
            value = {"bw": bw, "lt": lt}
            self.database[key1] = value
            self.database[key2] = value
        
        bw = min(bw, a.bw)
        bw = min(bw, b.bw)
        lt += a.lt + b.lt
        return bw, lt

    def estimate(self, a: Node, b: Node):
        """Estimate the link state between node a and b.

        Args:
            a (Node)
            b (Node)

        Returns:
            [bandwidth (MBps), latency (s)]
        """
        bw = min(a.bw, b.bw)
        lt = a.lt + b.lt
        return bw, lt

##### mapping

def R(config, sid):
    vmid = sid
    return vmid

##### cost

def cost_task(task: Task, provider: Node, storage: Node = None, estimate=True):
    ans = cost_task_1(task, provider, storage, estimate)
    if storage:
        ans += cost_task_2(task, provider, storage, estimate)
    return ans

def cost_task_1(task: Task, provider: Node, storage: Node = None, estimate=True):
    conf = provider.config
    vmid = R(conf, task.sid)
    vm: VM = conf['vm_database'][vmid]
    if storage:
        ans = provider.p_c * task.w +\
                provider.p_link * (t_u(task, provider, estimate) + t_vm(task, provider, storage, estimate) + t_d(task, provider, estimate)) +\
                provider.p_s * task.s * (t_vm(task, provider, storage, estimate) + t_p(task, provider)) +\
                provider.p_s * vm.block_size * t_p(task, provider) +\
                provider.p_s * conf['result_size'] * t_d(task, provider, estimate)
    else:
        # used by the leader to choose a provider
        ans = provider.p_c * task.w +\
                provider.p_link * (t_u(task, provider, estimate) + t_d(task, provider, estimate)) +\
                provider.p_s * task.s * t_p(task, provider) +\
                provider.p_s * vm.block_size * t_p(task, provider) +\
                provider.p_s * conf['result_size'] * t_d(task, provider, estimate)
    return ans

def cost_task_2(task: Task, provider: Node, storage: Node, estimate=True):
    # Assume the storage usage price is still t_p when provider == storage
    ans = storage.p_vm * (t_vm(task, provider, storage, estimate) + t_p(task, provider))
    return ans

##### time

def delta_t(task: Task, provider: Node, storage: Node = None, estimate=True):
    ans = t_u(task, provider, estimate) + t_p(task, provider) + t_d(task, provider, estimate)
    if storage:
        ans += t_vm(task, provider, storage, estimate) 
    return ans

def t_u(task: Task, provider: Node, estimate=True):
    # TODO: 问题出在调用 config 和 task 的时候非常慢
    if task.user() == provider:
        return 0.
    
    conf = provider.config
    user = task.user()
    if estimate:
        bw_uf, lt_uf = conf['link_check'].estimate(user, provider)
    else:
        bw_uf, lt_uf = conf['link_check'].check(user, provider)
    ans = task.s / bw_uf + lt_uf
    return ans

def t_vm(task: Task, provider: Node, storage: Node, estimate=True):
    if provider == storage:
        return 0.

    conf = provider.config
    vmid = R(conf, task.sid)
    vm: VM = conf['vm_database'][vmid]
    if estimate:
        bw_fd, lt_fd = conf['link_check'].estimate(provider, storage)
    else:
        bw_fd, lt_fd = conf['link_check'].check(provider, storage)
    rd_d = storage.rd
    ans = vm.block_size / min(bw_fd, rd_d) + lt_fd
    return ans

def t_p(task: Task, provider: Node):
    ans = task.w / provider.c
    return ans

def t_d(task: Task, provider: Node, estimate=True):
    if task.user() == provider:
        return 0.
    
    conf = provider.config
    user = task.user()
    if estimate:
        bw_uf, lt_uf = conf['link_check'].estimate(user, provider)
    else:
        bw_uf, lt_uf = conf['link_check'].check(user, provider)
    ans = conf['result_size'] / bw_uf + lt_uf
    return ans

##### social welfare

def social_welfare(task: Task, provider: Node, storage: Node = None, estimate=True):
    u = utility(task, provider, storage, estimate)    # TODO: 用掉了 0.005 s 左右
    c = cost_task(task, provider, storage, estimate)  # TODO: 用掉了 0.005 s 左右
    ans = u - c
    return ans

def utility(task: Task, provider: Node, storage: Node = None, estimate=True):
    dt = delta_t(task, provider, storage, estimate) # TODO: 用掉了 0.005 s 左右
    ans = task.value(dt)
    return ans