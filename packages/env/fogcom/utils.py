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
            bw = np.random.randint(1, 50) / 100.    # [0.1, 50.] MBps
            lt = np.random.randint(1, 300) * 1e-3   # [1, 200] * 1e-3 s
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
    return sid

##### cost

def cost_task(p_c, p_link, p_s, p_vm, task_s, task_w, c_p, rd_d, block_size, result_size, bw_uf, lt_uf, bw_fd, lt_fd):
    c1 = cost_task_1(p_c, p_link, p_s, task_s, task_w, c_p, rd_d, block_size, result_size, bw_uf, lt_uf, bw_fd, lt_fd)
    c2 = cost_task_2(p_vm, task_w, c_p, rd_d, block_size, bw_fd, lt_fd)
    return c1 + c2

def cost_task_1(p_c, p_link, p_s, task_s, task_w, c_p, rd_d, block_size, result_size, bw_uf, lt_uf, bw_fd, lt_fd):
    ans = p_c * task_w +\
            p_link * (t_u(task_s, bw_uf, lt_uf) + t_vm(block_size, rd_d, bw_fd, lt_fd) + t_d(result_size, bw_uf, lt_uf)) +\
            p_s * task_s * (t_vm(block_size, rd_d, bw_fd, lt_fd) + t_p(task_w, c_p)) +\
            p_s * block_size * t_p(task_w, c_p) +\
            p_s * result_size * t_d(result_size, bw_uf, lt_uf)
    return ans

def cost_task_2(p_vm, task_w, c_p, rd_d, block_size, bw_fd, lt_fd):
    # Assume the storage usage price is still t_p when provider == storage
    ans = p_vm * (t_vm(block_size, rd_d, bw_fd, lt_fd) + t_p(task_w, c_p))
    return ans

##### time

def delta_t(task_s, task_w, c_p, rd_d, block_size, result_size, bw_uf, lt_uf, bw_fd, lt_fd):
    return t_u(task_s, bw_uf, lt_uf) + t_p(task_w, c_p) + t_d(result_size, bw_uf, lt_uf) + t_vm(block_size, rd_d, bw_fd, lt_fd) 

def t_u(task_s: float, bw_uf: float, lt_uf: float):
    # 在调用的地方判断两个节点是否是一个 -> 0., 或者传入的链路 bw 为 0
    if bw_uf < 1e-6:
        return 0.
    return task_s / bw_uf + lt_uf

def t_vm(block_size: float, rd_d: float, bw_fd: float, lt_fd: float):
    if bw_fd < 1e-6 or rd_d < 1e-6:
        return 0.
    return block_size / min(bw_fd, rd_d) + lt_fd

def t_p(task_w: float, c_p: float):
    if c_p < 1e-6:
        return 1e6  # means this provider has no free computing resource
    return task_w / c_p

def t_d(result_size: float, bw_uf: float, lt_uf: float):
    if bw_uf < 1e-6:
        return 0.
    ans = result_size / bw_uf + lt_uf
    return ans