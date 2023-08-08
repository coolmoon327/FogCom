import numpy as np
from .node import Node
from .task import *
from .utils import *

class Server(Node):
    def __init__(self, id:int, config={}):
        super().__init__(id, config)
        self.reset()
    
    def reset(self):
        # 1. set properties
        super().reset()
        self.csp = np.random.randint(0, self.config['csp_num'])
        self.strategy = np.random.randint(0, self.config['follower_strategies_num'])
        
        # 2. set vm storage
        self.vms = []
        rn = np.random.randint(1, 10)
        while rn:
            rn -= 1
            rid = np.random.randint(0, self.config['vm_num'])
            while rid in self.vms:
                rid = np.random.randint(0, self.config['vm_num'])
            self.vms.append(rid)
        
        # 3. set known links
        # Notably, a server knows all the nodes from its CSP. We handle this condition in select_storage func.
        # assuming each node has visited 20% nodes in history
        self.is_known = [True if np.random.randint(0, 10) < 3 else False for _ in range(self.config['N_m'])]  # index: server_id, value: known or not
        
        # 4. set states
        self.occupied = False

    def next_slot(self):
        pass
    
    def is_estimate(self, node: Node):
        estimate = True
        if hasattr(node, 'csp'):
            if self.csp == node.csp:
                estimate = False
        if self.is_known[node.id]:
            estimate = False
        return estimate
    
    def bias(self, task: Task, storage: Node):
        estimate = self.is_estimate(storage)
        if self.strategy == 0:
            ans = 0
        elif self.strategy == 1:
            ans = t_vm(task, self, storage, estimate)
        elif self.strategy == 2:
            ans = - t_vm(task, self, storage, estimate)
        elif self.strategy == 3:
            ans = self.config['M'] if self.csp == storage.csp else 0
        else:
            ans = 0
        ans *= self.config['beta']
        return ans
    
    def cost(self, task: Task, storage: Node):
        estimate = self.is_estimate(storage)
        ans = cost_task(task, self, storage, estimate)
        return ans
    
    def price(self, task: Task, storage: Node):
        estimate = self.is_estimate(storage)
        dt = delta_t(task, self, storage, estimate)
        ans = self.config['alpha'] * task.value(dt)
        return ans
    
    def select_storage(self, task: Task, candidates: list):
        maxx = 0.
        target_s = None
        for node in candidates:
            if node.is_Null():
                continue
            obj = self.price(task, node) - self.cost(task, node) + self.bias(task, node)
            if obj > maxx:
                maxx = obj
                target_s = node
        
        return target_s