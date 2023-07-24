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

    def step(self):
        pass
    
    def select_storage(self, task: Task, candidates: list):
        # should judge if the two nodes are from the same CSP
        pass