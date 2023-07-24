import numpy as np
from .node import Node

class Server(Node):
    def __init__(self, id:int, config={}):
        super().__init__(id, config)
        self.reset()
    
    def reset(self):
        super().reset()
        
        self.vms = []
        rn = np.random.randint(1, 10)
        while rn:
            rn -= 1
            rid = np.random.randint(0, self.config['vm_num'])
            while rid in self.vms:
                rid = np.random.randint(0, self.config['vm_num'])
            self.vms.append(rid)

    def step(self):
        pass