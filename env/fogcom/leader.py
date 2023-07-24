import numpy as np
from .user import *
from .server import *
from .task import *
from .utils import *

class Leader(object):
    def __init__(self, config={}):
        self.config = config
    
    def set_users(self, users: list):
        self.users: list[User] = users
    
    def set_servers(self, servers: list):
        self.servers: list[Server] = servers

        for node in servers:
            for vmid in node.vms:
                self.config['vm_database'][vmid].add_server(node.id)
    
    def assign_provider(self, task: Task):
        """Assign a proper provider for the input task.

        Args:
            task (Task)
        
        Returns:
            int: 0 for false, 1 for success
        """
        # choose a provider by maximizing the estimated SW
        maxx = 0.
        target_p = None
        cand_exist = False
        for node in self.servers:
            vmid = R(self.config, task.sid)
            if vmid in node.vms:
                # TODO: 使用更复杂的环境时(如 storage 需要考虑同时最多服务的对象数量), 需要修改此处
                cand_exist = True
            if node.occupied:
                continue
            vm: VM = self.config['vm_database'][vmid]
            if task.s + vm.block_size > node.S:
                continue
            
            dt = delta_t()
            u = task.value(dt)
            c = cost_task(task, node)
            sw = u - c
            if sw > maxx:
                maxx = sw
                target_p = node
        
        if target_p and cand_exist:
            task.set_provider(target_p)
            return 1
        else:
            return 0