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
            
            es_sw = social_welfare(task, node)
            if es_sw > maxx:
                maxx = es_sw
                target_p = node
        
        if target_p and cand_exist:
            task.set_provider(target_p)
            return 1
        else:
            task.drop()
            return 0
    
    def search_candidates(self, task: Task):
        provider = task.provider()
        candidates = self.config['vm_database'][task.sid].get_servers()
        priorities = []
        for node in candidates:
            es_sw = social_welfare(task, provider, node)
            priorities.append(es_sw)
        
        # sort
        # 这个排序方法只有当被排序的数组中各元素不同才能用
        sorted_cand = sorted(candidates, key=lambda x: -priorities[candidates.index(x)])
        
        # padding
        while len(sorted_cand) < cand_num:
            sorted_cand.append(NullNode())
        
        # cutting
        cand_num = self.config['cand_num']
        if len(sorted_cand) > cand_num:
            sorted_cand = sorted_cand[0:cand_num]
        
        return sorted_cand
    
    def inform_candidates(self, task: Task, candidates: list):
        provider = task.provider()
        storage = provider.select_storage(candidates)
        if not storage:
            task.drop()
            return 0
        
        task.set_storage(storage)
        return 1