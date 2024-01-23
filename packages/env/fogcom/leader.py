import numpy as np
import time
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
                self.config['vm_database'][vmid].add_server(node)
        # make sure each vm has at least one host server
        for vm in self.config['vm_database']:
            if len(vm.get_servers()) == 0:
                server_id = np.random.randint(0, self.config['N_m'])
                server = self.servers[server_id]
                vm.add_server(server)
    
    def assign_provider(self, task: Task):
        """Assign a proper provider for the input task.

        Args:
            task (Task)
        
        Returns:
            int: 0 for false, 1 for success
        """
        # T=[]
        # T.append(time.time())
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
            # T.append(time.time())
            es_sw = self.social_welfare(task, node)  # TODO: 用掉了 0.01 s 左右
            # T.append(time.time())
            if es_sw > maxx:
                maxx = es_sw
                target_p = node
        
        # print("=================")
        # for i in range(len(T)-1):
        #     print(i, ":", T[i+1]-T[i])
        # print("=================")
        
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
            es_sw = self.social_welfare(task, provider, node)
            priorities.append(es_sw)
        
        # sort
        # 这个排序方法只有当被排序的数组中各元素不同才能用
        sorted_cand = sorted(candidates, key=lambda x: -priorities[candidates.index(x)])
        
        # padding
        cand_num = self.config['cand_num']
        while len(sorted_cand) < cand_num:
            sorted_cand.append(NullNode())
        
        # cutting
        if len(sorted_cand) > cand_num:
            sorted_cand = sorted_cand[0:cand_num]
        
        return sorted_cand
    
    def inform_candidates(self, task: Task, candidates: list):
        provider: Server = task.provider()
        storage = provider.select_storage(task, candidates)
        if not storage:
            task.drop()
            return 0
        
        task.set_storage(storage)
        return 1

    def social_welfare(self, task: Task, provider: Node, storage: Node = None, estimate=True):
        user = task.user()
        task_s = task.s
        task_w = task.w
        c_p = provider.c
        block_size = self.config["block_size"]
        result_size = self.config["result_size"]
        bw_fd, lt_fd = 0., 0.
    
        if estimate:
            bw_uf, lt_uf = self.config['link_check'].estimate(user, provider)
            if storage:
                bw_fd, lt_fd = self.config['link_check'].estimate(provider, storage)
        else:
            bw_uf, lt_uf = self.config['link_check'].check(user, provider)
            if storage:
                bw_fd, lt_fd = self.config['link_check'].check(provider, storage)
        
        p_c = provider.p_c
        p_link = provider.p_link
        p_s = provider.p_s
        p_vm = 0.
        rd_d = 0.
        if storage:
            p_vm = storage.p_vm
            rd_d = storage.rd
        
        dt = delta_t(task_s, task_w, c_p, rd_d, block_size, result_size, bw_uf, lt_uf, bw_fd, lt_fd)
        u = task.value(dt)
        c = cost_task(p_c, p_link, p_s, p_vm, task_s, task_w, c_p, rd_d, block_size, result_size, bw_uf, lt_uf, bw_fd, lt_fd)
        
        return u - c
        