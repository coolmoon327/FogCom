import numpy as np
import copy
from .node import Node

class VM(object):
    def __init__(self, id, size=-1.):
        self.id = id
        if size == -1.:
            # size = np.random.randint(1000, 10000) * 0.1   # [100., 1000.] MB
            size = 1000
        self.size = size
        # self.block_size = self.size * np.random.randint(1, 2) / 100. # [1, 20.] MB, [50, 100] blocks per vm
        self.block_size = self.size *  1/100.

        self.servers = []
        
    def add_server(self, server: Node):
        """Record the server instance of a new server storing this vm.

        Args:
            server (int): server instance
        """
        if server not in self.servers:
            self.servers.append(server)
    
    def get_servers(self):
        """Get server instances storing this vm.

        Returns:
            list[Node]: the list of server instances
        """
        return self.servers

class Task(object):
    total_tasks=0
    
    def __init__(self, t, s, w, sid, b0, alpha):
        self.id = Task.total_tasks
        Task.total_tasks += 1
        
        self.t = t
        self.s = s
        self.w = w
        self.sid = sid
        # b(x) = b0 - alpha * x
        self.b0 = b0
        self.alpha = alpha
        
        self.drpped = False
    
    def value(self, dt):
        ans = self.b0 - self.alpha * dt
        return ans
    
    b = value   # alias that keeps consistent with the paper
    
    def set_duration(self, duration):
        """Set how long this task waits to be finished.

        Args:
            duration (float): the time duration between arrived and finished
        """
        self._duration = duration
    
    def check_finished(self, slot_length, cslot):
        """Check whether this task is finished at the beginning of this slot.

        Args:
            slot_length (float): the length of a slot (s)
            cslot (int): the number of current slot

        Returns:
            Bool: True for finished, False for unfinished
        """
        if not hasattr(self, '_duration'):
            print(f"Unset property: \'_duration\' in task {self.id}.")
            return False
        
        if (cslot - self.t) * slot_length >= self._duration:
            return True
        else:
            return False
    
    def release(self):
        """Relase this task after finished."""
        # TODO: 使用更复杂的环境时(如 storage 需要考虑同时最多服务的对象数量), 需要修改此处
        if hasattr(self, '_provider'):
            self._provider.occupied = False
    
    def set_user(self, user: Node):
        self._user = user
    
    def set_provider(self, provider: Node):
        self._provider = provider
        provider.occupied = True
    
    def set_storage(self, storage: Node):
        # TODO: 使用更复杂的环境时(如 storage 需要考虑同时最多服务的对象数量), 需要修改此处
        self._storage = storage
    
    def user(self):
        if not hasattr(self, '_user'):
            print(f"Unset property: \'_user\' in task {self.id}.")
            return
        return self._user
    
    def provider(self):
        if not hasattr(self, '_provider'):
            print(f"Unset property: \'_provider\' in task {self.id}.")
            return
        return self._provider
    
    def storage(self):
        if not hasattr(self, '_storage'):
            print(f"Unset property: \'_storage\' in task {self.id}.")
            return
        return self._storage
    
    def drop(self):
        self.dropped = True
        self.release()
    