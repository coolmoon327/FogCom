import numpy as np

class VM(object):
    def __init__(self, id, size=-1.):
        self.id = id
        if size == -1.:
            size = np.random.randint(1000, 10000) * 0.1   # [100., 1000.] MB
        self.size = size
        self.block_size = self.size * np.random.randint(1, 2) / 100. # [1, 20.] MB, [50, 100] blocks per vm

        self.servers = []
        
    def add_server(self, server_id):
        """Record the ID of a new server storing this vm

        Args:
            server_id (int): server ID
        """
        if server_id not in self.servers:
            self.servers.append(server_id)
    
    def get_servers(self):
        """Get IDs of servers storing this vm

        Returns:
            list[int]: the list of server IDs
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
    
    def vmid(self):
        return self.sid
    
    def set_user_id(self, user_id):
        self.user_id = user_id
    
    def set_server_id(self, server_id):
        self.server_id = server_id
    
    def set_storage_id(self, storage_id):
        self.storage_id = storage_id
    
    def get_user_id(self):
        if not hasattr(self, 'user_id'):
            print(f"Unset property: \'user_id\' in task {self.id}.")
            return -1
        return self.user_id
    
    def get_server_id(self):
        if not hasattr(self, 'server_id'):
            print(f"Unset property: \'server_id\' in task {self.id}.")
            return -1
        return self.server_id
    
    def get_storage_id(self):
        if not hasattr(self, 'storage_id'):
            print(f"Unset property: \'storage_id\' in task {self.id}.")
            return -1
        return self.storage_id
    