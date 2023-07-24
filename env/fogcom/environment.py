import numpy as np
from gym import spaces
from .utils import *
from .leader import *
from .server import *
from .user import *

class Environment(object):
    def __init__(self, config={}):
        self.config = config
        self.generate_topology()
        self.reset()
    
    def reset(self):
        self.config['n_slot'] = 0   # used to synchronize the number of slots among different py files in this module
        
        for node in self.users + self.servers:
            node.reset()
    
    def generate_topology(self):
        self.link_check = LinkCheck()
        self.users: list[User] = []
        self.servers: list[Server] = []
        self.leader = Leader(self.config)

        for i in range(self.config['N_m']):
            self.servers.append(Server(id=i, config=self.config))
        
        offset = 100000 # the ID offset of user.id to prevent collision with server.id
        for i in range(self.config['N_u']):
            uid = i + offset
            self.users.append(User(id=uid, config=self.config))
        
        self.config['vm_database'] = [VM(i) for i in range(self.config['vm_num'])]
        self.leader.set_users(self.users)
        self.leader.set_servers(self.servers)
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def generate_topology(self):
        pass
    
    def step(self, action):
        self.config['n_slot'] += 1
        
        for node in self.users + self.servers:
            node.step()
            
        pass