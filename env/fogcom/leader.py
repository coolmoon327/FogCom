import numpy as np
from .user import *
from .server import *

class Leader(object):
    def __init__(self, config={}):
        self.config = config
    
    def set_users(self, users: list):
        self.users = users
    
    def set_servers(self, servers: list):
        self.servers = servers

        for node in servers:
            for vmid in node.vms:
                self.config['vm_database'][vmid].add_server(node.id)