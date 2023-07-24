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
        
        self.tasks: list[Task] = []     # all tasks, used for logging or analyzing
        self.act_tasks: list[Task]  =[]  # tasks still in execution
        self.new_tasks: list[Task]  = [] # tasks just generated in this new slot
        self.task_index = 0
        
        for node in self.users + self.servers:
            node.reset()
    
    def seed(self, seed):
        np.random.seed(seed)
    
    def generate_topology(self):
        self.config['link_check'] = LinkCheck()
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
    
    def next_slot(self):
        # 1. properties update
        self.config['n_slot'] += 1
        self.new_tasks.clear()
        self.task_index = 0
        
        # 2. node update
        for node in self.users + self.servers:
            node.step()
        
        # 3. new tasks
        for node in self.users:
            if len(node.tasks):
                self.tasks += node.tasks
                self.new_tasks += node.tasks
        
        # 4. executing tasks management
        release_list = []
        for task in self.act_tasks:
            if task.check_finished:
                task.release()
                release_list.append(task)
        for task in release_list:
            self.act_tasks.remove(task)
    
    def next_task(self):
        # 1. if no new task, entering a new slot
        while self.task_index == len(self.new_tasks):
            self.next_slot()
        
        # 2. get a new task
        task = self.new_tasks[self.task_index]
        self.task_index += 1
        
        # 3. assign a provider by estimating
        ret = self.leader.assign_provider(task)
        if not ret:
            # recursion
            return self.next_task()
        
        # 4. get all candidates
        pass
        
        # 5. generate state info
        state = []
        pass
    
        return state
    
    def step(self, action):
        # 1. execute action 
        pass
        
        # 2. update task info (storage, duration)
        pass
        
        # 3. calculate precise reward
        pass
        
        # 4. get next task (state)
        state = self.next_task()