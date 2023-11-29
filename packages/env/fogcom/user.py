import numpy as np
from .node import *
from .task import *

class User(Node):
    def __init__(self, id:int, config={}):
        super().__init__(id, config)
        self.reset()
    
    def reset(self):
        super().reset()
        self.tasks = []

    def generate_task(self):
        t = self.config['n_slot']
        s = np.random.randint(100, 200) * 0.1       # [10., 20.] MB
        w = 5 * np.random.randint(100, 1000) * 1e6  # [5e8, 5e9] cycles
        sid = np.random.randint(0, self.config['vm_num'])
        b0 = 12500 #np.random.randint(5000, 15000)
        alpha = np.random.randint(500, 1000) * 1.   # [500, 1000]
        task = Task(t, s, w, sid, b0, alpha)
        task.set_user(self)
        return task
    
    def next_slot(self):
        self.tasks.clear()
        
        r = np.random.randint(0, 10000) / 10000.
        n_tasks_per_user_in_slot = self.config['task_freq'] * self.config['slot_length'] / self.config['N_u']
        while n_tasks_per_user_in_slot >= 1.:
            # more than 1, directly add
            n_tasks_per_user_in_slot -= 1.
            self.tasks.append(self.generate_task())
        if r < n_tasks_per_user_in_slot:
            # less than 1, means possibility
            self.tasks.append(self.generate_task())