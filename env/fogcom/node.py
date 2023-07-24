import numpy as np

class Node(object):
    def __init__(self, id:int, config={}):
        self.config = config
        self.id = id
        
        self.c = np.random.randint(100, 1000) * 1e7            # [1e9, 1e10] cycle/s
        self.bw = np.random.randint(10, 500) / 10.             # [1, 50.] MBps
        self.lt =  np.random.randint(1, 100) * 1e-3            # [0.001, 0.1] s
        self.S = np.random.randint(100, 1000) * 1e-1           # [10., 100.] MB
        self.p_c = np.random.randint(10, 100) * 1e-8           # [1e-7, 1e-6] $/cycle
        self.p_link = np.random.randint(1000, 10000) * 0.1     # [100., 1000.] $/s
        self.p_s = np.random.randint(100, 500) * 0.1           # [10., 50.] $/(MB*s)
        
        self.rd = np.random.randint(100, 2000) / 20.           # [5., 100.] MB/s
        self.p_vm = np.random.randint(1000, 10000) * 0.1       # [100., 1000.] $/s
        self.reset()
    
    def reset(self):
        pass
