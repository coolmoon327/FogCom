import numpy as np
from node import Node

class LinkCheck(object):
    def __init__(self):
        self.database = {}  # key: f"{a.id},{b.id}" or f"{b.id},{a.id}"; value: {"bw": float, "lt": float}.
        
    def check(self, a: Node, b: Node):
        key1 = f"{a.id},{b.id}"
        key2 = f"{b.id},{a.id}"
        if key1 in self.database.keys:
            bw = self.database["bw"]
            lt = self.database["lt"]
        else:
            # no exist, create a new link
            bw = np.random.randint(1, 100) / 10.    # [0.1, 10.] MBps
            lt = np.random.randint(1, 100) * 1e-3   # [1, 100] ms
            value = {"bw": bw, "lt": lt}
            self.database[key1] = value
            self.database[key2] = value
        
        bw = min(bw, a.bw)
        bw = min(bw, b.bw)
        lt += a.lt + b.lt
        return {"bw": bw, "lt": lt}
        
        