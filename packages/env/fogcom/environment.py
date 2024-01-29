import numpy as np
from gym import spaces
from .utils import *
from .leader import *
from .server import *
from .user import *

class Environment(object):
    """
    ### Description
    
    This environment simulates a stackelberg game in a fog computing network.
    There are `N_u` users and `N_m` edge servers in the network, and edge servers will provide fog computing services for the users.
    Since each edge server has limited storage capacity, it only maintains parts of the `vm_num` virtual machines (VMs) and other VMs can be mounted from another server during service.
    In each time slot (of total `T` slots), some of the users will generate tasks and then inform the leader node (fog computing administrator).
    The leader node will select an end server to be the service provider and choose `cand_num` satisfied servers as storage candidates.
    Then these candidates' information will be packaged with the task information as the observation, and RL algorithm should make a decision based this observation to further select the candidates.
    The selected candidates will be informed to the service provider by the leader, after which the provider will decide the target storage node by its own strategy.

    ### Action Space
    
    The action is a `ndarray` with shape `(cand_num * 2,)`, in which each element will be mapped to `[0, 1]` indicating the drop rate or accept rate of its related node (based on the index).
    For example: 
        cand_num = 2, inputs = [10000., 0.1, 0., 0.21], 
        after mapping: [1., 0., 0., 1.], 
        action: node 0 is dropped and node 1 is accepted.
    
    **Note**: Sometimes there are less than `cand_num` valid candidates, and the rest are Null nodes (whose info is `(-1., -1., -1., -1.,)`). 
    If RL selects no valid node, it will be punished by the reward of `penalty`.
    
    ### Observation Space
    
    The observation is a `ndarray` with shape `(6+tag_len+4*cand_num,)` with the values corresponding to the information of task, provider, and candidates:
    
    | Num           | Observation                        | Min                  | Max                |
    |---------------|------------------------------------|----------------------|--------------------|
    | 0             | Task Size                          | -10.                 | 20.                |
    | 1             | Alpha in `B(t)=B_0-Alpha*t`        | 500.                 | 1000.              |
    | 2             | Price of Provider's Network Link   | 100.                 | 1000.              |
    | 3             | Price of Provider's Storage        | 10.                  | 50.                |
    | 4             | Bandwidth of Provider              | 1.                   | 50.                |
    | 5             | Latency of Provider                | 0.001                | 1.                 |
    | 6 ~ tag_len+5 | Tags of Provider's Strategy        | 0                    | 1                  |
    | tag_len+6     | Candidate 0's ISP = Provider's?    | 0                    | 1                  |
    | tag_len+7     | Price of Candidate 0's VM Service  | 100.                 | 1000.              |
    | tag_len+8     | Upload Speed of Candidate 0        | 1.                   | 50.                |
    | tag_len+9     | Latency of Candidate 0             | 0.001                | 1.                 |
    ...
    | 2+tag_len+4*cand_num | Candidate `cand_num`'s ISP = Provider's?   | 0     | 1                  |
    | 3+tag_len+4*cand_num | Price of Candidate `cand_num`'s VM Service | 100.  | 1000.              |
    | 4+tag_len+4*cand_num | Upload Speed of Candidate `cand_num`       | 1.    | 50.                |
    | 5+tag_len+4*cand_num | Latency of Candidate `cand_num`            | 0.001 | 1.                 |
    
    ### Rewards
    
    Since the RL algorithm is used as the policy of the leader node, the reward comes from Social Welfare which is the leader's optimize objective.
    Some parts of Social Welfare are unassociated with the policy, thus the reward contains only the related parts.
    
    Reward = - Alpha * t_vm - (p_link * t_vm + p_s * s * t_vm) - p_vm * t_vm
    
    The result of the above function is always negative, so the RL can use a term of moving average `M_AVG` to shape the reward when training the neuro network.
    
    Shaped_Reward = Reward - M_AVG
    
    ### Episode Termination
    
    The episode terminates if any one of the following occurs:
    1. Slot length is greater than `T`
    
    """
    
    def __init__(self, config={}):
        self.config = config
        self.generate_topology()
        self.reset()
        
        # Env Info
        self.env_name = 'FogComputing'
        
        # high = np.array(
        #     [20., 1000., 1000., 50., 50., 1.]
        #     +[config['N_m'] for _ in range(config['tag_len'])]
        #     +[1., 1000., 50., 1.,]*config['cand_num'],
        #     dtype=np.float32,
        # )

        high = np.array(
            [20., 1000., 1000., 50., 50., 1.]
            +[1. for _ in range(config['tag_len'])]
            +[1., 1000., 50., 1.,]*config['cand_num'],
            dtype=np.float32,
        )
        
        low = np.array(
            [-10., 500., 100., 10., 1., 0.001]
            +[0. for _ in range(config['tag_len'])]
            +[0., 100., 1., 0.001,]*config['cand_num'],
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # self.action_space = spaces.MultiBinary(config['cand_num'])
        act_low = np.array([-1e5 for _ in range(config['cand_num'] * 2)])
        act_high = np.array([1e5 for _ in range(config['cand_num'] * 2)])
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)
        
        # self.state_dim = 6 + config['tag_len'] + 4*config['cand_num']
        # self.action_dim = config['cand_num']
        self.state_dim = 6 + config['tag_len'] + 4*config['cand_num']
        self.action_dim = config['cand_num'] * 2
        self.if_discrete = False  # discrete action or continuous action
    
    def reset(self):
        self.config['n_slot'] = 0   # used to synchronize the number of slots among different py files in this module
        
        self.tasks: list[Task] = []     # all tasks, used for logging or analyzing
        self.act_tasks: list[Task]  =[]  # tasks still in execution
        self.new_tasks: list[Task]  = [] # tasks just generated in this new slot
        self.task_index = 0
        
        for node in self.users + self.servers:
            node.reset()
        
        # loggers
        self.drop_num = 0
        
        state = self.next_task()
        return state
    
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
            node.next_slot()
        
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
        while self.task_index >= len(self.new_tasks):
            self.next_slot()
        
        # 2. get a new task
        task = self.new_tasks[self.task_index]
        self.task_index += 1
        
        # 3. assign a provider by estimating
        ret = self.leader.assign_provider(task)
        if not ret:
            # self.drop_num += 1
            return self.next_task() # recursion
        provider: Node = task.provider()
        
        # 4. get all candidates
        self.raw_candidates = self.leader.search_candidates(task)
        if provider in self.raw_candidates:
            task.set_storage(provider)
            self.execute_task(task)
            return self.next_task() # recursion
        
        # 5. generate state info
        # 如果要对某组情况过拟合, 考虑加入节点编号
        # [s, alpha, p_link, p_s, bw, lt, tag[0], ..., tag[tag_len], is_same_csp[0], p_vm[0], min_bw_rd[0], lt[0], ...]
        # 6 + tag_len + 4 * cand_num
        state = []
        state.append(task.s)
        state.append(task.alpha)
        state.append(provider.p_link)
        state.append(provider.p_s)
        state.append(provider.bw)
        state.append(provider.lt)

        if self.config['force_worker_tag_in_state'] == -1:
            tag = provider.strategy
        else:
            tag = self.config['force_worker_tag_in_state']

        if tag == 0:
            state += [1., 0., 0., 0.]
        elif tag == 1:
            state += [0., 1., 0., 0.]
        elif tag == 2:
            state += [0., 0., 1., 0.]
        elif tag == 3:
            state += [0., 0., 0., 1.]
        else:
            state += [0., 0., 0., 0.]

        for node in self.raw_candidates:
            s = []
            if node.is_Null():
                s = [0., 1e6, 0., 1e3]
            else:
                s.append(1. if provider.csp == node.csp else 0.)
                s.append(node.p_vm)
                s.append(min(node.bw, node.rd))
                s.append(node.lt)
            state += s

        state = np.array(state)
        return state
    
    def execute_task(self, task: Task):
        provider: Server = task.provider()
        storage: Server = task.storage()
        
        # set task state
        real_duration = provider.delta_t(task, storage, False)
        task.set_duration(real_duration)
        
        # maintain active tasks list
        self.act_tasks.append(task)
        
        # log sw
        pass
    
    def execute_and_next(self, task, candidates):
        # 2. execute action  (storage, duration, act_tasks)
        ret = self.leader.inform_candidates(task, candidates)
        
        if not ret:
            self.drop_num += 1
            reward = self.config['penalty']
            sw = 0
        else:
            self.execute_task(task)
            # 3. calculate reward
            provider: Node = task.provider()
            storage: Node = task.storage()
            # Reward = - Alpha * t_vm - (p_link * t_vm + p_s * s * t_vm) - p_vm * t_vm
            reward = 10000 - (task.alpha + provider.p_link + provider.p_s * task.s + storage.p_vm) * provider.t_vm(task, storage, False) 
            sw = self.leader.social_welfare(task, provider, storage, False)
            # reward = sw

        # 4. get next task (state)
        state = self.next_task()
        
        terminal = self.config['n_slot'] >= self.config['T']
        info_dict = {"drop_num":self.drop_num, "sw":sw}
        return state, reward, terminal, info_dict
    
    def step_with_inner_policy(self, policy_id: int):
        # policy_id:
        # 0 - total random (can drop)
        # 1 - random
        # 2 - greedy
        # 3 - greedy (observe all)
        # 4 - all
        # 5 - all + just has follower with policy 0
        
        task = self.new_tasks[self.task_index-1]
        
        candidates = []
        
        if policy_id == 0:
            rid = np.random.randint(0, len(self.raw_candidates))
            candidates = [self.raw_candidates[rid]]
        elif policy_id == 1:
            len_not_null = 0
            while not self.raw_candidates[len_not_null].is_Null():
                len_not_null += 1
            rid = np.random.randint(0, len_not_null)
            candidates = [self.raw_candidates[rid]]
        elif policy_id == 2:
            candidates = [self.raw_candidates[0]]
        elif policy_id == 3:
            provider: Node = task.provider()
            maxx = -1e6
            for node in self.raw_candidates:
                if node.is_Null():
                    break
                sw = self.leader.social_welfare(task, provider, node, estimate=False)
                if sw >= maxx:
                    maxx = sw
                    candidates = [node]
        elif policy_id == 4 or policy_id == 5:
            len_not_null = 0
            while not self.raw_candidates[len_not_null].is_Null():
                len_not_null += 1
            candidates = self.raw_candidates[0:len_not_null]

        if policy_id == 5:
            task.provider().strategy = 0
            
        return self.execute_and_next(task, candidates)
    
    def step(self, action: np.ndarray):
        task = self.new_tasks[self.task_index-1]    # notice that the task_index indicates the latter task after the current task (seen in next_task 2.)
        
        # 1. translate action 
        candidates = []
        for i in range(int(len(action)/2)):
            inputs = action[i*2:i*2+2]
            is_selected = np.argmax(inputs) # TODO: 现在是取 max, 之后看情况改成根据概率来选
            if is_selected:
                candidates.append(self.raw_candidates[i])
        
        if not len(candidates) and self.config['ganrantee_policy']:
            candidates = [self.raw_candidates[0]]
        
        return self.execute_and_next(task, candidates)
        
        