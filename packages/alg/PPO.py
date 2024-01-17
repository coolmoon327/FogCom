import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import torch.multiprocessing as mp
from ..env.wrapper import EnvWrapper
from .eval import Evaluator
from .draw import ResultCurve
import openpyxl

class ActorPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action


class CriticPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


class Config:  # for on-policy
    def __init__(self, agent_class=None, env_class=None, env_args=None):
        self.agent_class = agent_class  # agent = agent_class(...)
        self.if_off_policy = False  # whether off-policy or on-policy of DRL algorithm

        self.env_class = env_class  # env = env_class(**env_args)
        self.env_args = env_args  # env = env_class(**env_args)
        if env_args is None:  # dummy env_args
            env_args = {'env_name': None, 'state_dim': None, 'action_dim': None, 'if_discrete': None}
        self.env_name = env_args['env_name']  # the name of environment. Be used to set 'cwd'.
        self.state_dim = env_args['state_dim']  # vector dimension (feature number) of state
        self.action_dim = env_args['action_dim']  # vector dimension (feature number) of action
        self.if_discrete = env_args['if_discrete']  # discrete or continuous action space

        '''Arguments for reward shaping'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 1.0  # an approximate target reward usually be closed to 256

        '''Arguments for training'''
        self.net_dims = (512, 128, 32)  # the middle layer dimension of MLP (MultiLayer Perceptron)
        self.learning_rate = 1e-7  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 5e-3  # 2 ** -8 ~= 5e-3
        self.batch_size = int(512)  # num of transitions sampled from replay buffer, default 128
        self.horizon_len = int(500)  # collect horizon_len step while exploring, then update network, default 2000
        self.buffer_size = None  # ReplayBuffer size. Empty the ReplayBuffer for on-policy.
        self.repeat_times = 8.0  # repeatedly update network using ReplayBuffer to keep critic's loss small, default 8.0

        '''Arguments for device'''
        self.gpu_id = int(0)  # `int` means the ID of single GPU, -1 means CPU
        self.thread_num = int(8)  # cpu_num for pytorch, `torch.set_num_threads(self.num_threads)`
        self.random_seed = int(0)  # initialize random seed in self.init_before_training()

        '''Arguments for evaluate'''
        self.cwd = None  # current working directory to save model. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = +np.inf  # break training if 'total_step > break_step'

        self.eval_times = int(10000)  # number of times that get episodic cumulative return, default 32
        self.eval_per_step = int(1)  # evaluate the agent per training steps, default 2e4
        
        '''Dict from config.yml'''
        self.env_config = []

    def init_before_training(self):
        if self.cwd is None:  # set cwd (current working directory) for saving model
            self.cwd = f'./{self.env_name}_{self.agent_class.__name__[5:]}'
        os.makedirs(self.cwd, exist_ok=True)

class AgentBase:
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.repeat_times = args.repeat_times
        self.reward_scale = args.reward_scale
        self.learning_rate = args.learning_rate
        self.if_off_policy = args.if_off_policy
        self.soft_update_tau = args.soft_update_tau

        self.last_state = []  # save the last state of the trajectory for training. `last_state.shape == (thread_num, state_dim)`
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = self.act_target = act_class(net_dims, state_dim, action_dim).to(self.device)
        self.cri = self.cri_target = cri_class(net_dims, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        self.criterion = torch.nn.SmoothL1Loss()

    @staticmethod
    def optimizer_update(optimizer, objective: Tensor):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        # assert target_net is not current_net
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


class AgentPPO(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

    def explore_env(self, env, horizon_len: int) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.last_state

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action, logprob = [t.squeeze(0) for t in get_action(state.unsqueeze(0))[:2]]

            ary_action = convert(action).detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = reward
            dones[i] = done

        self.last_state = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> [float]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            if states.device != self.device:
                states = states.to(self.device)
                actions = actions.to(self.device)
                logprobs = logprobs.to(self.device)
                rewards = rewards.to(self.device)
                undones = undones.to(self.device)
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 9  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0
        
        # print("开始训练")
        
        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = torch.tensor(self.last_state, dtype=torch.float32).to(self.device)
        next_value = self.cri(next_state.unsqueeze(0)).detach().squeeze(1).squeeze(0)

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages

def explore_and_store_result(agent, env, act_shared_model, act_target_shared_model, cri_shared_model, cri_target_shared_model, horizon_len):
    agent.act.load_state_dict(act_shared_model)
    agent.act_target.load_state_dict(act_target_shared_model)
    agent.cri.load_state_dict(cri_shared_model)
    agent.cri_target.load_state_dict(cri_target_shared_model)
    
    buffer_items = agent.explore_env(env, horizon_len)
    ret = []
    for item in buffer_items:
        ret.append(item.cpu().detach())
    return ret

def train_agent(args: Config, threads_num, result_list, lock):
    args.init_before_training()

    envs = [args.env_class(args.env_config) for _ in range(max(threads_num,1))]
    agents = [args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args) for _ in range(max(threads_num,1))]
    
    act_grad_file = './results/act_grad.pth'
    cri_grad_file = './results/cri_grad.pth'
    if os.path.exists(act_grad_file) and os.path.exists(cri_grad_file):
        print("Loaded model from files.")
        # 从文件中加载梯度
        act_shared_model = torch.load(act_grad_file)
        act_target_shared_model = copy.deepcopy(act_shared_model)
        cri_shared_model = torch.load(cri_grad_file)
        cri_target_shared_model = copy.deepcopy(cri_shared_model)
    else:
        # 使用 agents[0] 的梯度
        act_shared_model = agents[0].act.state_dict()
        act_target_shared_model = agents[0].act_target.state_dict()
        cri_shared_model = agents[0].cri.state_dict()
        cri_target_shared_model = agents[0].cri_target.state_dict()
    
    
    for i in range(max(threads_num,1)):
        envs[i].tag = i
        agents[i].last_state = envs[i].reset()

    evaluator = Evaluator(eval_env=args.env_class(args.env_config),
                          eval_per_step=args.eval_per_step,
                          eval_times=args.eval_times,
                          cwd=args.cwd)
    evaluator.agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    
    torch.set_grad_enabled(False)
    
    # 创建 evaluator 的进程池
    eval_pool = mp.Pool(processes=2)
    eval_result = None
    
    while True:  # start training
        if threads_num == 0:
            buffer_items = agents[0].explore_env(envs[0], args.horizon_len)
        else:
            # 创建 explorer 的进程池
            expl_pool = mp.Pool(processes=threads_num)
            
            # 提交任务并收集返回值
            results = []
            for i in range(threads_num):
                result = expl_pool.apply_async(explore_and_store_result, args=(agents[i], envs[i], act_shared_model, act_target_shared_model, cri_shared_model, cri_target_shared_model, args.horizon_len))
                results.append(result)
                        
            # 等待所有进程完成并获取返回值
            expl_pool.close()
            expl_pool.join()

            outputs = []

            exception_occurred = False
            for i, result in enumerate(results):
                try:
                    output = result.get(timeout=1)  # 设置超时时间为1秒
                    if not result.successful():
                        print(f"Process {i} did not complete successfully.")
                        exception_occurred = True
                    else:
                        outputs.append(output)
                except Exception as e:
                    print(f"Exception occurred in process {i}: {e}")
                    exception_occurred = True

            if exception_occurred:
                print("Some processes encountered exceptions.")

            N = len(outputs[0])
            buffer_items = [None] * N
            for i in range(N):
                buffer_items[i] = torch.cat([output[i] for output in outputs], dim=0)
            
            # 内存释放
            del results
            del outputs
            
        torch.set_grad_enabled(True)
        logging_tuple = agents[0].update_net(buffer_items)
        torch.set_grad_enabled(False)
        act_shared_model = agents[0].act.state_dict()
        act_target_shared_model = agents[0].act_target.state_dict()
        cri_shared_model = agents[0].cri.state_dict()
        cri_target_shared_model = agents[0].cri_target.state_dict()

        evaluator.total_step += args.horizon_len * threads_num
        
        if eval_result is None or eval_result.ready():
            eval_result = evaluate_and_save(eval_pool, evaluator, agents[0], logging_tuple)

            file_path = "./results/output.xlsx"
            if os.path.exists(file_path):
                workbook = openpyxl.load_workbook(file_path)
                sheet = workbook.active
                # 读取 SW 数据（SW 数据在第 9 列）
                sw_data = []
                for row in sheet.iter_rows(min_row=2, values_only=True):
                    sw_value = row[8]  # 第 9 列的索引为 8
                    sw_data.append(sw_value)
                curve = ResultCurve()
                curve.set_results(sw_data)
                curve.save_plot("./results/reward_curve.png")
            
        # evaluator.evaluate_and_save(agents[0].act, args.horizon_len * threads_num, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            break  # stop training when reach `break_step` or `mkdir cwd/stop`

        # 内存释放
        del buffer_items

def evaluate_and_save(pool, evaluator, agent, logging_tuple, save=True):
    evaluator.eval_step = evaluator.total_step
    evaluator.agent.act.load_state_dict(agent.act.state_dict())
    eval_result = pool.apply_async(evaluator.evaluate_and_save, args=(logging_tuple,))
    
    if save:
        if not os.path.exists("./results"):
            os.makedirs("./results")
        torch.save(agent.act.state_dict(), './results/act_grad.pth')
        torch.save(agent.cri.state_dict(), './results/cri_grad.pth')
    
    return eval_result

def set_args(config):
    agent_class = AgentPPO  # DRL algorithm name
    env_class = EnvWrapper
    env_instance = env_class(config)
    env_args = {
        'env_name': env_instance.env_name,  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': env_instance.state_dim,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': env_instance.action_dim,  # the torque applied to free end of the pendulum
        'if_discrete': env_instance.if_discrete  # continuous action space, symbols → direction, value → force
    }
    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.env_config = config
    args.batch_size = config['batch_size']
    args.break_step = config['break_step']  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = config['gamma']  # discount factor of future rewards
    args.repeat_times = config['repeat_times']  # repeatedly update network using ReplayBuffer to keep critic's loss small
    return args

def train_ppo_for_fogcom(config, threads_num, result_list, lock):
    args = set_args(config)
    train_agent(args, threads_num, result_list, lock)

def test(config, test_times=1):
    config['penalty'] = 0.
    config['ganrantee_policy'] = False
    if config['ganrantee_policy']:
        print("允许切换算法")
    args = set_args(config)
    
    act_grad_file = './results/act_grad.pth'
    
    evaluator = Evaluator(eval_env=EnvWrapper(config), eval_times=1000000)
    evaluator.agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    
    evaluator.total_step = 111
    logging_tuple = (0., 0.)

    for i in range(test_times):
        act_model = torch.load(act_grad_file)
        evaluator.agent.act.load_state_dict(act_model)

        evaluator.evaluate_and_save(logging_tuple)
        evaluator.total_step += 1