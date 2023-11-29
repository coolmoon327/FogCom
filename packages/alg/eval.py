import torch
import time
import numpy as np
import os
import torch.multiprocessing as mp
from ..env.wrapper import EnvWrapper

total_steps = 1

class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.', print_head = True):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agent per training steps

        self.agent = None
        
        self.recorder = []
        if print_head:
            print(f"\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
                f"\n| `time`: Time spent from the start of training to this moment."
                f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
                f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
                f"\n| `avgS`: Average of steps in an episode."
                f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
                f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
                f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")

    def evaluate_and_save(self, logging_tuple: tuple):
        # print("开始测试")
        actor = self.agent.act
        
        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary)
        info = rewards_steps_ary[:, 2]
        rewards_steps_ary = np.array(rewards_steps_ary[:, :2], dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode
        avg_drop_num = np.mean([info[i]['drop_num'] for i in range(len(info))])
        avg_sw = np.mean([info[i]['sw'] for i in range(len(info))])

        # print("结束测试")
        
        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}  "
              f"| drop_num={avg_drop_num:4.2f} sw={avg_sw:8.2f}")
    
    def test_with_inner_policy(self, policy_id):
        rewards_steps_ary = [step_with_inner_policy(self.env_eval, policy_id) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary)
        info = rewards_steps_ary[:, 2]
        rewards_steps_ary = np.array(rewards_steps_ary[:, :2], dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode
        avg_drop_num = np.mean([info[i]['drop_num'] for i in range(len(info))])
        avg_sw = np.mean([info[i]['sw'] for i in range(len(info))])
        
        used_time = time.time() - self.start_time
        print(f"| {self.total_step}  {used_time:8.0f}  "
              f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
              f"| None  None  "
              f"| drop_num={avg_drop_num:4.2f} sw={avg_sw:8.2f}")


def get_rewards_and_steps(env, actor, if_render: bool = False):  # cumulative_rewards and episode_steps
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    drop_num = 0
    sw = 0.0

    state = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    
    for episode_steps in range(total_steps):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, info = env.step(action)
        cumulative_returns += reward

        drop_num += info["drop_num"]
        sw += info["sw"]

        if if_render:
            env.render()
        if done:
            break

    others = {"drop_num":drop_num, "sw":sw}
    return cumulative_returns, episode_steps + 1, others

def step_with_inner_policy(env, policy_id: int):
    env.reset()
    drop_num = 0
    sw = 0.0
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(total_steps):
        state, reward, done, info = env.step_with_inner_policy(policy_id)
        cumulative_returns += reward

        drop_num += info["drop_num"]
        sw += info["sw"]

        if done:
            break

    others = {"drop_num":drop_num, "sw":sw}
    return cumulative_returns, episode_steps + 1, others

def test(config):
    config["penalty"] = 0.
    num = 5
    
    evaluators = [Evaluator(eval_env=EnvWrapper(config), eval_times=1000000, print_head=False) for _ in range(num-1)]
    evaluators.append(Evaluator(eval_env=EnvWrapper(config), eval_times=100000))   # 独立一个出来打印
    
    pool = mp.Pool(processes=10)
    for i in range(num):
        evaluators[i].total_step = i
        pool.apply_async(evaluators[i].test_with_inner_policy, args=(i,))

    pool.close()
    pool.join()