1. PPO.py 改了 Config 的 horizon_len eval_times eval_per_step batch_size repeat_times 以进行调试
2. 改了 action
3. 调整了 get_rewards_and_steps 中的 for episode_steps in range(12345), 在测试过程中需要改大 step 上限

修改:
1. reward 好像有点问题?
2. 或许 follower 的 policy 有问题

========

1. 先把各个地方的随机性降低看看
- task.py
- user.py

2. 全部设置为已知
- server.py: self.is_known = True

效果不行

========

改神经网络，增加 batch norm，改网络规模
暂时调小训练总步数
加了 eval()