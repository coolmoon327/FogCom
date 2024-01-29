1. PPO.py 改了 Config 的 horizon_len eval_times eval_per_step batch_size repeat_times 以进行调试
2. 改了 action
3. 调整了 get_rewards_and_steps 中的 for episode_steps in range(12345), 在测试过程中需要改大 step 上限

修改:
1. reward 好像有点问题?
2. 或许 follower 的 policy 有问题

======== 1.27 1

1. 先把各个地方的随机性降低看看
- task.py
- user.py

2. 全部设置为已知
- server.py: self.is_known = True

效果不行

======== 1.27 2

改神经网络，增加 batch norm，改网络规模
暂时调小训练总步数
加了 eval()

效果不错

======== 1.27 3

增大训练量
增大探索的次数

======== 1.28 1

改回 server.py 的 self.is_known


========= 1.28 2

改了一些计价相关参数

========= 1.29 1

之前效果不好，重新把 self.is_known 调高
调整 alpha 和 beta，让 policy=5 的情况比 2 好，policy=4 的情况比 2 差

========= 1.29 2

调大失败惩罚
改了 repeat time
