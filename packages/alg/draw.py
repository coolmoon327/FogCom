import matplotlib.pyplot as plt

class ResultCurve:
    def __init__(self):
        self.random_action = -500.87
        self.random_select = 2400.52
        self.greedy = 717.08
        self.all = 579.63
        self.optimal = 3799.86
        self.ppo_results = []
        self.time_points = []
        self.steps = 0

    def set_results(self, results: list):
        self.ppo_results = results
        self.steps = len(self.ppo_results)
        self.time_points = range(0, self.steps)

    def update_results(self, new_results: list):
        self.ppo_results += new_results
        self.steps = len(self.ppo_results)
        self.time_points = range(0, self.steps)

    def save_plot(self, filename):
        # 设置图形尺寸和标题
        plt.figure(figsize=(8, 6))
        plt.title("Rewards Comparison")

        # 绘制基准线和 PPO 奖励曲线
        plt.plot(self.time_points, [self.random_action for _ in range(self.steps)], label="Random Action")
        plt.plot(self.time_points, [self.random_select for _ in range(self.steps)], label="Random Select")
        plt.plot(self.time_points, [self.greedy for _ in range(self.steps)], label="Greedy")
        plt.plot(self.time_points, [self.all for _ in range(self.steps)], label="All")
        plt.plot(self.time_points, [self.optimal for _ in range(self.steps)], label="Optimal")
        plt.plot(self.time_points, self.ppo_results, label="PPO")

        # 添加图例和坐标轴标签
        plt.legend()
        plt.xlabel("Training Step")
        plt.ylabel("Social Welfare")

        # 保存图像到指定位置
        plt.savefig(filename)

if __name__ == "__main__":
    # 创建 ResultCurve 对象
    curve = ResultCurve()

    # 更新 PPO 奖励值
    new_results = [14, 15, 13, 12, 11]
    curve.update_results(new_results)

    # 保存图像
    curve.save_plot("./data/reward_curve.png")
