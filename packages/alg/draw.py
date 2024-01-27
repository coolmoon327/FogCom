import matplotlib.pyplot as plt
import os
import openpyxl

class ResultCurve:
    def __init__(self):
        self.random_action = 1575.24    # 0
        self.random_select = 4305.29    # 1
        self.greedy = 5939.08    # 2
        self.all = 5812.87  # 4
        self.optimal = 6358.53  # 3
        self.ppo_results = []
        self.time_points = []   # TODO: 这里不太对，应该和 step 统一
        self.steps = 0

    def set_results(self, results: list):
        self.ppo_results = results
        self.steps = len(self.ppo_results)
        self.time_points = range(0, self.steps)

    def set_points(self, pionts):
        self.time_points = pionts

    def update_results(self, new_results: list):
        self.ppo_results += new_results
        self.steps = len(self.ppo_results)
        self.time_points = range(0, self.steps)

    def save_plot(self, filename):
        # 设置图形尺寸和标题
        plt.figure(figsize=(8, 6))
        # plt.title("Results Comparison")

        # 绘制基准线和 PPO 奖励曲线
        # plt.plot(self.time_points, [self.random_action for _ in range(self.steps)], label="Random Action")
        plt.plot(self.time_points, [self.random_select for _ in range(self.steps)], label="Random")
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
        plt.close()

if __name__ == "__main__":
    file_path = "../../results/output.xlsx"
    if os.path.exists(file_path):
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active
        points_data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            points_value = row[0]
            points_data.append(points_value)
        sw_data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            sw_value = row[8]  # 读取 SW 数据（SW 数据在第 9 列）, 第 9 列的索引为 8
            sw_data.append(sw_value)
        curve = ResultCurve()
        curve.set_results(sw_data)
        curve.set_points(points_data)
        curve.save_plot("../../results/reward_curve.png")