import matplotlib.pyplot as plt
import os
import openpyxl
from statistics import mean
import numpy as np

class ResultCurve:
    def __init__(self):
        self.random_action = 615.98   # 0
        self.random_select = 1074    # 1
        self.greedy = 2710    # 2
        self.all = 2371  # 4
        self.optimal = 5085  # 3
        self.ppo_results = []
        self.time_points = []   # TODO: 这里不太对，应该和 step 统一
        self.steps = 0

        self.draw_var = False

    def set_results(self, results: list):
        self.ppo_results = results
        self.steps = len(self.ppo_results)
        self.time_points = range(0, self.steps)

    def set_points(self, points):
        self.time_points = [int(x) for x in points]

    def update_results(self, new_results: list):
        self.ppo_results += new_results
        self.steps = len(self.ppo_results)
        self.time_points = range(0, self.steps)

    def set_variance(self, ceil_list: list, floor_list: list):
        self.ceil_list = ceil_list
        self.floor_list = floor_list
        self.draw_var = True

    def save_plot(self, filename):
        # 设置图形尺寸和标题
        plt.figure(figsize=(12, 8))
        # plt.title("Results Comparison")

        # 绘制基准线和 PPO 奖励曲线
        # plt.plot(self.time_points, [self.random_action for _ in range(self.steps)], label="Random Action")
        plt.plot(self.time_points, [self.random_select for _ in range(self.steps)], color='pink', linewidth=3, label="Random")
        plt.plot(self.time_points, [self.greedy for _ in range(self.steps)], color='green', linewidth=3, label="Greedy")
        plt.plot(self.time_points, [self.all for _ in range(self.steps)], color='orange', linewidth=3, label="All")
        plt.plot(self.time_points, [self.optimal for _ in range(self.steps)], color='red', linewidth=3, label="Optimal")
        plt.plot(self.time_points, self.ppo_results, color='blue', linewidth=2, label="PPO")

        if self.draw_var:
            # 填充两根线之间的区域
            plt.fill_between(self.time_points, self.ceil_list, self.floor_list, color='lightblue', alpha=0.5, label="Std")
            plt.ylim(0, 6000)
            plt.xlim(0, max(self.time_points))

        # 添加图例和坐标轴标签
        plt.legend(loc='lower right', fontsize=16)
        plt.xlabel("Training Step", fontsize=24, weight='bold')
        plt.ylabel("Social Welfare", fontsize=24, weight='bold')

        plt.tick_params(axis='x', labelsize=16)
        plt.tick_params(axis='y', labelsize=16)

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
            points_data.append(points_value / 10000)    # threads_num * horizon_len
        sw_data = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            sw_value = row[8]  # 读取 SW 数据（SW 数据在第 9 列）, 第 9 列的索引为 8
            sw_data.append(sw_value)

        # 计算滑动平均值
        window_size = 30  # 滑动窗口大小
        smoothed_sw_data = []
        smoothed_ceil_data = []
        smoothed_floor_data = []
        for i in range(len(sw_data) - window_size + 1):
            window = sw_data[i : i + window_size]
            smoothed_value = mean(window)
            smoothed_sw_data.append(smoothed_value)

            std = np.std(window)
            ceil_value = smoothed_value + std
            floor_value = smoothed_value - std

            # sorted_window = sorted(window)
            # ceil_value = sorted_window[-2]
            # floor_value = sorted_window[1]

            smoothed_ceil_data.append(ceil_value)
            smoothed_floor_data.append(floor_value)


        curve = ResultCurve()
        curve.set_results(smoothed_sw_data)  # 使用滑动平均值数据
        curve.set_points(points_data[:len(smoothed_sw_data)])
        curve.set_variance(smoothed_ceil_data, smoothed_floor_data)
        curve.save_plot("../../results/reward_curve.png")
