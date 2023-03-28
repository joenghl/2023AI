"""
第5次作业, 用遗传算法解决TSP问题
本次作业可使用`numpy`库和`matplotlib`库以及python标准库
请不要修改类名和方法名
"""
import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgTSP:
    def __init__(self, tsp_filename):
        self.cities = None  # 读取文件数据, 存储到该成员中
        self.population = None  # 初始化种群, 会随着算法迭代而改变

    def iterate(self, num_iterations):
        # 基于self.population进行迭代,返回当前较优路径
        pass


if __name__ == "__main__":
    tsp = GeneticAlgTSP("dj38.tsp")  # 读取Djibouti城市坐标数据
    T = 10
    tour = tsp.iterate(T)  # 对算法迭代10次
    print(tour)  # 打印路径(以列表的形式)
