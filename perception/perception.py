#!/user/bin/env python
"""
感知机实现
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Perception:
    """
    感知机基本实现
    xlb为特征数据
    ylb为类别标签，1、-1
    miu学习率
    epoch最大迭代次数，感知机要求数据必须线性可分，否则，无法实现分类；此处设置了最大迭代次数，避免死循环
    """
    def __init__(self, xlb, ylb, miu, epoch):
        self.miu = miu
        self.b = 1
        self.epoch = epoch
        self.xlb = xlb
        self.ylb = ylb
        n = np.shape(xlb)[1]
        self.w = np.ones((n,), dtype=np.int8)

    def display_param(self):
        print("w:", self.w)
        print("b:", self.b)

    def perception(self):
        """
        基本原理实现，采用随机梯度下降法
        :return:
        """
        m = np.shape(self.xlb)[0]
        flag = 0
        cnt = 0
        while not flag and cnt < self.epoch:
            for i in range(0, m):
                cnt = cnt + 1
                print("w:", self.w, "b:", self.b, "x:", self.xlb[i], "y:", self.ylb[i])
                print(np.transpose(self.xlb[i]))
                print(np.matmul(self.w, np.transpose(self.xlb[i]))+self.b)
                re = self.ylb[i] * (np.matmul(self.w, np.transpose(self.xlb[i])) + self.b)
                print(re)
                if re > 0:
                    if i == m-1:
                        flag = 1
                        break
                    continue
                else:
                    self.w = self.w + self.miu * self.ylb[i] * self.xlb[i]  #随机梯度下降
                    self.b = self.b + self.miu * self.ylb[i]
                    break
        return self.w, self.b

    def classify(self, x):
        """
        :param x: 待分类的数据
        :return: 类别标签1或-1
        """
        x_arr = np.array(x)
        re = np.matmul(self.w, np.transpose(x_arr)) + self.b
        if re > 0:
            return 1
        else:
            return -1

    def display_margin(self):
        """
        打印得到的边界
        :return:,无返回值
        """
        m = np.linspace(0, max(self.xlb[:, 0]), 50)
        n = -(self.w[0] * m + self.b)/self.w[1]
        plt.plot(m, n, 'r')

