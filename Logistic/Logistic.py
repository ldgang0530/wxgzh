#!/user/bin/env python
import numpy as np
import random


class Logistic:
    def __init__(self, x, y, train_test_rate=0.3, alpha=0.5, epoch=150):
        '''
        :param x:样本数据
        :param y:  样本标签
        :param train_test_rate: 训练集与测试集比例
        :param alpha: 学习步长
        :param epoch: 迭代次数
        '''
        x_arr = np.array(x)
        y_arr = np.array(y)
        z = y_arr.reshape(-1, 1)
        xdata = np.hstack((x_arr, z))
        m, n = np.shape(xdata)
        sample_num = int(m*train_test_rate)
        xdata_list = xdata.tolist()
        data_test = random.sample(xdata_list, sample_num)
        data_train = []
        for i in range(0, m):
            if xdata_list[i] not in data_test:
                data_train.append(xdata_list[i])

        self.data_test = np.array(data_test)
        self.data_train = np.array(data_train)
        self.w = np.ones(n - 1)
        self.b = 0
        self.alpha = alpha
        self.epoch = epoch

    def sigmoid(self, x):
        """
        sigmoid函数
        :param x:
        :return:
        """
        z = np.matmul(self.w, x.reshape(-1, 1))+self.b
        return 1.0/(1+np.exp(-z))

    def train(self):
        """
        模型训练
        :return:
        """
        m, n = np.shape(self.data_train)
        for i in range(0, self.epoch):
            for j in range(m):
                y = self.sigmoid(self.data_train[j, :-1])
                y_t = self.data_train[j, n-1]
                error = y - y_t
                self.w = self.w + self.alpha*error*self.data_train[j, :-1]
                self.b = self.b + self.alpha*error*self.b
        return self.w, self.b

    def classify(self, x):
        """
        预测分类
        :param x:
        :return:
        """
        tmpx = np.array(x)
        y = self.sigmoid(tmpx)
        if y > 0.5:
            return 1
        else:
            return 0

    def right_rate(self):
        """
        测试集正确率
        :return:
        """
        m, n = np.shape(self.data_test)
        if m == 0:
            return 0
        right_num = 0
        for i in range(m):
            y = self.classify(self.data_test[i, :-1])
            y_t = self.data_test[i, n-1]
            if y == y_t:
                right_num += 1
        return 1.0*right_num/m