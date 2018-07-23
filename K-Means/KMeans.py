#!/user/bin/env python

import numpy as np
import random


class KMeans:
    """
    Kmeans无监督聚类，不需要预先训练
    缺陷：需要预先确定类别数目K；易受初始的K个中心的影响，初始点不同，得到的结果可能不同;
        对于异常点很敏感； 运算量大
    """
    def __init__(self, k=1, iter_num=5):
        """"
        k, 聚类数目； iter_num迭代次数；
        此外，还可以加上阈值，当新的数据中心与旧数据中心相差不大时退出，此处没有设定
        """
        self.k = k
        self.iter_num = iter_num

    def classify(self, x_data):
        """
        聚类
        :param x_data:
        :return: 返回聚类中心和对应的数据集
        """
        iter_num = self.iter_num
        x_data_mat = np.mat(x_data)
        m1, n1 = np.shape(x_data_mat)
        #按列求最大最小值
        x_min = np.min(x_data_mat, axis=0)
        x_max = np.max(x_data_mat, axis=0)
        #最大减去最小，因要做除法，故须删除掉最大与最小值相同的情况，这也减少了计算量
        den = x_max - x_min
        idx = np.where(den != 0)
        idy = np.where(den == 0)
        x_min_div = x_min[idx]
        x_max_div = x_max[idx]
        x_equal = x_data_mat[:, idy[1]]
        x_data_mat_t = x_data_mat.transpose().getA()
        x_data_div = np.mat(x_data_mat_t[idx[1]]).transpose()
        #归一化，可以有效避免不同特征值之间量的影响，
        # 比如人的岁数为28岁，而体重有180公斤，不在一个量级上，欧式距离计算影响较大,故须归一化
        normal_data = np.divide(1.0*(x_data_div-x_min_div), x_max_div-x_min_div)
        normal_data_mat = np.mat(normal_data)
        m, n = np.shape(normal_data_mat)
        if m <= self.k:  #若样本数目小于要分类的个数，则直接返回数据即可
            return np.mat(x_data), x_data
        seq = random.sample([idx for idx in range(0, m)],  self.k)
        class_center = np.mat(normal_data_mat.getA()[seq])  #随机选择k个样本作为初始类别中心
        class_data = []
        while iter_num > 0: #若迭代次数大于0
            class_data = []
            for i in range(0, self.k):
                class_data.append([])
            for i in range(0, m):
                # 计算数据中心与各个数据样本的欧式距离（未求平方根，直接求平方和）
                x_bias = np.sum(np.square(class_center - normal_data_mat[i, :]), axis=1)
                c_idx = np.argmin(x_bias) #选择与样本距离最小的中心作为该样本的类别中心
                class_data[c_idx].append(normal_data_mat[i, :].getA()[0])
            class_data_list = []
            for i in range(0, self.k):
                class_data_list.append(np.mean(np.mat(class_data[i]), axis=0).getA()[0])  #求每个类别新的数据中心
            class_center = np.mat(class_data_list)
            iter_num = iter_num - 1

        #下面的操作主要是为了还原为归一化前的数据
        result_center = np.add(np.multiply(class_center, x_max_div - x_min_div), x_min_div).getA()  #聚类中心恢复
        k = 0
        m, n = np.shape(result_center)
        v1_data = np.mat(np.ones(m)).transpose()
        for i in idy[1]:
            if result_center[:, i:].size == 0:
                v1_mat = x_equal[0, k]*v1_data
            else:
                v1_mat = np.hstack((x_equal[0, k]*v1_data, result_center[:, i:]))
            if result_center[:, :i].size == 0:
                result_center = v1_mat
            else:
                result_center = np.hstack((result_center[:, :i], v1_mat))
            k = k+1
        result_class_data = []  #聚类对应的数据恢复
        for data in class_data:
            data_mat = np.mat(data)
            data_mat = np.add(np.multiply(data_mat, x_max_div - x_min_div), x_min_div).getA()
            m1, n1 = np.shape(data_mat)
            v2_data = np.mat(np.ones(m1)).transpose()
            k = 0
            for i in idy[1]:
                if data_mat[:, i:].size == 0:
                    v2_mat = x_equal[0, k] * v2_data
                else:
                    v2_mat = np.hstack((x_equal[0, k] * v2_data, data_mat[:, i:]))
                if data_mat[:, :i].size == 0:
                    data_mat = v2_mat
                else:
                    data_mat = np.hstack((data_mat[:, :i], v2_mat))
                k = k + 1
            result_class_data.append(data_mat)
        return result_center, result_class_data  #返回数据中心与对应的数据集





