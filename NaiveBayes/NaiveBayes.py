#!/user/bin/env python
import numpy as np
from sklearn.model_selection import train_test_split


class NavieBayes:

    def pre_data_handle(self, data_list, ratio, random_state=5):
        """
        数据预处理
        :param data_list: 样本集
        :param ratio: 训练集与样本集比例
        :param random_state:  随机种子
        :return:
        """
        data = np.array(data_list)
        x = data[:, 1:]
        y = data[:, 0]
        #划分训练集与样本集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio, random_state=random_state)
        #将字符串转换为float或int型
        return np.float32(x_train), np.float32(x_test), np.int8(y_train), np.int8(y_test)

    def train(self, x_train, y_train, class_num):
        """
        贝叶斯训练
        :param x_train: 训练样本特征
        :param y_train: 训练样本类别
        :param class_num:  分类类别数目
        :return: 概率
        """
        m, n = np.shape(x_train)
        y_train_mat = np.int8(y_train)
        #划分数据集
        class_num_dict = {} #记录每个类别的数目
        class_x_train_data_dict = {} #记录每个类别对应的训练样本
        for i in range(0, class_num):
            num = y_train_mat[y_train_mat == i + 1].size
            class_num_dict.update({i+1: num})
            flag = (y_train_mat == i + 1)
            class_x_train_data_dict.update({i+1: np.mat(x_train)[(np.where(flag == True))]})
        #计算各类别的概率（如P(C1)）
        prob_num_dict = {}
        for j in class_num_dict:
            prob_num_dict.update({j: 1.0*class_num_dict.get(j)/m})  #不同类别的概率，P（C）
        #计算固定类别中每个特征的概率分布
        prob_dict = {}
        for j in class_x_train_data_dict:
            x_data = np.float32(class_x_train_data_dict.get(j))
            x_data_var = np.var(x_data, axis=0) #按列求方差
            x_data_mean = np.mean(x_data, axis=0) #按列求均值
            # 同一类别中不同特征的均值和方差，可用于记录概率分布函数（P（Xi/Cj），注意每一个特征都有均值和方差）
            prob_dict.update({j: {'var': x_data_var, 'mean': x_data_mean}})
        return prob_num_dict, prob_dict

    def classify(self, prob_num_dict, prob_dict, data, class_num):
        """
        根据训练得到的概率密度函数，对待测数据分类
        :param prob_num_dict: P（Xi/C）
        :param prob_dict:  P（C）
        :param data: 待测数据
        :param class_num: 类别个数
        :return: 预测的数据类别
        """
        m0, n0 = np.shape(data)
        #class_result = {}
        class_result = []
        for j in range(0, m0):
            class_prob = -np.inf
            class_predict = -1
            for i in range(0, class_num):  #遍历每一个类别
                prob_class = prob_num_dict.get(i+1)
                data_var = prob_dict.get(i+1).get("var")
                data_mean = prob_dict.get(i+1).get("mean")
                m1, n1 = np.shape(data_mean)
                if n0 != n1:
                    print(" Test data feature num isn't right , test data index:"+str(j)+"class index:"+str(i+1))
                else:
                    #统计待分类数据的概率
                    prob_data_class = np.multiply(1.0 / np.sqrt(2 * np.pi * data_var),
                                                  np.exp(-1.0 * np.square(data[j]-data_mean) / (2 * data_var)))
                    #计算（(P(x1/c)*P(x2/c)*...*P(xn/c))P(c）)，因值较小，且乘法不好运算，所以此处取对数求和
                    prob = np.sum(np.log(prob_data_class))+np.log(prob_class)
                    if class_prob < prob:
                        class_predict = i + 1
                        class_prob = prob
            #class_result.update({j: class_predict})
            class_result.append(class_predict) #记录预测的类别
        return class_result

    def calc_right_rate(self, y_true, y_predict):
        """
        统计正确率
        :param y_true: 真实类别
        :param y_predict:  预测类别
        :return: 正确率
        """
        num = len(y_true) #总样本数
        if num != len(y_predict):
            print("y_true and y_predict data num isn't match")
            return 0.0
        y_true_mat = np.mat(y_true)
        y_predict_mat = np.mat(y_predict)
        comp_mat = y_true_mat - y_predict_mat #对应元素相减
        right_num = np.sum(comp_mat == 0) #统计0的元素的个数，即真实值与预测值相同的个数
        return 1.0*right_num/num

