#!/user/bin/env python
import numpy as np


class GBDT:

    def calc_var(self, y_data):  #计算方差
        data_var = np.var(y_data)*np.size(y_data)
        return data_var

    def calc_mean(self, y_data): #计算均值
        data_mean = np.mean(y_data)
        return data_mean

    def basic_tree(self, x_data, y_data, regress_dict, x_org_data, y_org_data):  #选择最佳特征分类点（针对某一特征）
        x_data_len = np.size(x_data[0])
        split_point = -1
        split_index = -1
        var_value = np.inf
        y_org_predict = self.predict_fun(regress_dict, x_org_data)  #前几轮迭代组合成的强学习器的预测值
        for i in range(0, x_data_len-1):
            tmp_split_point = (x_data[0, i]+x_data[0, i+1])/2.0  #相邻特征的均值作为待选点，因此需要排序
            n1 = i+1
            y1 = self.calc_mean(y_data[0, 0:n1])
            y2 = self.calc_mean(y_data[0, n1: x_data_len])

            y_org_predict_tmp = y_org_predict.copy()  #必须是深拷贝
            bt = np.where(x_org_data > tmp_split_point)
            lt = np.where(x_org_data <= tmp_split_point)
            y_org_predict_tmp[bt] = y_org_predict_tmp[bt] + y2 #大于分割点的加上y2
            y_org_predict_tmp[lt] = y_org_predict_tmp[lt] + y1 #小于分割点的加上y1
            tmp_var_value = self.loss(y_org_data, y_org_predict_tmp) #计算损失,注意是与真实数据的平方误差损失

            if tmp_var_value < var_value:  #比较损失，若是比当前损失小，就记录转换
                var_value = tmp_var_value
                split_point = tmp_split_point
                split_index = i
        return split_point, split_index

    def train(self, x_data, y_data, threshold=0.1, max_iter_num=3):  #训练
        #根据x_data大小排序
        y_data = np.mat(y_data.A1[np.argsort(x_data).A1])
        x_data.sort()

        iter_num = max_iter_num
        var_value = np.inf
        x_data_iter = np.copy(x_data)
        y_data_iter = np.copy(y_data)
        regress_dict = {}
        k = 0
        while var_value > threshold and iter_num > 0:
            split_point, split_index = self.basic_tree(x_data_iter, y_data_iter, regress_dict, x_data, y_data) #基本学习器
            num = np.size(y_data_iter)
            y1 = np.mat(y_data_iter)[0, 0:split_index+1]
            y2 = np.mat(y_data_iter)[0, split_index+1:num]
            # 此处就相当于一棵树，该树只有根节点
            regress_dict.update({k:{"split_point": split_point, "lmean": np.mean(y1), "rmean": np.mean(y2)}})
            x_data_iter = x_data_iter
            y1 = y1 - np.mean(y1) #原数据减去均值
            y2 = y2 - np.mean(y2)
            y_data_iter = np.hstack((y1, y2)) #组合成新的数据
            iter_num = iter_num - 1
            k = k + 1
            y_predict = self.predict_fun(regress_dict, x_data) #根据已有的强学习器，输出预测值
            var_value = self.loss(y_predict, y_data) #平方误差损失
            print("迭代次数:"+str(k)+" 训练数据损失:"+str(var_value))
        return regress_dict

    def predict_fun(self, regress_dict, x): #预测输入x的输出y
        y_predict = np.mat(np.zeros(np.size(x)))
        for (key, value) in regress_dict.items(): #遍历已有的分类器，计算预测值
            tree_dict = value
            bt = np.where(x > tree_dict['split_point'])
            lt = np.where(x <= tree_dict['split_point'])
            y_predict[bt] = y_predict[bt] + tree_dict['rmean']
            y_predict[lt] = y_predict[lt] + tree_dict['lmean']
        return y_predict

    def loss(self, y_true, y_predict): #计算平方误差损失
        if np.size(y_true) != np.size(y_predict):
            return np.inf
        y_error = y_true-y_predict
        loss = np.sum(np.square(y_error))
        return loss






