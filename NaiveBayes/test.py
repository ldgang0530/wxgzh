#!/user/bin/env python
import os
import NaiveBayes


def load_dataset(file_path):
    fd = open(file_path, 'r')
    file_lines = fd.readlines()
    data_list0 = []
    for line in file_lines:
        data_list0.append(line.split(","))
    return data_list0


def disp_result(y_true, y_predict):
    num = len(y_true)
    if num != len(y_predict):
        print("y_true and y_predict data num isn't match")
    else:
        for i in range(0,num):
            print("测试集样本类别： 真实类别:"+str(y_true[i])+" 预测类别:"+str(y_predict[i]))


if __name__ == "__main__":
    path = "/".join(os.getcwd().split("\\"))
    data_list = load_dataset(path+"/Data/wine_data.txt")

    navieBayes = NaiveBayes.NavieBayes()
    x_train, x_test, y_train, y_test = navieBayes.pre_data_handle(data_list, ratio=0.3, random_state=5)
    prob_num_dict, prob_dict = navieBayes.train(x_train, y_train, class_num=3)
    y_predict = navieBayes.classify(prob_num_dict, prob_dict, x_test, class_num=3)

    disp_result(y_test, y_predict)
    right_rate = navieBayes.calc_right_rate(y_test, y_predict)
    print("Right rate:"+str(right_rate))
