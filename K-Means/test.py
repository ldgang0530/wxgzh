#!/user/bin/env python

import KMeans
import numpy as np


def load_dataset():  #加载数据集
    x = [(7, 2, 1, 2), (2, 3, 1, 2), (3, 4, 1, 2), (4, 5, 1, 2), (5, 6, 1, 2)]
    return x


if __name__ == "__main__":
    x_data = load_dataset()
    class_num = 3
    kmeans = KMeans.KMeans(k=class_num, iter_num=5)  #实例化
    class_center, class_data = kmeans.classify(x_data)  #聚类
    if class_num > len(x_data):
        class_num = len(x_data)
    for i in range(0, class_num):
        print("class NO", i, "\n    class_center:", class_center[i, :], "\n    class_data:", class_data[i].tolist())  #打印聚类结果