#! /user/bin/env python
import cv2
import numpy as np

import GeneralFunc


def compute_hog(img_list, wsize = (64, 128)):
    """
    计算图像的HOG特征
    :param img_list:
    :param wsize: #图像的大小
    :return:
    """
    gradient_list = []
    hog = cv2.HOGDescriptor()  #初始化HOG描述子
    for i in range(len(img_list)):
        if img_list[i].shape[1] >= wsize[0] and img_list[i].shape[0] >= wsize[1]:   #图像要大于需要的图像的大小
            roi = img_list[i][(img_list[i].shape[0] - wsize[1]) // 2: (img_list[i].shape[0] - wsize[1]) // 2 + wsize[1],
                  (img_list[i].shape[1] - wsize[0])//2 : (img_list[i].shape[1] - wsize[0])//2 + wsize[0]]
            hog_data = hog.compute(roi)  #计算截取的图像的 HOG特征
            gradient_list.append(hog_data)
    return gradient_list  #返回图像的HOG特征


def svm_init():  #初始化svm
    svm = cv2.ml.SVM_create()  #
    svm.setCoef0(0)
    svm.setDegree(3)   #多项式核时，变量的阶数
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)  #
    svm.setKernel(cv2.ml.SVM_LINEAR)  #SVM的核函数
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)  #SVM的类型
    return svm


def get_svm_detector(svm):   #获取SVM特征向量
    sv = svm.getSupportVectors() #获取支持向量
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


if __name__ == "__main__":
    base_path = 'D:/Samples/DataSets/INRIAPerson/'
    # 加载正例图像，大小均为（64*128）
    pos_list = GeneralFunc.load_imge(base_path+"Train/pos.lst", imageNum=600, showImage=False)   #正样本
    # 负例图像，大小不定，需要裁剪
    neg_list_all = GeneralFunc.load_imge(base_path+"Train/neg.lst", imageNum=600, showImage=False)
    neg_list = GeneralFunc.sample_neg(neg_list_all, imageNum=600, size=[64, 128])  #行人，一般是站立的，负例图像

    ylabels = []
    gradient_list = []
    pos_gradient_list = compute_hog(pos_list, wsize=(64, 128))   #计算HOG特征
    [ylabels.append(1) for _ in range(len(pos_list))] #正例标签
    neg_gradient_list = compute_hog(neg_list, wsize=(64, 128))  #计算HOG特征
    [ylabels.append(-1) for _ in range(len(neg_list))] #负例标签
    gradient_list.extend(pos_gradient_list)
    gradient_list.extend(neg_gradient_list)

    svm = svm_init()  #初始化svm
    svm.train(np.array(gradient_list), cv2.ml.ROW_SAMPLE, np.array(ylabels))  #训练SVM模型

    hog = cv2.HOGDescriptor()  #hog描述子
    hard_neg_list = []
    hog.setSVMDetector(get_svm_detector(svm))  #setSVMDetector用于加载svm模型，对hog特征分类的svm的系数赋值
    #加入错误分类的标签，重新对svm进行训练
    for i in range(len(neg_list_all)):
        rects, wei = hog.detectMultiScale(neg_list_all[i], winStride=(4,4), padding=(8,8), scale=1.05)
        for (x,y, w,h ) in rects:
            hardExample = neg_list_all[i][y:y+h, x:x+w]
            hard_neg_list.append(cv2.resize(hardExample,(64,128)))
    hard_gradient_list = compute_hog(hard_neg_list)
    [ylabels.append(-1) for _ in range(len(hard_neg_list))]
    gradient_list.extend(hard_gradient_list)
    #训练svm
    svm.train(np.array(gradient_list), cv2.ml.ROW_SAMPLE, np.array(ylabels))

    #保存hog
    hog.setSVMDetector(get_svm_detector(svm))
    hog.save('myHogDector.bin')
    print("Train success")
