#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tf_verificationCode
@Product name: PyCharm
@Time: 2018/3/6 16:09
@Author: ldgang
'''
import tensorflow as tf
import numpy as np
from param import *
from PIL import Image

#彩色图像转换为灰度图像
def rgb2gray(image):
    if(len(np.shape(image))>2):
        return tf.image.rgb_to_grayscale(image)
    else:
        return image

#文字转向量
def txt2vec(txt): #文字转向量，用于标签和预测的对比
    txtLen = len(txt)
    vec = np.zeros(CODE_NUM*charSetLen)
    for i in range(txtLen):
        idx = i*charSetLen+globCharSet.index(txt[i])
        vec[idx] = 1
    return vec


def vec2txt(vec):
    txt = []
    vecLen = len(vec)
    if (vecLen % charSetLen != 0) or (vecLen/charSetLen != 4):
        print("Value Error")
        return "0000"
    for i in range(4):
        vecLabel = vec[i*charSetLen :(i+1)*charSetLen]
        txt.append(globCharSet[np.where(vecLabel==1)[0][0]])
    return txt


def get_image_data(fileName): #获取图像数据
    curFile = fileName
    image = Image.open(curFile)
    grayImage = rgb2gray(image) #RGB转灰度图像
    return grayImage


def get_image_label(fileName):
    txt = fileName[0:CODE_NUM]  #名字的前CODE_NUM个字母即为value值
    return txt2vec(txt)


