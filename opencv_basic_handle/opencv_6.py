#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/15 21:28
@Author: ldgang
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv2.imread('img/haizei1.jpg')
    #cv2.imshow('haizei1', img)
    print(img.shape)  #行、列、通道数
    print(img.size)  #像素数目
    print(img.dtype)  #返回图像的数据类型

    ball = img[50:100, 260:400]
    img[200:250,50:190] = ball
    #cv2.imshow('haizei1', img)
    #拆分/合并
    r, g, b = cv2.split(img)  # 拆分
    #cv2.imshow('haizei_r', r)
    #img = cv2.merge(r,g, b)

    #边缘处理
    img = cv2.imread('img/meixi1.jpg')
    blue = [255, 0, 0]
    replicate = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_REFLECT101)
    wrap = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=blue)

    plt.subplot(231), plt.imshow(img, 'gray'), plt.title('original')
    plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('replicate')
    plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('reflect')
    plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('reflect101')
    plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('wrap')
    plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('constant')

    #img = cv2.merge(r, g, b)  # 合并
    cv2.waitKey()
    cv2.destroyAllWindows()