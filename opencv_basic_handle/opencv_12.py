#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/17 21:04
@Author: ldgang
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt


def cov_mean():
    img = cv2.imread('img/haizei3.jpg')
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    plt.subplot(121), plt.imshow(img), plt.title('original')
    plt.subplot(122), plt.imshow(dst), plt.title('averaging')
    plt.show()


def blur_mean():
    img = cv2.imread('img/haizei3.jpg')
    blur = cv2.blur(img, (5, 5))
    while (1):
        cv2.imshow('image', img)
        cv2.imshow('blur', blur)
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


def gaussian_blur():
    img = cv2.imread('img/haizei3.jpg')
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    while(1):
        cv2.imshow('image', img)
        cv2.imshow('guassian_blur', blur)
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


def median_blur():
    img = cv2.imread('img/haizei3.jpg')
    blur = cv2.medianBlur(img, 5)
    while(1):
        cv2.imshow('image', img)
        cv2.imshow('median_blur', blur)
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


def bilateralFilter_blur():
    img = cv2.imread('img/haizei3.jpg')
    blur = cv2.bilateralFilter(img,9,75,75)
    while (1):
        cv2.imshow('image', img)
        cv2.imshow('bilateralFilter_blur', blur)
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #cov_mean()
    #blur_mean()
    #gaussian_blur()
    #median_blur()
    bilateralFilter_blur()