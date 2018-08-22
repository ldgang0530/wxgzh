#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/16 21:58
@Author: ldgang
'''

import cv2
from matplotlib import pyplot as plt
import numpy as np

def draw_resize():
    img = cv2.imread('img/haizei3.jpg')
    #下面的None本应该是输出图像的尺寸，但是因为后面我们设置了缩放因子，所以，这里为None
    res1 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
    #or
    #这里直接设置输出图像的尺寸，所以不用设置缩放因子
    height , width =img.shape[:2]
    res2 = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)

    while(1):
        cv2.imshow('res',res1)
        cv2.imshow('img',img)
        cv2.imshow('res2', res2)
        if cv2.waitKey(1)&0xFF == 27:
            break
    cv2.destroyAllWindows()


def draw_rotate():
    img = cv2.imread('img/haizei3.jpg', 0)
    rows, cols = img.shape
    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子以及窗口大小来防止旋转后超出边界的问题。
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.6)
    # 第三个参数是输出图像的尺寸中心
    dst = cv2.warpAffine(img, M, (2 * cols, 2 * rows))
    while (1):
        cv2.imshow('img', dst)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


def draw_affine():
    img = cv2.imread('img/haizei3.jpg')
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    # 行，列，通道数
    M = cv2.getAffineTransform(pts1, pts2)

    dst = cv2.warpAffine(img, M, (cols, rows))

    plt.subplot(1,2,1);plt.imshow(img); plt.title("Input")
    plt.subplot(1,2,2); plt.imshow(dst); plt.title('output')
    plt.show()


def draw_perspective():
    img = cv2.imread('img/haizei3.jpg')
    rows, cols, ch = img.shape

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (300, 300))
    plt.subplot
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Input')
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.title('Output')
    plt.show()


if __name__ == "__main__":
    #draw_resize()
    #draw_rotate()
    #draw_affine()
    draw_perspective()
