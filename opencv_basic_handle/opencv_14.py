#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/17 22:55
@Author: ldgang
'''

import cv2
import numpy
from matplotlib import pyplot as plt

if __name__ == "__main__":
    img = cv2.imread('img/meixi3.jpg',0)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
    plt.title('original'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')
    plt.title('laplacian'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')
    plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')
    plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])

    plt.show()