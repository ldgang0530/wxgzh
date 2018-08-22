#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/17 23:01
@Author: ldgang
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    img = cv2.imread('img/meixi2.jpg',0)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap='gray')
    plt.title('original'),plt.xticks([]),plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap='gray')
    plt.title('edge'),plt.xticks([]),plt.yticks([])
    plt.show()