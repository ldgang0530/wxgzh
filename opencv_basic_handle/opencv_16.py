#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/17 23:07
@Author: ldgang
'''


import numpy as np
import cv2


if __name__ == "__main__":
    img = cv2.imread('img/meixi1.jpg')
    lower_reso = cv2.pyrDown(img)
    higher_reso2 = cv2.pyrUp(img)

    while(1):
        cv2.imshow('img',img)
        cv2.imshow('lower_reso',lower_reso)
        cv2.imshow('higher_reso2',higher_reso2)
        if cv2.waitKey() == ord('q'):
            break
    cv2.destroyAllWindows()