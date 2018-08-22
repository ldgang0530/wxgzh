#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/17 21:36
@Author: ldgang
'''
import cv2
import numpy as np


def erode():
    img = cv2.imread('img/haizei3.jpg', 0)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    while (1):
        cv2.imshow('image', img)
        cv2.imshow('erosion', erosion)
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


def dilation():
    img = cv2.imread('img/haizei3.jpg', 0)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(img,kernel,iterations=1)
    while (1):
        cv2.imshow('image', img)
        cv2.imshow('dilate', dilate)
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


def morphotogyEx():
    img = cv2.imread('img/haizei3.jpg', 0)
    kernel = np.ones((5, 5), np.uint8)
    MORPH_OPEN = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    MORPH_CLOSE = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    MORPH_GRADIENT = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    MORPH_TOPHAT = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    MORPH_BLACKHAT = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    while (1):
        cv2.imshow('image', img)
        cv2.imshow('MORPH_OPEN', MORPH_OPEN)
        cv2.imshow('MORPH_CLOSE', MORPH_CLOSE)
        cv2.imshow("MORPH_GRADIENT", MORPH_GRADIENT)
        cv2.imshow("MORPH_TOPHAT", MORPH_TOPHAT)
        cv2.imshow("MORPH_BLACKHAT", MORPH_BLACKHAT)
        k = cv2.waitKey(1)
        if k == ord('q'):  # 按q键退出
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #erode()
    #dilation()
    morphotogyEx()
