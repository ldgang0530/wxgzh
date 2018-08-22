#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/15 22:12
@Author: ldgang
'''
import cv2


if __name__ == "__main__":
    img = cv2.imread("img/haizei0.jpg")
    cv2.imshow("haizei0", img)
    img2 = cv2.add(img, 300)
    cv2.imshow('img2', img2)
    img_flip = cv2.flip(img, -1)
    cv2.imshow('img_flip', img_flip)

    img_add = cv2.addWeighted(img, 0.7, img_flip, 0.3, 0)
    cv2.imshow('img_add', img_add)

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img_flip.shape
    roi = img[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img_flip, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    # 取ROI中与mask中不为零的值对应的像素的值，其让值为0 。
    # 注意这里必须有mask=mask或者mask=mask_inv，其中mask=不能忽略
    img_bg = cv2.bitwise_and(roi, roi, mask=mask)
    # 取roi中与mask_inv中不为零的值对应的像素的值，其他值为0
    # Take only region of logo from logo image.
    img_flip_fg = cv2.bitwise_and(img_flip, img_flip, mask=mask_inv)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img_bg, img_flip_fg)
    img[0:rows, 0:cols] = dst
    cv2.imshow('res', img)

    cv2.waitKey()