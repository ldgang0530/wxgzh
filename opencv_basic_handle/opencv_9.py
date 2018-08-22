#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/15 23:03
@Author: ldgang
'''
import cv2
import numpy as np

def track():
    cap = cv2.VideoCapture(0)
    while (1):
        # 获取每一帧
        ret, frame = cap.read()
        # 转换到HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 设定蓝色的阀值
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        # 根据阀值构建掩模
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # 对原图和掩模进行位运算
        res = cv2.bitwise_and(frame, frame, mask=mask)
        # 显示图像
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:  #ESC键
            break
    # 关闭窗口
    cv2.destroyAllWindows()


if __name__ == "__main__":
    flags = [i for i in dir(cv2) if i.startswith( 'COLOR_')]
    print(flags)

    track()

    input()