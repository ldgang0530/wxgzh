#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/14 23:50
@Author: ldgang
'''

import numpy as np
import cv2


def videoFromCapture():
    cap = cv2.VideoCapture(0)  # 摄像头，0代表默认
    while (True):
        # capture frame-by-frame
        ret, frame = cap.read()  # 按帧读取

        # our operation on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 进行色彩空间转换q

        # display the resulting frame
        cv2.imshow('frame', gray)  # 显示
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break
    # when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()

def videoFromFile():
    cap = cv2.VideoCapture('img/videoLearning.mp4')  # 文件名及格式
    while (True):
        # capture frame-by-frame
        ret, frame = cap.read()
        if(ret == True):
            # our operation on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # display the resulting frame
            cv2.imshow('frame', gray)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            break
    # when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #videoFromCapture()
    videoFromFile()
