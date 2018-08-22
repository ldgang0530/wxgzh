#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/14 23:32
@Author: ldgang
'''

import cv2

if __name__ == "__main__":
    img = cv2.imread("img/haizei0.jpg", 0)  #读
    cv2.imshow("haizei0", img) #显示
    cv2.imwrite("haizei0_gray.jpg", img)  #写入保存

    cv2.waidtKey()
    cv2.destroyAllWindows()
