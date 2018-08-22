#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: opencv_basic_handle
@Product name: PyCharm
@Time: 2018/8/20 21:22
@Author: ldgang
'''
import cv2
import numpy as np


#不进行任何操作
def nothing(x):
    pass

def _init():  #初始化函数
    cv2.namedWindow('videoWindow', cv2.WINDOW_NORMAL)  #定义videocWindow
    cv2.namedWindow('imgeWindow', cv2.WINDOW_NORMAL)  #定义imgeWindow
    cv2.namedWindow('toolWindow', cv2.WINDOW_NORMAL)  #定义toolWindow

    image_path = "img/haizei2.jpg"   #不进行操作时，初始显示的图片
    img = cv2.imread(image_path, 1)
    cv2.imshow('videoWindow', img)
    cv2.imshow('imgeWindow', img)

    #screen capture
    cv2.createTrackbar('Capture', 'toolWindow', 0, 1, nothing)  #定义Capture按钮，值取0或1，1表示截图
    #brightness
    cv2.createTrackbar("Brightness", 'toolWindow', 0, 255, nothing)  #定义Brightness按钮，取值范围时0~255
    cv2.setTrackbarPos("Brightness", "toolWindow", 125)  #设置亮度初值，图片的亮度 = 原图像亮度 + pos - 125
    #record video
    cv2.createTrackbar("RecordVideo", 'toolWindow', 0, 1, nothing)  #定义RecordVideo按钮，1表示录视频
    #bgr2gray
    cv2.createTrackbar("Colour", 'toolWindow', 0, 1, nothing)  #定义Colour按钮，1表示灰度图像；0表示彩色图像


if __name__ == "__main__":
    _init()  #初始化函数
    img_cnt = 0  #用于图片的命名
    cap = cv2.VideoCapture(0)      #视频来源，0表示默认的摄像头
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  #视频编码格式
    out = cv2.VideoWriter('out.avi', fourcc, 20, (640, 480))  #创建VideoWriter对象，定义视频存储路径
    face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_alt.xml')  #已经训练好的人脸检测Haar特征级联分类器
    while cap.isOpened():
        ret, frame = cap.read()  #按帧读取图像
        if ret :
            #gray用于人脸检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #将图像转为灰度图像
            # 人脸检测识别
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(5, 5)
            )

            #黑白或彩色
            colourFlag = cv2.getTrackbarPos("Colour", "toolWindow")  #获取colour按钮的值，判断是否需要灰度化
            if colourFlag == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #亮度调整
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  #将BGR转为HSV，主要用于调节亮度
            BrightNum = cv2.getTrackbarPos("Brightness", "toolWindow")
            frame[:,:,2] = frame[:,:,2] + BrightNum-125
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)  #再由HSV转为BGR
            #添加文字
            cv2.putText(frame,"AI-START", (450, 40),1, 2, (255, 255, 255), 2)  #创建标签

            #在视频中圈出人脸
            show_frame = frame
            for (x, y, w, h) in faces:
                cv2.circle(show_frame, (int((x + x + w) / 2), int((y + y + h) / 2)), int(w / 2), (0, 255, 0), 2)
            cv2.imshow('videoWindow', show_frame)

            #录制视频
            SAVE_FLAG = cv2.getTrackbarPos('RecordVideo', 'toolWindow')
            if SAVE_FLAG == 1:
                out.write(frame)
            #截屏
            CUT_FLAG = cv2.getTrackbarPos('Capture', 'toolWindow')
            if CUT_FLAG == 1:
                cv2.imshow('imgeWindow', frame)    #在imageWindow中显示截取的图片
                cv2.setTrackbarPos('Capture', 'toolWindow', 0)
                cv2.imwrite('cut_img_'+str(img_cnt)+".jpg", frame)  #保存图片
                img_cnt = img_cnt + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):   #按'q'键退出程序
            break
    cap.release()  #关闭摄像头
    out.release()  #关闭videoWriter
    cv2.destroyAllWindows()  #关闭所有窗口
