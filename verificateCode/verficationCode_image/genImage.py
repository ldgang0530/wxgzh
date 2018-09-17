#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tf_verificationCode
@Product name: PyCharm
@Time: 2018/3/6 12:42
@Author: ldgang
'''
from captcha.image import ImageCaptcha
import random
import numpy as np
from PIL import Image
import os
import random
import shutil
import time
from param import *

#根据设定的图片数目，生成图片集，文件名称为验证码内容
def genImage(imageNum,imagePath, charSet=globCharSet):
    def getImageTxt(): #获取image上的验证码
        txtList = []
        for i in range(CODE_NUM):
            c = random.choice(charSet) #从字符集中随机选取一个字符
            txtList.append(c)
        return txtList

    def getImageWithTxt(txt): #生成图片
        image = ImageCaptcha()
        imageTxt = ''.join(txt)
        while(1):
            captcha=image.generate(imageTxt,'JPEG')
            captchaImage = Image.open(captcha)
            captChaImagDim = np.array(captchaImage).shape
            if captChaImagDim == (IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS):  #判断生成的图片是否满足60*160*3
                captchaImage.save(imagePath + imageTxt + '.jpg','JPEG')
                break

    for i in range(imageNum):
        txt = getImageTxt()
        getImageWithTxt(txt)


def getTestSet(testNum,imagePath,testPath):  #从图片集中提取部分图片作为训练集
    fileNameList = []  #存储目录下的文件名，不包含后缀
    for filePath in os.listdir(imagePath):  # 从图像路径下读取图像文件
        if filePath.endswith('jpg'):
            captcha_name = filePath.split('/')[-1]
            fileNameList.append(captcha_name)
    random.seed(time.time())  # 设置random种子
    random.shuffle(fileNameList)  # 打乱文件名
    for i in range(testNum):
        name = fileNameList[i]
        shutil.move(imagePath + name, testPath + name)  # 将测试集移动到测试目录下
'''
if __name__ == '__main__':
    genImage(IMAGE_NUM,IMAGE_PATH)
    getTestSet(TEST_NUM,IMAGE_PATH,TEST_PATH)
'''
