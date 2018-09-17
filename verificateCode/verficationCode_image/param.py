#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tf_verificationCode
@Product name: PyCharm
@Time: 2018/3/6 16:21
@Author: ldgang
'''
number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
globCharSet = number+alphabet+ALPHABET #字符集
charSetLen = len(globCharSet) #字符集长度

CODE_NUM = 4 #4位的验证码
IMAGE_WIDTH = 160 #验证码宽度
IMAGE_HEIGHT = 60 #验证码高度
IMAGE_CHANNELS = 3 #RGB三色

IMAGE_PATH = 'Captcha/images/'  #图片路径
TEST_PATH = 'Captcha/test/' #测试集路径
MODEL_PATH = 'Captcha/models/' #训练完成后model存放路径

IMAGE_NUM = 10000  #产生的图片集数目
TEST_NUM = 1000 #测试集图片数目，从图片集中提取，因此应小于IMAGE_NUM

VALIDATE_PRECENT = 0.7 #样本从IMAGE_PATH中获取，训练集与验证集的比例
EPOCH_NUM = 40 #样本遍历的次数
BATCH_SIZE = 100