#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tf_verificationCode
@Product name: PyCharm
@Time: 2018/3/6 19:57
@Author: ldgang
'''

from genImage import *
from train import *
from test import *
if __name__ == '__main__':
    #genImage(IMAGE_NUM,IMAGE_PATH,globCharSet)
    #getTestSet(TEST_NUM,IMAGE_PATH,TEST_PATH)

    train()
#    modelTest()