#! /user/bin/env python
import cv2
import random


def load_imge(dirname, imageNum = 500, showImage = False):
    '''
    加载图像
    :param dirname: 存储图像名称的文件，如pos.lst
    :param imageNum:  要读取的图像数目
    :param showImage:  是否显示图像
    :return: 返回存储图像内容的list
    '''
    img_list = []
    cnt  = imageNum
    file = open(dirname) #
    img = file.readline()
    while img != '':  #文件末尾
        img_name = dirname.rsplit(r'/',1)[0] + r'/'+ img.split('/', 1)[1].strip('\n')  #图像路径
        img_content = cv2.imread(img_name,0) #将图像读取，并转为灰度图像
        img_list.append(img_content)
        if showImage: #是否显示图像，默认否
            cv2.imshow(img_name, img_content)
            cv2.waitKey(10)
        cnt = cnt-1
        if cnt == 0:
            break
        img = file.readline()
    return img_list  #返回所有图像的内容


def sample_neg(neg_sample_all, imageNum = 500, size=(64,128)):
    """
    获取负样例，从没有行人的图片中截取64*128的图像作为训练负样例
    :param neg_sample_all:  所有没有行人的图像
    :param imageNum:  负样本的数目
    :param size:  截取的图像的大小
    :return:  返回负样本图像的内容
    """
    neg_sample_list = []
    cnt = imageNum
    width, height = size[0], size[1]  #图像的宽度和高度
    for i in range(len(neg_sample_all)):
        row, col = neg_sample_all[i].shape
        #for j in range(10):
        y = int(random.random()*(row - height)) #随机选择图像截取的起点，也就是说从图像中随机截取的图像
        x = int(random.random()*(col - width))
        neg_sample_list.append(neg_sample_all[i][y:y+height, x:x+width])  #截取图像
        cnt = cnt-1
        if cnt == 0:
            break
    return neg_sample_list
