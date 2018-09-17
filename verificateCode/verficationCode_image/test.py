#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tf_verificationCode
@Product name: PyCharm
@Time: 2018/3/6 20:08
@Author: ldgang
'''

import tensorflow as tf
from param import *
from func import *
from PIL import Image
import os
import matplotlib.pyplot as plt
def modelTest():
    nameList = [fileName.split('/')[-1] for fileName in os.listdir(TEST_PATH)]
    totalNum = len(nameList)

    saver = tf.train.import_meta_graph(MODEL_PATH+'crack_captcha.model-'+str(EPOCH_NUM)+'.meta')
    graph = tf.get_default_graph()
    keepProbHolder = graph.get_tensor_by_name('keep_prob:0')
    inputHolder = graph.get_tensor_by_name('dataInput:0')
    pMaxIndexHolder = graph.get_tensor_by_name('predictMaxIndex:0')
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        count = 0
        for imageName in nameList:
            imageData = get_image_data(TEST_PATH + imageName)
            imageData = imageData.eval()
            xData = imageData.flatten() / 255
            yLabel = imageName[:CODE_NUM]  #存储的是字符
            predict = sess.run(pMaxIndexHolder, feed_dict={inputHolder:[xData],keepProbHolder:1.0})

            print(imageName)
            image = Image.open(TEST_PATH+imageName)
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            vec = np.zeros(CODE_NUM*charSetLen)
            k=0
            for i in range(len(predict[0])):
                vec[k*charSetLen+predict[0][i]] += 1
                k += 1
            predictResult = ''.join(vec2txt(vec))
            if(yLabel==predictResult):
                count += 1
            print("正确标签：{}  预测标签：{}".format(yLabel, predictResult))
        print('正确率：%.2f' %(count*1.0/totalNum))


