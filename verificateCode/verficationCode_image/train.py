#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
@Project name: tf_verificationCode
@Product name: PyCharm
@Time: 2018/3/6 16:42
@Author: ldgang
'''

import os
from func import *
import random
import numpy as np

def getImageInfo(fileNameList, trainFlag):
    fileNum = len(fileNameList)
    imageData = np.zeros([fileNum, IMAGE_HEIGHT * IMAGE_WIDTH])
    imageLabel = np.zeros([fileNum, CODE_NUM * charSetLen])
    if trainFlag:
        filePath = IMAGE_PATH
    else:
        filePath = TEST_PATH
    for i in range(fileNum):
        image = get_image_data(filePath + fileNameList[i])
        image = image.eval()
        imageData[i, :] = image.flatten() / 255
        # batchY
        imageLabel[i, :] = get_image_label(fileNameList[i])
        print("iter:",i)
    return imageData, imageLabel

def get_batch_info(iter, data, label, batchSize=128):
    batchX = np.zeros([batchSize, IMAGE_HEIGHT * IMAGE_WIDTH])
    batchY = np.zeros([batchSize, CODE_NUM * charSetLen])
    totalImageNum = len(data)
    iterNum = iter * batchSize
    for i in range(batchSize):
        id = (iterNum+i)%totalImageNum
        batchX[i,:] = data[id]
        batchY[i,:] = label[id]
    return batchX, batchY

def get_batch_info_first(iter=0, batchSize=128, fileNameList=[], trainImageData=[],trainImageLabel=[],trainFlag=True): #获取一个batch
    batchX = np.zeros([batchSize,IMAGE_HEIGHT*IMAGE_WIDTH])
    batchY = np.zeros([batchSize,CODE_NUM*charSetLen])
    tmpXData = np.zeros([batchSize,IMAGE_HEIGHT*IMAGE_WIDTH])
    tmpYData = np.zeros([batchSize,CODE_NUM*charSetLen])
    totalImageNum = len(fileNameList)
    iterNum = iter*batchSize
    if trainFlag:
        filePath = IMAGE_PATH
    else:
        filePath = TEST_PATH

    data_len = len(trainImageData)
    for i in range(batchSize):
        #batchX
        fileName = fileNameList[(iterNum+i)%totalImageNum]  #若超出了总数目就从头开始取
        image = get_image_data(filePath+fileName)
        image = image.eval()
        batchX[i, :] = image.flatten()/255
        #batchY
        tlabel = get_image_label(fileName)
        batchY[i, :] = tlabel

        tmpXData[i, :] = image.flatten() / 255
        tmpYData[i, :] = tlabel
    trainImageData.extend(tmpXData)
    trainImageLabel.extend(tmpYData)
    return batchX, batchY



def cnn(X,keep_prob):

    def weight_var(shape, name='weight'): #权重矩阵
        init = tf.truncated_normal(shape, stddev=0.01)
        var = tf.Variable(initial_value=init, name=name)
        return var

    def bias_var(shape, name='bias'): #偏置矩阵
        init = tf.truncated_normal(shape, stddev = 0.1)
        var = tf.Variable(initial_value=init, name=name)
        return var

    def conv2d(x, w,name='conv2d'):  #卷积
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', name=name)

    def max_pool(value, name='max_pool'): #最大池化
        return tf.nn.max_pool(value, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME', name=name)

    #输入层
    x_input = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='x-input')
    #第一层 60*160*1 ,图片转换为灰度图片
    w1 = weight_var([3,3,1,32],name='w1')
    b1 = bias_var([32],name='b1')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_input,w1,name='conv1'),b1))  #Relu作为激活函数
    conv1_pool = max_pool(conv1,name='max_pool1') #池化
    conv1_drop = tf.nn.dropout(conv1_pool,keep_prob=keep_prob) #dropout
    #第二层 #30*80*32
    w2 = weight_var([3,3,32,64],name='w2')
    b2 = bias_var([64],name='b2')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(conv1_drop, w2, name='conv2') , b2))  # Relu作为激活函数
    conv2_pool = max_pool(conv2, name='max_pool2')  # 池化
    conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)  # dropout
    #第三层 #15*40*64
    w3 = weight_var([3, 3, 64, 64], name='w3')
    b3 = bias_var([64], name='b3')
    conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(conv2_drop, w3, name='conv3') ,b3))  # Relu作为激活函数
    conv3_pool = max_pool(conv3, name='max_pool3')  # 池化
    conv3_drop = tf.nn.dropout(conv3_pool, keep_prob=keep_prob)  # dropout

    #全连接层
    w = int(conv3_drop.shape[1])
    h = int(conv3_drop.shape[2])
    w4 = weight_var([w*h*64, 1024], name='w4')
    b4 = bias_var([1024], name='b4')
    fc1 = tf.reshape(conv3_drop, [-1, w* h * 64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, w4), b4))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #输出层
    w_out = weight_var([1024, CODE_NUM * len(globCharSet)], 'w_out')
    b_out = bias_var([CODE_NUM * len(globCharSet)], 'b_out')
    out = tf.add(tf.matmul(fc1, w_out), b_out, 'output')
    return out

def train():
    import time
    start_time = time.time()
    fileNameList = [filePath.split('/')[-1] for filePath in os.listdir(IMAGE_PATH)]  # 从图像路径下读取图像文件
    fileNum = len(fileNameList)
    random.seed(start_time)
    random.shuffle(fileNameList)
    trainNum = int(fileNum*VALIDATE_PRECENT)
    trainImageNameList = fileNameList[0:trainNum]
    validateImageNameList = fileNameList[trainNum:]

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH], name='dataInput')
    Y = tf.placeholder(tf.float32, [None, CODE_NUM*charSetLen], name='labelInput')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')

    output = cnn(X, keep_prob)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, CODE_NUM, charSetLen], name='predict')
    labels = tf.reshape(Y, [-1, CODE_NUM, charSetLen], name='labels')

    p_max_idx = tf.argmax(predict, 2, name='predictMaxIndex')
    l_max_idx = tf.argmax(labels,2,name='labelMaxIndex')
    equalVec = tf.equal(p_max_idx, l_max_idx)
    accuracy = tf.reduce_mean(tf.cast(equalVec, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 将所有图像读入内存（1万张图像大约40M，可以接受）
        #trainImageData, trainImageLabel = getImageInfo(trainImageNameList, True)
        #testImageData, testImageLabel = getImageInfo(validateImageNameList, True)
        testImageData = []
        testImageLabel = []
        trainImageData = []
        trainImageLabel = []

        acc_rate = 0.90
        targetArrived = False
        acc = 0.0
        for epoch in range(EPOCH_NUM):
            steps = 0
            while steps*BATCH_SIZE < len(trainImageNameList):
                if(epoch > 0):
                    train_data, train_label = get_batch_info(steps, trainImageData, trainImageLabel, batchSize=BATCH_SIZE)
                else:
                    train_data, train_label = get_batch_info_first(steps, BATCH_SIZE, trainImageNameList, trainImageData, trainImageLabel, True)

                sess.run(optimizer, feed_dict={X: train_data, Y: train_label, keep_prob: 0.7})
                if steps % 100 == 0:
                    if(steps/100*BATCH_SIZE < len(validateImageNameList)):
                        test_data, test_label = get_batch_info_first(steps, BATCH_SIZE, validateImageNameList,
                                                                     testImageData, testImageLabel, True)
                    else:
                        test_data, test_label = get_batch_info(steps, testImageData, testImageLabel, BATCH_SIZE)

                    acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
                    print("steps=%d, accuracy=%f" % (steps, acc))
                    if acc >= acc_rate:
                        saver.save(sess, MODEL_PATH + str(epoch+acc) +"crack_captcha.model", global_step=steps)
                        acc_rate += 0.1
                        if acc>=0.97:
                            targetArrived = True
                            break
                steps = steps + 1
                print("steps", steps)
            if(targetArrived):
                break
            if(epoch>=5 and ((epoch%10 == 0) or (epoch == EPOCH_NUM-1))):
                saver.save(sess, MODEL_PATH + str(epoch+acc) +"epoch_crack_captcha.model", global_step=steps)
            print('Epoch:',epoch,'  steps:', steps)


if __name__ == '__main__':
    train()
    print("haha")

