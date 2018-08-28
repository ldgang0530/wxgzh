#!/user/bin/env python

import cv2
import GeneralFunc


if __name__ == "__main__":
    hog = cv2.HOGDescriptor()  #初始化hog描述子
    hog.load('myHogDector.bin')  #加载训练好的检测器
    #cap = cv2.VideoCapture(0)

    base_path = 'D:/Samples/DataSets/INRIAPerson/'
    test_img_list = GeneralFunc.load_imge(base_path+"Test/pos.lst", imageNum=10)  #加载图片

    for i in range(len(test_img_list)):
        img = test_img_list[i]
        rects,wei = hog.detectMultiScale(img, winStride=(4,4), padding=(8,8), scale=1.05)   #进行检测
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  #标记检测的结果
        cv2.imshow("a"+str(i), img)
        cv2.waitKey(10)  #打开部分文件时，需要部分处理时间
        if cv2.waitKey(1)&0xff == 27:  #按"q"键退出
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()