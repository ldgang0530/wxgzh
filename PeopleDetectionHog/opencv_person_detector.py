#!/user/bin/env python
import cv2
import GeneralFunc


if __name__ == "__main__":
    base_path = 'D:/Samples/DataSets/INRIAPerson/'
    test_img_list = GeneralFunc.load_imge(base_path + "Test/pos.lst", imageNum=10) #加载图像，pos.lst中存储了图片名称
    hog = cv2.HOGDescriptor()  #初始化HOG描述子
    # 设置支持向量机，使其称为一个预先训练好的行人检测器
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    for i in range(len(test_img_list)):
        img = test_img_list[i]
        rects, wei = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05) #检测行人
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) #将检测结果在图像中圈出来
        cv2.imshow("a" + str(i), img) #显示图像
        cv2.waitKey(10)  # 显示文件时，有些格式的文件需要时间处理
        if cv2.waitKey(1) & 0xff == 27: #按 "q" 键退出
            break
    cv2.waitKey(0)  #按任意键退出
    cv2.destroyAllWindows() #清除所有的窗口
