#!/user/bin/env python
import GBDT
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset():
    x_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_train = [5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]
    return x_train, y_train

def main():
    x, y = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=6)
    x_train = np.mat(x_train)
    x_test = np.mat(x_test)
    y_train = np.mat(y_train)
    y_test = np.mat(y_test)
    gbdt = GBDT.GBDT()
    regress_dict = gbdt.train(x_train, y_train, threshold=0.05, max_iter_num=20)

    y_predict = gbdt.predict_fun(regress_dict, x_test)
    loss = gbdt.loss(y_test, y_predict)
    print("\n测试情况：\n测试数据：x="+str(x_test)+"\n对应y="+str(y_test)+"\n预测值y_predict="+str(y_predict)+"\n测试数据误差："+str(loss))

    test_x = np.mat([5.5])
    test_y_predict = gbdt.predict_fun(regress_dict, test_x)
    print("\n预测：\n输入X="+str(test_x)+", 预测y_predict="+str(test_y_predict))


if __name__ == "__main__":
    main()
