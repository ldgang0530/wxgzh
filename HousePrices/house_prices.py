#!/user/bin/env python
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def load_datasets(file_path):
    cur_dir = os.getcwd()
    test_df = pd.read_csv(cur_dir+"\\"+file_path+"\\test.csv", index_col=0)
    train_df = pd.read_csv(cur_dir+"\\"+file_path+"\\train.csv", index_col=0)
    return train_df, test_df


def data_clean(train_df, test_df):
    #获取训练数据样本的价格，并从train_df中去除
    y_train = np.log1p(train_df.pop("SalePrice"))
    #将训练样本与测试样本的特征合并，便于一块儿进行数据清洗
    all_df = pd.concat((train_df, test_df), axis=0)

    #print(all_df["MSSubClass"].value_counts())  #计算各个离散值的数目
    #print(pd.get_dummies(all_df["MSSubClass"], prefix="MSSubClass").head(5))

    #离散变量 转换为 独热码
    all_dummy_df = pd.get_dummies(all_df)
    #print(all_dummy_df.head(5))

    #处理缺失值，添加均值
    #all_dummy_df.isnull().sum().sort_values(ascending=False).head()
    mean_cols = all_dummy_df.mean()
    all_dummy_df = all_dummy_df.fillna(mean_cols)  #将缺失的数据赋上均值
    #标准化连续值的属性
    numberic_cols = all_df.columns[all_df.dtypes != 'object']  #把值为0/1的列剔除
    numberic_cols_means = all_dummy_df.loc[:, numberic_cols].mean()
    numberic_cols_std = all_dummy_df.loc[:, numberic_cols].std()
    all_dummy_df.loc[:, numberic_cols] = (all_dummy_df.loc[:, numberic_cols] - numberic_cols_means)/numberic_cols_std

    train_dummpy_df = all_dummy_df.loc[train_df.index] #训练数据
    test_dumpy_df = all_dummy_df.loc[test_df.index] #测试数据
    return train_dummpy_df, test_dumpy_df, y_train


def train_ridge(train_dummy_df, y_train):
    X_train = train_dummy_df.values
    alphas = np.logspace(-3, 2, 50)
    test_scores = []
    for alpha in alphas:
        clf = Ridge(alpha)   #alpha L2正则化项，值越大，正则化项越大
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train , cv = 10, scoring="neg_mean_squared_error"))
        test_scores.append(np.mean(test_score))
    return alphas, test_scores


def train_random_forest(train_dummy_df, y_train):
    X_train = train_dummy_df.values
    max_features = [.1, .3, .5, .7, .9, .99]
    test_scores = []
    for max_feat in max_features:
        clf = RandomForestRegressor(n_estimators=200, max_features=max_feat)
        test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=5, scoring="neg_mean_squared_error"))
        test_scores.append(np.mean(test_score))
    return max_features, test_scores


def display_param(x, y, title):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    #读取数据
    train_df, test_df = load_datasets("Data")
    #数据清洗
    train_dummpy_df, test_dumpy_df, y_train = data_clean(train_df, test_df)
    #模型建立、选定超参
    #print("Ridge begin")
    alphas, test_scores_1 = train_ridge(train_dummpy_df, y_train)
    #display_param(alphas, test_scores_1, "ridge: alphas vs test_scores")
    #print("Ridge end")
    #print("Forest begin")
    max_features, test_scores_2 = train_random_forest(train_dummpy_df, y_train)
    #print("Forest end")

    ridge_score = np.min(test_scores_1)
    ridge_alpha = alphas[np.argmin(test_scores_1)]
    random_forest_score = np.min(test_scores_2)
    random_forest_features = max_features[np.argmin(test_scores_2)]
    #设定模型、训练
    ridge = Ridge(alpha=ridge_alpha)
    random_forest = RandomForestRegressor(n_estimators=500, max_features=random_forest_features)
    ridge.fit(train_dummpy_df, y_train)
    random_forest.fit(train_dummpy_df, y_train)
    #预测
    y_ridge = np.expm1(ridge.predict(test_dumpy_df))
    y_rf = np.expm1(random_forest.predict(test_dumpy_df))
    #y_final = (y_ridge + y_rf) / 2   #两个模型输出求平均
    y_final = ridge_score/(ridge_score+random_forest_score)*y_rf+random_forest_score/(ridge_score+random_forest_score)*y_ridge
    #保存结果
    cur_dir = os.getcwd()
    submission_df = pd.DataFrame(data={'Id': test_df.index, 'SalePrice': y_final})
    submission_df.to_csv(cur_dir+'\\submission.csv', columns=['Id', 'SalePrice'], index=False)
    input()