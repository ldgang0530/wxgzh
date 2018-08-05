#!/user/bin/env python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import stats
from sklearn.model_selection import cross_val_score

def load_datasets(file_path, file1, file2):
    cur_dir = os.getcwd()
    train_df = pd.read_csv(cur_dir+"\\"+file_path+"\\"+file1, index_col=0)
    test_df = pd.read_csv(cur_dir+"\\"+file_path+"\\"+file2, index_col=0)
    return train_df, test_df


def anova(frame, qualitative):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[1]   #pval值大于0.05就可以认为该变量对SalePrice没有作用
        pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')


def encode(x_data, frame, feature):
    ordering = pd.DataFrame()
    ordering['val'] = frame[feature].unique()
    ordering.index = ordering.val
    ordering['price_mean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
    ordering = ordering.sort_values('price_mean')
    ordering['order'] = range(1, ordering.shape[0]+1)
    ordering = ordering['order'].to_dict()
    for attr_v, score in ordering.items():
        frame.loc[frame[feature] == attr_v, feature+'_E'] = score
        x_data.loc[x_data[feature] == attr_v, feature+'_E'] = score


def spearman(frame, features):
    """
    采用斯皮尔曼相关系数计算变量与房价的相关性。相比皮尔逊系数，斯皮尔曼相关系数能更好的处理等级变量的相关性(定性的特征)
    :param frame:
    :param features:
    :return:
    """
    spr = pd.DataFrame()
    spr['feature'] = features
    spr['corr'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
    spr = spr.sort_values('corr')
    plt.figure(figsize=(6, 0.25*len(features)))
    sns.barplot(data=spr, y='feature', x='corr', orient='h')
    plt.yticks(fontsize='7')


def data_clean(train_df, test_df):
    """
    数据清洗
    :param train_df:
    :param test_df:
    :return:
    """
    #1, 训练数据，样本标签缺失，直接丢弃
    train_df_no_price = train_df.dropna(subset=["SalePrice"])
    train_df_no_price.pop('SalePrice')
    train_df_with_price = train_df.dropna(subset=["SalePrice"])
    testData_df = test_df
    all_data = pd.concat((train_df_no_price, testData_df), axis=0)
    #2，统计数据集值的缺失情况
    # print(train_df_no_price.isnull().sum().sort_values(ascending=False))
    miss_data = all_data.isnull().sum().sort_values(ascending=False)
    miss_data = miss_data[miss_data > 0]
    types = all_data[miss_data.index].dtypes
    percent = (all_data[miss_data.index].isnull().sum()
               / all_data[miss_data.index].isnull().count()).sort_values(ascending=False)
    miss_info = pd.concat([miss_data, percent, types], axis=1, keys=['Total', 'Percent', 'types'])
    miss_info.sort_values('Total', ascending=False, inplace=True)
    # print(miss_info)
    # miss_data.plot.bar()
    # 3，将缺失数目大于百分之15的特征删除都是删除
    train_df_with_price.drop((miss_info[miss_info['Percent'] > 0.15]).index, 1, inplace=True)
    testData_df.drop((miss_info[miss_info['Percent'] > 0.15]).index, 1, inplace=True)
    all_data.drop((miss_info[miss_info['Percent'] > 0.15]).index, 1, inplace=True)
    #4,定量特征与定性特征分析
    quantitative = [f for f in all_data.columns if all_data.dtypes[f] != 'object']  # 数值变量集合
    qualitative = [f for f in all_data.columns if all_data.dtypes[f] == 'object']  # 类型变量集合
    print("quantitative: {}, qualitative: {}".format(len(quantitative), len(qualitative)))
    ##4.1定量特征分析
    ## 计算定量特征与价格的相关性
    #corrmat = train_df_with_price[quantitative+['SalePrice']].corr()
    #k = len(quantitative)  # 热力图
    # cols = corrmat.nlargest(k, "SalePrice")['SalePrice'].index
    # cm = np.corrcoef(train_df_with_price[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
    #                  xticklabels=cols.values)
    # plt.show()
    # plt.close()
    #经过热力图定量相关性分析，部分特征相关性较强，可以去掉其中一部分。
    # 可以把这些特征去掉['GarageYrBlt', 'MasVnrArea', 'TotRmsAbvGrd'， 'GarageCars', '1stFlrSF']
    train_df_with_price.drop(['GarageYrBlt', 'MasVnrArea', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF'], axis=1, inplace=True)
    testData_df.drop(['GarageYrBlt', 'MasVnrArea', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF'], axis=1, inplace=True)
    all_data.drop(['GarageYrBlt', 'MasVnrArea', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF'], axis=1, inplace=True)
    for c in ['GarageYrBlt', 'MasVnrArea', 'TotRmsAbvGrd', 'GarageCars', '1stFlrSF']:
        quantitative.remove(c)
    ##分析定量特征的分布
    #f = pd.melt(train_df_with_price, value_vars=quantitative[0:8])
    #g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
    #g = g.map(sns.distplot, 'value')
    #弥补定量特征缺失值
    #print(all_data[quantitative].isnull().sum().sort_values(ascending=False))
    # 通过上述打印出定量特征缺失的情况
    # BsmtHalfBath  2  地下室半浴室  取众数补0
    # BsmtFullBath  2  地下室齐全的浴室  取众数补0
    # BsmtFinSF1   1 类型1完成平方英尺  #补均值
    # TotalBsmtSF   1  地下室总面积 #补均值
    # BsmtUnfSF   1  地下室未完成的平方英尺  #补均值
    # GarageArea   1 车库的面积，平方英尺 #补均值
    # BsmtFinSF2   1  2型成品平方英尺 补均值
    train_df_with_price[['BsmtHalfBath', 'BsmtFullBath']] = train_df_with_price[['BsmtHalfBath','BsmtFullBath']].fillna(0)
    testData_df[['BsmtHalfBath', 'BsmtFullBath']] = testData_df[['BsmtHalfBath', 'BsmtFullBath']].fillna(0)
    all_data[['BsmtHalfBath', 'BsmtFullBath']] = all_data[['BsmtHalfBath', 'BsmtFullBath']].fillna(0)
    #测试集的缺失值按照总数据集的数据统计填充
    data_mean = all_data.loc[:, ['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageArea', 'BsmtFinSF2']].mean()
    train_df_with_price[['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF',  'GarageArea', 'BsmtFinSF2']] \
        = train_df_with_price[['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF',  'GarageArea', 'BsmtFinSF2']].fillna(data_mean)
    testData_df[['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF',  'GarageArea', 'BsmtFinSF2']] \
        = testData_df[['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF',  'GarageArea', 'BsmtFinSF2']].fillna(data_mean)
    all_data[['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageArea', 'BsmtFinSF2']] \
        = all_data[['BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'GarageArea', 'BsmtFinSF2']].fillna(data_mean)
    ##4.2 定性特征分析,添加 NONE
    #print(x_data[qualitative].isnull().sum().sort_values(ascending=False))
    # GarageCond    159
    # GarageQual    159
    # GarageFinish    159
    # GarageType    157
    # BsmtCond    82
    # BsmtExposure    82
    # BsmtQual    81
    # BsmtFinType2    80
    # BsmtFinType1    79
    # MasVnrType    24
    # MSZoning   4
    # Utilities   2
    # Functional   2
    # Electrical   1
    # KitchenQual   1
    # SaleType   1
    # Exterior2nd   1
    # Exterior1st   1
    #训练集丢弃只有4个及以下缺失的记录， 测试集则填充众数值
    discard = ['MSZoning', 'Utilities', 'Functional', 'Electrical', 'KitchenQual', 'SaleType', 'Exterior2nd', 'Exterior1st']
    train_df_with_price = train_df_with_price.dropna(subset=discard)
    testData_df[discard] = testData_df[discard]\
        .fillna({'MSZoning':'RL', 'Utilities':'AllPub', 'Functional':'Typ', 'Electrical':'SBrkr',
                 'KitchenQual':'Gd', 'SaleType':'WD','Exterior2nd':'VinylSd', 'Exterior1st':'VinylSd'}) #以数目最多的值作为填充
    #其余缺失的全作为新的特征值，标记为NONE
    train_df_with_price[qualitative] = train_df_with_price[qualitative].fillna("NONE")
    testData_df[qualitative] = testData_df[qualitative].fillna("NONE")
    #方差分析
    a = anova(train_df_with_price, qualitative+['SalePrice'])
    # a['disparity'] = np.log(1. / a['pval'].values)
    # sns.barplot(data=a, x='feature', y='disparity')
    # x = plt.xticks(rotation=90)
    #将a['pval']>0.05的特征剔除
    #train_df_with_price.drop(a[a['pval'] > 0.05]['feature'], 1, inplace=True)
    #testData_df.drop(a[a['pval'] > 0.05]['feature'], 1, inplace=True)
    # 为定性变量重新编码，比one_hot较有优势
    qualitative = [f for f in train_df_with_price.columns if train_df_with_price.dtypes[f] == 'object']  # 类型变量集合
    qual_encoded = []
    for q in qualitative:
        encode(testData_df, train_df_with_price, q)
        qual_encoded.append(q + "_E")
    #print(qual_encoded)
    train_df_with_price.drop(qualitative, axis=1, inplace=True)
    testData_df.drop(qualitative, axis=1, inplace=True)
    #定性特征热力图
    # corrmat = train_df_with_price[qual_encoded+['SalePrice']].corr()
    # k = len(qual_encoded)  # 热力图
    # cols = corrmat.nlargest(k, "SalePrice")['SalePrice'].index
    # cm = np.corrcoef(train_df_with_price[cols].values.T)
    # sns.set(font_scale=1.25)
    # hm = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
    #                   xticklabels=cols.values)
    # plt.show()
    # plt.close()
    # 结合热力图，去掉部分相关性较强的特征
    #['Exterior1st_E', 'Exterior2nd_E'],['GarageQual_E', 'GarageCond_E']:二选一
    train_df_with_price.drop(['Exterior2nd_E', 'GarageCond_E'], axis=1, inplace=True)
    testData_df.drop(['Exterior2nd_E', 'GarageCond_E'], axis=1, inplace=True)

    #5.斯皮尔曼相关
    #features = [f for f in train_df_no_price.columns]
    #spearman(train_df_with_price, features)
    return train_df_with_price, testData_df


def rmse_cv(clf, X_train, y_train, cv_value):
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=cv_value, scoring="neg_mean_squared_error"))
    return test_score



if __name__ == "__main__":
    print("aaaaa")
    train_df, test_df = load_datasets("Data", "train.csv", "test.csv")
    data_clean(train_df, test_df)
    print("haha")
