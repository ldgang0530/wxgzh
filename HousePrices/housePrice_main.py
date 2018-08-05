#!/user/bin/env python

from sklearn.linear_model import Ridge,LassoCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import os
import pandas as pd
import xgboost as xgb

import housePrices_func as hp



if __name__ == "__main__":
    cur_dir = os.getcwd()
    ####数据的清洗，只需要执行一次就可以，得到预处理数据，保存路径Data\\pre_train_df.csv，Data\\pre_test_df.csv ###
    train_df, test_df = hp.load_datasets("Data", "train.csv", "test.csv")
    pre_train_df, pre_test_df = hp.data_clean(train_df, test_df)
    save_file_df = pd.DataFrame(data=pre_train_df)
    save_file_df.to_csv(cur_dir + '\\Data\\pre_train_df.csv', columns=[f for f in pre_train_df.columns], index=True)
    save_file_df = pd.DataFrame(data=pre_test_df)
    save_file_df.to_csv(cur_dir + '\\Data\\pre_test_df.csv', columns=[f for f in pre_test_df.columns], index=True)
    #加载预处理数据
    pre_train_df, pre_test_df = hp.load_datasets("Data", "pre_train_df.csv", "pre_test_df.csv")
    y_train = np.log1p(pre_train_df.pop("SalePrice"))
    X_train = pre_train_df
    # 设定模型、训练
    #LassoCV
    clf1 = LassoCV(alphas=[1, 0.1, 0.05, 0.01, 0.005, 0.003, 0.001])
    clf1.fit(X_train, y_train)
    lasso_preds = np.expm1(clf1.predict(pre_test_df))
    score1 = hp.rmse_cv(clf1, X_train, y_train, 5)
    print("\nLasso score: {:.4f} ({:.4f})\n".format(score1.mean(), score1.std()))
    #ELASTIC NET
    clf2 = ElasticNet(alpha=0.001, l1_ratio=0.9)
    clf2.fit(X_train, y_train)
    elas_preds = np.expm1(clf2.predict(pre_test_df))
    score2 = hp.rmse_cv(clf2, X_train, y_train, 5)
    print("\nELASTIC score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))
    #XGBOOST
    clf3 = xgb.XGBRegressor(colsample_bytree=0.4,
                            gamma=0.045,
                            learning_rate=0.07,
                            max_depth=20,
                            min_child_weight=1.5,
                            n_estimators=300,
                            reg_alpha=0.65,
                            reg_lambda=0.45,
                            subsample=0.95)
    clf3.fit(X_train, y_train)
    xgb_preds = np.expm1(clf3.predict(pre_test_df))
    score3 = hp.rmse_cv(clf3, X_train, y_train, 5)
    print("\nXGBOOST score: {:.4f} ({:.4f})\n".format(score3.mean(), score3.std()))

    #模型融合
    y_final = 0.4*lasso_preds+0.3*xgb_preds+0.3*elas_preds
    # 保存结果
    cur_dir = os.getcwd()
    submission_df = pd.DataFrame(data={'Id': pre_test_df.index, 'SalePrice': y_final})
    submission_df.to_csv(cur_dir + '\\submission.csv', columns=['Id', 'SalePrice'], index=False)