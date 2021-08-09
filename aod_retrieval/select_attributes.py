'''
@ ! D:/Python366
@ -*- coding:utf-8 -*-
@ Time: 2019/7/14, 9:49
@ Author: LiangHongli
@ Mail: l.hong.li@foxmail.com
@ File: select_attributes.py
@ Software: PyCharm
尝试输入所有可能有用的变量，用树得到的结果查看哪些变量重要，从中选择权重较大的变量
'''
from pathlib import Path
import h5py
import numpy as np
import random as rd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import drawing_mapping_funcs as dmf
import matplotlib.pyplot as plt
import joblib

def load_data(fname):

    with h5py.File(fname,'r') as f:
        for k in range(182,366):
            try:
                xk = f[str(k)][:]
                n = int(xk.shape[0]*0.03) # 每天的有效数据量不同，按比例，取1%
                rd.seed(2018)
                index = rd.sample(range(xk.shape[0]),n)
            except:
                print('没有第{:d}天的数据'.format(k))
                continue

            if k==182:
                X = xk[index,:]
            else:
                X = np.append(X,xk[index,:],axis=0)

    # 对land type做OneHot编码
    coder = OneHotEncoder(handle_unknown='ignore',sparse=False)
    lt = coder.fit_transform(X[:,-2].reshape(-1,1)) # 对land type做one-hot编码
    x = np.concatenate((X[:,:-2],lt),axis=1)
    y = X[:,-1]
    print(x.shape)
    return x,y

def main():
    fpath = Path(r'H:\RS competition\training_dataset\final dataset\mersi_modis_db_patch6_landtype.HDF')
    x,y = load_data(fpath)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2018)
    print(x_train.shape[0])
    btree = GradientBoostingRegressor(learning_rate=0.1,
                                      n_estimators=300,
                                      max_depth=3,
                                      subsample=1,
                                      min_samples_split=3,
                                      validation_fraction=0.2,
                                      n_iter_no_change=500,
                                      tol=1e-5,
                                      random_state=2018)

    btree.fit(x_train,y_train)
    preds_train = btree.predict(x_train)
    preds_test = btree.predict(x_test)

    mse_train = mean_squared_error(y_train,preds_train)
    mae_train = mean_absolute_error(y_train,preds_train)
    r2_train = r2_score(y_train,preds_train)
    print('训练集的MSE，MAE，R2分别为：',mse_train,mae_train,r2_train)

    mse_test = mean_squared_error(y_test,preds_test)
    mae_test = mean_absolute_error(y_test,preds_test)
    r2_test = r2_score(y_test,preds_test)
    print('测试集的MSE，MAE，R2分别为：',mse_test,mae_test,r2_test)
    rank = btree.feature_importances_
    print('特征重要性排名：',rank)
    x = rank.reshape(-1,1)
    # np.savetxt(r'H:\RS competition\log\feature_rank_db_patch6.txt',x,'%.4f')

    # 保存模型及参数
    # fmodel = 'gbtree_juesai_dt_patch6.m'
    # joblib.dump(btree,fmodel)

    plt.figure(1)
    dmf.plot_scatter(y_train,preds_train,'ground truth AOD','GBT predicted AOD',mse=mse_train,mae=mae_train,r2=r2_train)
    plt.figure(2)
    dmf.plot_scatter(y_test,preds_test,'ground truth AOD','GBT predicted AOD',mse=mse_test,mae=mae_test,r2=r2_test)

if __name__=="__main__":
    main()



