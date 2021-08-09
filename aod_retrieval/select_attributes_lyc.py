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
import drawing_mapping_funcs_lyc as dmf
import matplotlib.pyplot as plt
import joblib
from scipy.sparse import hstack,vstack,coo_matrix
from sklearn.model_selection import cross_val_score
from scipy import stats,special

def boxcox(x):


    return y
# 0.01 大约是1w条数据
def load_data(fname, num_data=1,m='',year=2018):
    if year==2018:
        sd = np.arange(182,366)
        start = 182
    else:
        sd = np.arange(120,153)
        start = 121
    with h5py.File(fname,'r') as f:
        for k in sd:
            try:
                xk = f[str(k)][:]
                n = int(xk.shape[0]*num_data) # 每天的有效数据量不同，按比例，取1%
                rd.seed(2018)
                index = rd.sample(range(xk.shape[0]),n)
            except:
                print('没有第{:d}天的数据'.format(k))
                continue

            if k==start:
                X = xk[index,:]
            else:
                X = np.append(X,xk[index,:],axis=0)

    x = X[:,:-1]
    y = X[:,-1]
    valid_mask1 = y > 0
    # print(x[:100,6])
    if m=='':
        valid_mask = valid_mask1
    elif m=='0.35d':
        valid_mask2 = x[:,6]<=0.35 # 按照2.13μm的表观反射率分段训练
        valid_mask = np.logical_and(valid_mask1, valid_mask2)
    elif m=='0.35u':
        valid_mask2 = x[:,6]>0.35 # 按照2.13μm的表观反射率分段训练
        valid_mask = np.logical_and(valid_mask1, valid_mask2)
    else:
        print('输入有误')
        return

    x = x[np.where(valid_mask)]
    y = y[np.where(valid_mask)]

    # 对land type做OneHot编码
    coder = OneHotEncoder(handle_unknown='ignore',sparse=True)
    lt = coder.fit_transform(x[:,-1].reshape(-1,1)) # 对land type做one-hot编码
    x = hstack([x[:,:-1],lt])

    print(x.shape)
    print(y.shape)
    # plt.hist(y,bins=200)
    # plt.show()
    return x,y


def main():
    # dt_or_db = 'dt'
    # num_data = 0.03
    # learning_rate = 0.1
    # max_depth = 5

    dt_or_db = 'db'
    num_data = 0.25
    learning_rate = 0.1
    max_depth = 5
    m = '0.35u'
    fpath = Path(r'G:\RS competition\training_dataset\final dataset\mersi_modis_{}_patch6_landtype.HDF'.format(dt_or_db))
    figpath = Path(r'G:\RS competition\figures\results of methods\juesai\scatter')
    fpath2019 = Path(r'G:\RS competition\training_dataset\final dataset\mersi_modis_{}_patch6_201905_recalibration.HDF'.format(dt_or_db))

    # model_dt = joblib.load(r'H:\RS competition\models\dt_numdata_50868_learningrate_0.10000_maxdepth_5_trainMSE_0.00646_trainMAE_0.05563_trainR2_0.92866_valMSE_0.01174_valMAE_0.07023_valR2_0.87317_patch6.m')
    # model_db_35d = joblib.load(r'H:\RS competition\models\db_numdata_0.35_down_45374_learningrate_0.10000_maxdepth_3_trainMSE_0.01879_trainMAE_0.08572_trainR2_0.66200_valMSE_0.02313_valMAE_0.09235_valR2_0.58924_patch6.m')
    # model_db_35u = joblib.load(r'H:\RS competition\models\db_numdata_0.35_up_29548_learningrate_0.10000_maxdepth_3_trainMSE_0.01813_trainMAE_0.08510_trainR2_0.73004_valMSE_0.02142_valMAE_0.09225_valR2_0.66373_patch6.m')
    x2018,y2018 = load_data(fpath, num_data,m)
    x05,y05 = load_data(fpath2019,num_data,m,year=2019)
    if x05.shape[1]>95:
        temp = x05.todense()
        temp = temp[:,:-1]
        x05 = coo_matrix(temp)

    x = vstack((x2018,x05))
    y = np.concatenate((y2018,y05))
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2018)
    print(x_train.shape[0])
    btree = GradientBoostingRegressor(learning_rate=learning_rate,
                                      n_estimators=300,
                                      max_depth=max_depth,
                                      subsample=1,
                                      min_samples_split=3,
                                      validation_fraction=0.2,
                                      n_iter_no_change=500,
                                      tol=1e-5,
                                      random_state=2018)

    btree.fit(x_train,y_train)
    # btree = model_dt

    preds_train = btree.predict(x_train)
    preds_test = btree.predict(x_test)
    # preds = btree.predict(x)
    cross_score = cross_val_score(btree,x_train,y_train,cv=5)
    print('cross validation score is:',cross_score)

    mse_train = mean_squared_error(y_train,preds_train)
    r2_train = r2_score(y_train, preds_train)
    print('训练集的MSE，R2分别为：',mse_train,r2_train)

    mse_test = mean_squared_error(y_test,preds_test)
    r2_test = r2_score(y_test,preds_test)
    print('测试集的MSE，R2分别为：',mse_test,r2_test)

    # mean = np.mean(preds)
    # bias = np.mean(preds-y)
    # mse = mean_squared_error(y,preds)
    # mae = mean_absolute_error(y,preds)
    # r2 = r2_score(y, preds)
    # print('MSE，MAE，R2，均值、偏差分别为：',mse,mae,r2,mean,bias)

    # rank = btree.feature_importances_
    # print('特征重要性排名：',rank)
    # x = rank.reshape(-1,1)
    # np.savetxt(r'H:\RS competition\log\feature_rank_db_patch6.txt',x,'%.4f')

    # 保存模型及参数
    fmodel = 'gbtree_{}_plus2019_{}.m'.format(dt_or_db,m)
    joblib.dump(btree,fmodel)
    day_or_month = 'day'
    hist_or_plot = 'plot'
    plt.figure(figsize=(8,8))
    dmf.plot_scatter(y_train,preds_train,hist_or_plot,day_or_month,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse_train,r2=r2_train)
    # figname = '{}_{}_{}'.format(dt_or_db, 'train',hist_or_plot)+m+'.png'
    # plt.savefig(figpath/figname,dpi=100)
    plt.show()

    plt.figure(figsize=(8,8))
    dmf.plot_scatter(y_test,preds_test,hist_or_plot,day_or_month,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse_test,r2=r2_test)
    # figname = '{}_{}_{}'.format(dt_or_db, 'test',hist_or_plot)+m+'.png'
    # plt.savefig(figpath/figname,dpi=100)
    plt.show()

    # hist_or_plot = 'plot'
    # plt.figure(figsize=(8,8))
    # dmf.plot_scatter(y,preds,hist_or_plot,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse,r2=r2)
    # figname = '{}_{}_{}'.format(dt_or_db, 'all',hist_or_plot)+d+'.png'
    # plt.savefig(figpath/figname,dpi=100)


    hist_or_plot = 'hist'
    plt.figure(figsize=(8,8))
    dmf.plot_scatter(y_train,preds_train,hist_or_plot,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse_train,r2=r2_train)
    figname = '{}_{}_{}'.format(dt_or_db, 'train',hist_or_plot)+m+'.png'
    plt.savefig(figpath/figname,dpi=100)

    plt.figure(figsize=(8,8))
    dmf.plot_scatter(y_test,preds_test,hist_or_plot,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse_test,r2=r2_test)
    figname = '{}_{}_{}'.format(dt_or_db, 'test',hist_or_plot)+m+'.png'
    plt.savefig(figpath/figname,dpi=100)

    # hist_or_plot = 'hist'
    # plt.figure(figsize=(8,8))
    # dmf.plot_scatter(y,preds,hist_or_plot,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse,r2=r2,mean=mean,bias=bias)
    # figname = '{}_{}_{}'.format(dt_or_db, 'all',hist_or_plot)+d+'.png'
    # plt.savefig(figpath/figname,dpi=100)


if __name__=="__main__":
    main()



