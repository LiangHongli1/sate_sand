'''
@ ! D:/Python366
@ -*- coding:utf-8 -*-
@ Time: 28/07/2019, 18:03
@ Author: LiangHongli
@ Mail: l.hong.li@foxmail.com
@ File: validation_plotting.py
@ Software: PyCharm
'''
from pathlib import Path
import h5py
import numpy as np
import random as rd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import drawing_mapping_funcs_lyc as dmf
import matplotlib.pyplot as plt
import joblib
from scipy.sparse import hstack,coo_matrix
import time


# 加载需要验证的数据
def load_data(fname, date,m):

    if len(date)==8: # 获取单天的数据
        struct = time.strptime(date,'%Y%m%d')
        dia = time.strftime('%Y%j',struct)[-3:]
        with h5py.File(fname,'r') as f:
            try:
                X = f[dia][:]
            except:
                print('没有第{:d}天的数据'.format(int(dia)))
                return
    elif len(date)==6: # 获取某个月的数据
        struct = time.strptime(date+'01','%Y%m%d')
        dia = time.strftime('%Y%j',struct)[-3:]
        with h5py.File(fname,'r') as f:
            for k in range(int(dia),int(dia)+31):
                try:
                    xk = f[str(k)][:]
                except:
                    print('没有第{:d}天的数据'.format(int(dia)))
                    continue

                if k==int(dia):
                    X = xk
                else:
                    X = np.append(X,xk,axis=0)

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
    if x.shape[0]==0:
        return
    # 对land type做OneHot编码
    coder = OneHotEncoder(handle_unknown='ignore',sparse=True)
    lt = coder.fit_transform(x[:,-1].reshape(-1,1)) # 对land type做one-hot编码
    x = hstack([x[:,:-1],lt])

    print(x.shape)
    print(y.shape)
    # plt.hist(y,bins=200)
    # plt.show()
    return x,y

# 计算各项评估指标
def calc_score(preds,y):
    mean = np.mean(preds)
    bias = np.mean(preds-y)
    mse = mean_squared_error(y,preds)
    r2 = r2_score(y, preds)
    print('MSE，R2，均值、偏差分别为：',mse,r2,mean,bias)

    return mean,bias,mse,r2


def main():
    fpath = Path(r'G:\RS competition\training_dataset\final dataset\mersi_modis_{}_patch6_201905.HDF'.format('dt'))
    dbpath = Path(r'G:\RS competition\training_dataset\final dataset\mersi_modis_{}_patch6_201905.HDF'.format('db'))
    figpath = Path(r'G:\RS competition\figures\results of methods\juesai\scatter\day_month\2019_new_model')

    # model_dt = joblib.load(r'G:\RS competition\models\dt_numdata_50868_learningrate_0.10000_maxdepth_5_trainMSE_0.00646_trainMAE_0.05563_trainR2_0.92866_valMSE_0.01174_valMAE_0.07023_valR2_0.87317_patch6.m')
    # model_db_35d = joblib.load(r'G:\RS competition\models\db_numdata_0.35_down_45374_learningrate_0.10000_maxdepth_3_trainMSE_0.01879_trainMAE_0.08572_trainR2_0.66200_valMSE_0.02313_valMAE_0.09235_valR2_0.58924_patch6.m')
    # model_db_35u = joblib.load(r'G:\RS competition\models\db_numdata_0.35_up_29548_learningrate_0.10000_maxdepth_3_trainMSE_0.01813_trainMAE_0.08510_trainR2_0.73004_valMSE_0.02142_valMAE_0.09225_valR2_0.66373_patch6.m')

    # 试用新的用了201905数据的模型做2019年5月的验证
    model_dt = joblib.load(r'G:\RS competition\models\gbtree_dt_plus2019_.m')
    model_db_35d = joblib.load('G:\RS competition\models\gbtree_db_plus2019_0.35d.m')
    model_db_35u = joblib.load('G:\RS competition\models\gbtree_db_plus2019_0.35u.m')

    day_or_month = 'day'
    # 预测单天的AOD
    for d in validation[day_or_month]:
        print(d)
        xdt, ydt = load_data(fpath,d,m='')
        pred_dt = model_dt.predict(xdt)

        xdbd, ydbd = load_data(dbpath,d,m='0.35d')
        if xdbd.shape[1]>95:
            temp = xdbd.todense()
            temp = temp[:,:-1]
            xdbd = coo_matrix(temp)
        pred_db_35d = model_db_35d.predict(xdbd)
        try:
            xdbu, ydbu = load_data(dbpath,d,m='0.35u')
            pred_db_35u = model_db_35u.predict(xdbu)
            preds = np.concatenate((pred_dt,pred_db_35d,pred_db_35u))
            y = np.concatenate((ydt,ydbd,ydbu))
        except:
            print('没有db_35u的数据')
            preds = np.concatenate((pred_dt,pred_db_35d))
            y = np.concatenate((ydt,ydbd))

        mean,bias,mse,r2 = calc_score(preds,y)

        hist_or_plot = 'plot'
        plt.figure(figsize=(8,8))
        dmf.plot_scatter(y,preds,hist_or_plot,day_or_month,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse,r2=r2)
        figname = '{}_{}_{}'.format(d, 'all',hist_or_plot)+'.png'
        plt.savefig(figpath/figname,dpi=200)

        hist_or_plot = 'hist'
        plt.figure(figsize=(8,8))
        dmf.plot_scatter(y,preds,hist_or_plot,day_or_month,'AOD from Aqua/MODIS','AOD from FY3D/MERSI',mse=mse,r2=r2,mean=mean,bias=bias)
        figname = '{}_{}_{}'.format(d, 'all',hist_or_plot)+'.png'
        plt.savefig(figpath/figname,dpi=200)



if __name__ == "__main__":
    # 需要验证的数据时间列表
    validation = {
        'day': ['20190524'],
        'month': ['201905']
    }

    # validation = {
    #     'day': ['20181018','20181112','20181114','20181219','201809','201810','201811','201812'],
    #     'month': ['201807','201808','201809','201810','201811','201812']
    # }
    main()

