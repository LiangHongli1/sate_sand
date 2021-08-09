# -*- coding: utf-8 -*-
"""
@author: lianghongli
@time: 20190520
处理原始匹配的数据：去除无效值；选出MERSI和MODIS时间差在1小时内的数据；
                 剔除均值和中位数相差较大、方差较大的数据；
每个样本的特征量包括：19个可见光通道的表观反射率，6个红外通道的亮温，4种角度的余弦值，地表高程DEM，地表类型，云量，水汽含量，经纬度，时间（季节），AOD
注意：太阳天顶角、卫星天顶角、相对方位角的单位是“degree”，需要转换为弧度；散射角的单位是“rad”
"""
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import time
from sklearn.preprocessing import OneHotEncoder

def season(day):
    date = '2018-'+str(day)
    struct = time.strptime(date,'%Y-%j')
    ym = time.strftime("%Y%m%d",struct)
    if int(ym[4:6]) in [3,4,5]:
        s = 0
    elif int(ym[4:6]) in [6,7,8]:
        s = 1
    elif int(ym[4:6]) in [9,10,11]:
        s = 2
    else:
        s = 3

    return s

def land_type(x):
    '''
    对x中的地表分类重新划分
    :param x: 地表类型变量，向量
    :return: 重新划分的地表类型，向量
    '''
    for j in range(len(x)):
        if x[j] in range(1,6):
            x[j] = 0
        elif x[j] in [6,8,9,11,12,14]:
            x[j] = 1
        elif x[j] in [7,10,13,15,16]:
            x[j] = 2
        else:
            x[j] = 3

    return x

def main():
    dt_or_db = 'db'
    fpath = Path(r'G:\RS competition\training_dataset\mersi_modis_{}_patch6_201905.HDF'.format(dt_or_db))
    fsave = Path(r'G:\RS competition\training_dataset\final dataset\mersi_modis_{}_patch6_201905.HDF'.format(dt_or_db))
    with h5py.File(fpath,'r') as f:
        for d in range(120,160):
            try:
                X = pd.DataFrame(data=f[str(d)][:])
            except:
                print('没有第{:d}天的数据'.format(d))
                continue

            x = X[X!=-99.]
            # tmersi = x.iloc[:,94] #MERSI的时间
            # tm4 = x.iloc[:,96] #MYD04的时间
            print(x.shape)
            xvalid = x.loc[np.abs(x.iloc[:,92]-x.iloc[:,94])<1,:] #只保留时间差在1小时内的数据
            xvalid = xvalid.loc[np.abs(xvalid.iloc[:,90]-xvalid.iloc[:,95])<0.01,:] # 经度差
            xvalid = xvalid.loc[np.abs(xvalid.iloc[:,91]-xvalid.iloc[:,96])<0.01,:] # 纬度差
            xvalid = xvalid.loc[xvalid.iloc[:,-1]<2,:]
            if dt_or_db=='dt':
                xvalid = xvalid.loc[xvalid.iloc[:,6]<=0.25,:] # 2.13μm通道的表观反射率大于0.25时选择用深蓝算法，否则用暗目标算法
            elif dt_or_db=='db':
                xvalid = xvalid.loc[xvalid.iloc[:,6]>0.25,:]

            xvalid = xvalid.dropna(how='any')
            if xvalid.shape[0]==0:
                continue
            print(xvalid.shape)

            xvalid.drop(np.arange(92,97),axis=1,inplace=True)
            # xvalid.drop(np.arange(98,103),axis=1,inplace=True)
            xv = xvalid.values
            # 将角度计算为余弦值
            xv[:,25:28] = np.cos(xv[:,25:28]*np.pi/180)
            xv[:,28] = np.cos(xv[:,28])
            # 对land type重新做分类
            for k in range(xv.shape[0]):
                if xv[k,-2] in range(1,6):
                    xv[k,-2] = 0
                elif xv[k,-2] in [6,8,9,11,12,14]:
                    xv[k,-2] = 1
                elif xv[k,-2] in [7,10,13,15,16]:
                    xv[k,-2] = 2
                else:
                    xv[k,-2] = 3

            if d==182:
                with h5py.File(fsave,'w') as g:
                    x = g.create_dataset(str(d),data=xv,chunks=True,compression='gzip')
            else:
                with h5py.File(fsave,'a') as g:
                    x = g.create_dataset(str(d),data=xv,chunks=True,compression='gzip')
                    if d==365:
                        g.attrs['variables'] = 'reflectances(19),BT(6),angles(4),DEM,median of the former 30 variables,' \
                                                'std of the 30 variables, longitude, latitude(MERSI),landtype,AOD'



if __name__=="__main__":
    main()







