'''
@ ! D:/Python366
@ -*- coding:utf-8 -*-
@ Time: 2019/7/16, 19:44
@ Author: LiangHongli
@ Mail: l.hong.li@foxmail.com
@ File: validation_mapping.py
@ Software: PyCharm
用训练好的模型预测MERSI(训练数据)，查看分布是否正确
'''
from pathlib import Path
from drawing_mapping_funcs_lyc import mapping
import joblib
import h5py
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy.sparse import hstack,coo_matrix
from scipy.spatial import cKDTree


def predict_data(xdt,xdb):
    '''
    将xdt、xdb重新组织为模型所需输入的样子
    :param xdt: 用于暗目标模型的数据，array
    :param xdb: 用于深蓝模型的数据，array
    :return: preds,GBT模型预测的AOD；longitude：AOD所对应的经度；latitude：AOD所对应发纬度
    '''
    dt_lons = xdt[:,-4]
    dt_lats = xdt[:,-3]
    coder = OneHotEncoder(handle_unknown='ignore',sparse=True) # 稀疏矩阵

    dt_data = hstack((xdt[:,:-2],coder.fit_transform(xdt[:,-2].reshape(-1,1))))
    if dt_data.shape[1]<96:
        dt_data = hstack((dt_data,np.zeros((xdt.shape[0],96-dt_data.shape[1]))))

    # dt_data = dt_data.todense()
    print('shape of dt_data:',dt_data.shape)

    # 按照2.13μm的表观反射率
    db35d = xdb[xdb[:,6]<=0.35,:]
    lons35d = xdb[xdb[:,6]<=0.35,90]
    lats35d = xdb[xdb[:,6]<=0.35,91]
    aod35d = xdb[xdb[:,6]<=0.35,-1]

    db35u = xdb[xdb[:,6]>0.35]
    lons35u = xdb[xdb[:,6]>0.35,90]
    lats35u = xdb[xdb[:,6]>0.35,91]
    aod35u = xdb[xdb[:,6]>0.35,-1]

    db35d = hstack((db35d[:,:-2],coder.fit_transform(db35d[:,-2].reshape(-1,1))))
    if db35d.shape[1]<95:
        db35d = hstack((db35d,np.zeros((db35d.shape[0],95-db35d.shape[1]))))
    elif db35d.shape[1]>95:
        temp = db35d.todense()
        temp = temp[:,:-1]
        db35d = coo_matrix(temp)

    # 预测
    dt_preds = model_dt.predict(dt_data)
    db_preds35d = model_db_35d.predict(db35d)

    if db35u.shape[0]!=0:
        db35u = hstack((db35u[:,:-2],coder.fit_transform(db35u[:,-2].reshape(-1,1))))
        if db35u.shape[1]<95:
            db35u = hstack((db35u,np.zeros((db35u.shape[0],95-db35u.shape[1]))))
        elif db35u.shape[1]>95:
            temp = db35u.todense()
            temp = temp[:,:-1]
            db35u = coo_matrix(temp)

        db_preds35u = model_db_35u.predict(db35u)
        preds = np.concatenate((dt_preds,db_preds35d,db_preds35u))
        groundtruth = np.concatenate((xdt[:,-1],aod35d,aod35u))
        longitude = np.concatenate((dt_lons,lons35d,lons35u))
        latitude = np.concatenate((dt_lats,lats35d,lats35u))
    else:
        preds = np.concatenate((dt_preds,db_preds35d))
        groundtruth = np.concatenate((xdt[:,-1],aod35d))
        longitude = np.concatenate((dt_lons,lons35d))
        latitude = np.concatenate((dt_lats,lats35d))

    return preds, groundtruth, longitude, latitude

def make_grid(region_lon=(73,135),region_lat=(17,55),resolution=0.1):
    '''
    根据所给经纬度构建网格，用于在做AOD月平均时搜索最邻近点
    :param region_lon: 经度范围
    :param region_lat: 纬度范围
    :param resolution: 空间分辨率，度
    :return: 网格和网格上的初始点
    '''
    xx = np.arange(region_lon[0],region_lon[1],resolution)
    yy = np.arange(region_lat[0],region_lat[1],resolution)
    xgrid,ygrid = np.meshgrid(xx,yy)
    xgrid = xgrid[:,:,np.newaxis]
    ygrid = ygrid[:,:,np.newaxis]
    xy_grid = np.concatenate((xgrid,ygrid),axis=2)
    xy = xy_grid.reshape(-1,2)
    initial = np.full((xy.shape[0],5), fill_value=0, dtype=float)

    return xy, initial

def val_day(fdt,fdb,date):
    # 做单独的一天的个例分析
    struct = time.strptime(date,'%Y%m%d')
    dia = time.strftime('%Y%j',struct)[-3:]
    with h5py.File(fdt,'r') as f:
        xdt = f[dia][:]
    with h5py.File(fdb,'r') as f:
        xdb = f[dia][:]

    preds, gt, longitude, latitude = predict_data(xdt,xdb)
    return np.concatenate((preds.reshape(-1,1),longitude.reshape(-1,1),latitude.reshape(-1,1),gt.reshape(-1,1)),axis=1)

def val_month(fdt,fdb,date):
    '''
    做预测的AOD的月平均
    :param fdt: filename or path，存用于暗目标模型训练数据的文件
    :param fdb: 存用于深蓝模型训练数据的文件
    :param month: 月份
    :return: GBT预测的AOD的月平均分布图
    '''
    xy, initial = make_grid()
    struct = time.strptime(date+'01','%Y%m%d')
    dia = time.strftime('%Y%j',struct)[-3:]
    for k in range(int(dia),int(dia)+32):
        with h5py.File(fdt,'r') as f:
            try:
                xdt = f[str(k)][:]
            except:
                print('没有第{}天的数据'.format(k))
                continue
        with h5py.File(fdb,'r') as f:
            try:
                xdb = f[str(k)][:]
            except:
                print('没有第{}天的数据'.format(k))
                continue

        preds, gt, lons, lats = predict_data(xdt,xdb)
        lonlat = np.concatenate((lons.reshape(-1,1),lats.reshape(-1,1)),axis=1)
        tree = cKDTree(lonlat)
        neighbors = tree.query_ball_point(xy,r=0.05,p=2)

        for j in range(len(neighbors)):
            if len(neighbors[j])==0:
                continue
            else:
                initial[j,0] += np.sum(preds[neighbors[j]]) # 预测AOD的最近邻点的和，用于最后平均
                initial[j,1] += np.sum(lons[neighbors[j]])
                initial[j,2] += np.sum(lats[neighbors[j]])
                initial[j,3] += np.sum(gt[neighbors[j]])
                initial[j,-1] += len(neighbors[j])

        print('第{}天计算完成'.format(k))
        # print(len(initial[:,0]!=0))
        # print(initial[np.where(initial!=0)])
    rs = np.where(initial[:,0]!=0)
    initial[rs,0] /= initial[rs,-1]
    initial[rs,1] /= initial[rs,-1]
    initial[rs,2] /= initial[rs,-1]
    initial[rs,3] /= initial[rs,-1]
    # print(initial[rs,0],initial[rs,1],initial[rs,2])
    mask = np.where(initial[:,0]!=0) # 保存有效值
    initial = initial[mask]

    return initial[:,:-1]

def main():
    # dt_or_db = 'dt'
    fdt = Path(r'G:\RS competition\training_dataset\final dataset\mersi_modis_{}_patch6_201905_recalibration.HDF'.format('dt'))
    fdb = Path(r'G:\RS competition\training_dataset\final dataset\mersi_modis_{}_patch6_201905_recalibration.HDF'.format('db'))
    figpath = Path(r'G:\RS competition\figures\results of methods\juesai\new colorbar')
    day_or_month = 'month'
    for date in validation[day_or_month]:
        if day_or_month=='day':
            aod_preds = val_day(fdt,fdb,date)
        else:
            aod_preds = val_month(fdt,fdb,date)
    # with h5py.File(r'H:\RS competition\training_dataset\final dataset\preds_2018{}_mean111.HDF'.format(mes),'w') as f:
    #     x = f.create_dataset('preds_aod',dtype=float,data=aod_preds,compression='gzip')

    # with h5py.File(r'F:\RS competition\training_dataset\final dataset\preds_201811_mean.HDF','r') as f:
    #     aod_preds = f['preds_aod'][:]

    # 对预测值做截断，便于画图的比较
    # aod_preds[aod_preds[:,0]>1.5,0] = 1.5

        aod_preds[aod_preds[:,0]>1.5,0] = 1.5
        aod_preds[aod_preds[:,3]>1.5,3] = 1.5
        aod_preds[aod_preds[:,0]>1.,0] = 1.
        aod_preds[aod_preds[:,3]>1.,3] = 1.
        aod_preds[aod_preds[:,0]<0,0] = np.nan
        aod_preds[aod_preds[:,3]<0,3] = np.nan
        print(aod_preds[:,1],aod_preds[:,2],aod_preds[:,0])
        plt.figure()
        title = date+' FY3D/MERSI'
        mapping(aod_preds[:,1],aod_preds[:,2],aod_preds[:,0],title,latb,latu,lonl,lonr,label='AOD')
        figname = 'MERSI_'+date
        plt.savefig(figpath/figname,dpi=300)
        # plt.show()
        plt.figure()
        title = date+' Aqua/MODIS'
        mapping(aod_preds[:,1],aod_preds[:,2],aod_preds[:,3],title,latb,latu,lonl,lonr,label='AOD')
        figname = 'MODIS_'+date
        plt.savefig(figpath/figname,dpi=300)
        # plt.show()


if __name__ == "__main__":
    latb = 17
    latu = 55
    lonl = 73
    lonr = 135
    model_dt = joblib.load(r'G:\RS competition\models\dt_numdata_50868_learningrate_0.10000_maxdepth_5_trainMSE_0.00646_trainMAE_0.05563_trainR2_0.92866_valMSE_0.01174_valMAE_0.07023_valR2_0.87317_patch6.m')
    model_db_35d = joblib.load(r'G:\RS competition\models\db_numdata_0.35_down_45374_learningrate_0.10000_maxdepth_3_trainMSE_0.01879_trainMAE_0.08572_trainR2_0.66200_valMSE_0.02313_valMAE_0.09235_valR2_0.58924_patch6.m')
    model_db_35u = joblib.load(r'G:\RS competition\models\db_numdata_0.35_up_29548_learningrate_0.10000_maxdepth_3_trainMSE_0.01813_trainMAE_0.08510_trainR2_0.73004_valMSE_0.02142_valMAE_0.09225_valR2_0.66373_patch6.m')

    validation = {
        'day': ['20190520','20190524','20190517'],
        'month': ['201905']
    }
    # validation = {
    #     'day': ['20181018','20181112','20181114','20181219'],
    #     'month': ['201807','201808','201809','201810','201811','201812']
    # }
    main()
