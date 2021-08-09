#！/usr/bin/env.python3
# _*_ coding:utf-8 _*_

# @Author: lianghongli
# @Email: l.hong.li@foxmail.com
# @Time: 2019-5-27
from pathlib2 import Path
import joblib
import numpy as np
# import util
# import read_files as rf
# from sklearn.preprocessing import OneHotEncoder
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import h5py

def mapping(lons,lats,data,t):
    # plt.figure(figsize=(10,6))
    m = Basemap(llcrnrlat=35,urcrnrlat=50,urcrnrlon=101,llcrnrlon=75,lat_0=0.)
    font = {'family':'Times New Roman',
            'weight':1,
            'size':10}
    mpl.rc('font',**font)
    meridians = np.arange(75,136,10)
    parallels = np.arange(20,51,10)
    m.drawmeridians(meridians,labels=[0,0,0,1],color='w',linewidth=0.5)
    m.drawparallels(parallels,labels=[1,0,0,0],color='w',linewidth=0.5)
    m.drawmapboundary(color='w',fill_color='k')
    m.drawlsmask(ocean_color='k')
    m.readshapefile(r'G:\RS competition\china\china',name='china',linewidth=0.6,color='w')
    # nm = Normalize(vmin=0.0,vmax=1.5,clip=True)
    cs = m.pcolormesh(lons,lats,data,vmin=0.0,vmax=1.0,cmap='jet',latlon=True)
    plt.title(t)
    m.colorbar(cs,location='right',size='3%',pad='3%')
    # plt.show()

# def main():
#     path10 = Path(r'G:\RS competition\raw_dataset\FY3D_MERSI_L1\case data\20181018')
#     ec10 = Path(r'G:\RS competition\raw_dataset\EC\20181001-1031.nc')
#     files10 = [x for x in path10.glob('*.HDF') if '1000M' in x.name]
#     # path11 = Path(r'G:\RS competition\raw_dataset\FY3D_MERSI_L1\case data\20181112')
#     # ec11 = Path(r'G:\RS competition\raw_dataset\EC\20181101-1130.nc')
#     # files11 = [x for x in path11.glob('*.HDF') if '1000M' in x.name]
#
#     grids, grids_hash = util.draw_grid()
#
#     for f in files10:
#         t = f.name.split('_')[4]+'_'+ f.name.split('_')[5]
#         mersi = np.full((grids.shape[0],grids.shape[1],16),-99.)
#         ec = np.ones(shape=(grids.shape[0], grids.shape[1], 4))* -99.
#         u10, v10, t2m, sp, lon_latsec = rf.read_ec(ec10,18)
#         ec =util.adjust_ec(grids, grids_hash, lon_latsec, u10, v10, t2m, sp, ec)
#
#         model_db = joblib.load('GBT_DB.m')
#         # model_dt = joblib.load('GBT_DT.m')
#         xmersi,lon_lats = rf.read_MERSI(f)
#         mersi = util.adjust_mersi(grids, grids_hash, lon_lats, xmersi, mersi)
#
#         seasons = 3
#         # 建立季节、一天中的时刻，两个特征的张量(行数，列数，1)
#         seasons = np.full((grids.shape[0], grids.shape[1], 1), seasons)
#         print(mersi.shape,ec.shape)
#         data = np.concatenate((mersi,ec,seasons),axis=2)
#         data2d = data.reshape((-1,data.shape[-1]))
#
#         ydb_grid = np.full((grids.shape[0]*grids.shape[1], 1), -99.)
#         rs = []
#         for k in range(data2d.shape[0]):
#             if (data2d[k,:]!=-99.).all():
#                 rs.append(k)
#
#         if len(rs)==0:
#             print('第',t,'没有符合条件的数据')
#             continue
#
#         x = data2d[rs,:]
#         print(x.shape)
#         print(x[:2])
#         # enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
#         # season_onehot_feature = enc.fit_transform(x[:,-1].reshape((-1,1)))
#         season_onehot_feature = np.tile(np.array([0,1,0]),(x.shape[0],1))
#         x = x[:,:-1]
#         x = np.hstack((x,season_onehot_feature))
#         ydb = model_db.predict(x)
#         print(ydb.shape,ydb[:5])
#         print(len(ydb>0.))
#         # ydt = model_dt.predict(x)
#         ydb_grid[rs] = ydb.reshape((len(ydb),1))
#         ydb_grid = ydb_grid.reshape((grids.shape[0],grids.shape[1]))
#         lons = grids[:,:,0]
#         lats = grids[:,:,1]
#         # t = '20181018'
#         mapping(lons,lats,ydb_grid,t)

def main():
    west_path = r'G:\RS competition\training_dataset\x_mersi_west.HDF'
    east_path = r'G:\RS competition\training_dataset\x_mersi_east.HDF'
    model_db = joblib.load('GBT_DB.m')
    # model_dt = joblib.load('GBT_DT.m')
    with h5py.File(west_path,'r') as f:
        x = f['X'][:]
        sites_west = f['site'][:]
        x1018_west = x[1,:,:,:]
        # x1112_west = x[4,:,:,:]

    # with h5py.File(east_path,'r') as f:
    #     x = f['X'][:]
    #     sites_east = f['site'][:]
    #     x1018_east = x[1,:,:,:]

        # x1112_east = x[4,:,:,:]

    parts = [0,1000,2000,3390]
    t = 'MERSI 20181018 AOD_DB'
    for k in range(3):
        xpart_west = x1018_west[parts[k]:parts[k+1],:,:]
        sitepart_west = sites_west[parts[k]:parts[k+1],:,:]
        lons_west = sitepart_west[:,:,0]
        lats_west = sitepart_west[:,:,1]

        r,c,h = xpart_west.shape
        data2d = xpart_west.reshape((-1,h))
        ydb_grid = np.full((r*c, 1), -99.)
        rs = []
        for j in range(data2d.shape[0]):
            if (data2d[j,:]!=-99.).all():
                rs.append(j)

        if len(rs)==0:
            print('第',t,'没有符合条件的数据')
            continue
        else:
            x = data2d[rs,:]
        x[:,:4] = x[:,:4]*0.9967029224603491
        x[:,6:10] = x[:,6:10]*0.9967029224603491*0.9967029224603491
        print(x.shape)
        # print(x[:2])
        season_onehot_feature = np.tile(np.array([0,1,0]),(x.shape[0],1))
        x = x[:,:-1]
        x = np.hstack((x,season_onehot_feature))
        ydb = model_db.predict(x)
        # print(ydb.shape,ydb[:5])
        print(np.max(ydb))

        ydb_grid[rs] = ydb.reshape((len(ydb),1))
        ydb_grid = ydb_grid.reshape((r,c))
        ydb_grid[ydb_grid==-99.] = np.nan
        mapping(lons_west,lats_west,ydb_grid,t)

        # print(x1018_east.shape)
        # xpart_east = x1018_east[parts[k]:parts[k+1],:,:]
        # sitepart_east = sites_east[parts[k]:parts[k+1],:,:]
        # lons_east = sitepart_east[:,:,0]
        # lats_east = sitepart_east[:,:,1]
        #
        # r,c,h = xpart_east.shape
        # data2d = xpart_east.reshape((-1,h))
        # ydb_grid = np.full((r*c, 1), -99.)
        # rs = []
        # for k in range(data2d.shape[0]):
        #     if (data2d[k,:]!=-99.).all():
        #         rs.append(k)
        #
        # if len(rs)==0:
        #     print('第',t,'没有符合条件的数据')
        #     continue
        # else:
        #     x = data2d[rs,:]
        #
        # x[:,:4] = x[:,:4]*0.9967029224603491
        # x[:,6:10] = x[:,6:10]*0.9967029224603491*0.9967029224603491
        # print(x.shape)
        # # print(x[:2])
        # season_onehot_feature = np.tile(np.array([0,1,0]),(x.shape[0],1))
        # x = x[:,:-1]
        # x = np.hstack((x,season_onehot_feature))
        # ydb = model_db.predict(x)
        # # print(ydb.shape,ydb[:5])
        # print(np.max(ydb))
        #
        # ydb_grid[rs] = ydb.reshape((len(ydb),1))
        # ydb_grid = ydb_grid.reshape((r,c))
        # ydb_grid[ydb_grid==-99.] = np.nan
        # mapping(lons_east,lats_east,ydb_grid,t)

    figname = r'G:\RS competition\figures\results of methods\mersi_west\fig'+t+'.png'
    plt.savefig(figname,doi=100)
    # plt.show()


if __name__ == '__main__':
    main()
