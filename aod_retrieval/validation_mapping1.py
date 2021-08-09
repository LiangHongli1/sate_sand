'''
@ ! D:/Python366
@ -*- coding:utf-8 -*-
@ Time: 2019/7/19, 10:50
@ Author: LiangHongli
@ Mail: l.hong.li@foxmail.com
@ File: validation_mapping1.py
@ Software: PyCharm
读MERSI L1数据，自行做云检测等处理，用训练好的模型预测，并和原始的MODIS分布做比较
'''
from pathlib import Path
import numpy as np
import read_files as rf
import drawing_mapping_funcs as dmf
import h5py
from scipy import stats

def main():
    fpath = Path(r'I:\FY3D')
    files11 = [x for x in fpath.glob('*.HDF') if '20181126' in x.name and '1000M' in x.name]
    n_valid = int(6**2*0.9)
    xdt = np.zeros((1,93),dtype=float)
    xdb = np.zeros((1,93),dtype=float)
    for k in range(len(files11)):
        f = files11[k]
        #加经纬度判断，先判断是否有在该范围的lonlat_valid,否则查找很费时间
        label = f.name.split('_')
        label[6] = 'GEO1K'
        geoname = '_'.join(label)
        fgeo = f.parent/geoname
        with h5py.File(fgeo, 'r') as g:
            lats = g['Geolocation/Latitude'][:]
            lat = np.mean(lats)
            lons = g['Geolocation/Longitude'][:]
            lon = np.mean(lons)
        if lat > 50 or lat < 20 or lon > 125 or lon < 70:
            print('文件{}覆盖范围不包括中国区域或覆盖范围很小'.format(f.name))
            continue
        else:
            xmersi,lon_lats = rf.read_MERSI(f,fgeo)
            xmersi = xmersi[:,:,:-1] # 去掉时间

        for ii in range(xmersi.shape[0]):
            # is3 = xmersi[] #去除冰雪像元
            for jj in range(xmersi.shape[1]):
                is1 = any(xmersi[ii,jj,:]==-99) # 去除无效像元
                #对于MODIS的算法，是取3*3窗口，0.51μm的表观反射率标准差大于0.006时，判定为有云，所以这里的云判别标准可以调整
                is2 = np.std(xmersi[ii-3:ii+3,jj-3:jj+3,1])>0.006 or any(xmersi[ii-3:ii+3,jj-3:jj+3,0])>0.25 #去除云像元
                if is1 or is2:
                    continue
                else:
                    temp = np.zeros((1,93),dtype=float)
                    for k in range(xmersi.shape[-1]-1):
                        xx = xmersi[ii-3:ii+3,jj-3:jj+3,k]
                        xx = xx[xx!=-99]
                        if len(xx)<n_valid:
                            continue
                        else:
                            temp[0,k] = np.mean()
                            temp[0,k+30] = np.median(xmersi[ii-3:ii+3,jj-3:jj+3,k])
                            temp[0,k+60] = np.std(xmersi[ii-3:ii+3,jj-3:jj+3,k])
                            temp[0,-3] = np.mean(lon_lats[ii-3:ii+3,jj-3:jj+3,0]) # longitude
                            temp[0,-2] = np.mean(lon_lats[ii-3:ii+3,jj-3:jj+3,1]) # latitude
                            temp[0,-1] = stats.mode(xmersi[ii-3:ii+3,jj-3:jj+3,-1]) # land type

                    # 判断用深蓝还是暗目标算法
                    if temp[0,6]>0.25:
                        xdb = np.append(xdb,temp,axis=0)
                    elif temp[0,6]<=0.25 and all(temp!=0):
                        xdt = np.append(xdt,temp,axis=0)


        print('MERSI文件{}完成'.format(f.name))
    xdt = xdt[1:]
    xdb = xdb[1:]
    with h5py.File(r'H:\RS competition\training_dataset\for_validation.HDF') as g:
        g['x_dt'] = xdt
        g['x_db'] = xdb

if __name__ =="__main__":
    main()
