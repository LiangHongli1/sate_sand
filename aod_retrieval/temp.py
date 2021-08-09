'''
@ ! D:/Python366
@ -*- coding:utf-8 -*-
@ Time: 2019/7/25, 21:15
@ Author: LiangHongli
@ Mail: l.hong.li@foxmail.com
@ File: temp.py
@ Software: PyCharm
'''
import numpy as np
from scipy.io import netcdf_file as nc
import time


def read_ec(fname,day=None):
    '''读EC的气象数据，包括10m处风速、2m处温度等
    day：一年中的第几天
    :returns u10,v10,t2m,sp (124, 313, 497), lon_lats (313,497), time(124,)
    '''
    filval = -99.
    with nc(fname,'r') as f:
        lon = f.variables['longitude'][:]
        lat = f.variables['latitude'][:]
        tiempo = f.variables['time'][:]
        temp = f.variables['t'][:]
        # 取空间特定位置的温度
        point_lon = 100
        point_lat = 35
        lon_index = np.where(lon==point_lon)[0]
        lat_index = np.where(lat==point_lat)[0]
        print(temp.shape)
        point_t = temp[:,lat_index,lon_index]


        # struct = time.strptime('2018-'+str(day),'%Y-%j')
        # date = time.strftime('%Y%m%d',struct)
        # d = int(date[6:])
        # hour_index = (d-1)*4+1
    #     u10_scale = f.variables['u10'].scale_factor
    #     u10_offset = f.variables['u10'].add_offset
    #     u10 = f.variables['u10'][hour_index].astype(np.float32)*u10_scale+u10_offset # 10m处x轴（east）向风速，unit=m/s
    #     u10fil = -32727*u10_scale+u10_offset
    #     v10_scale = f.variables['v10'].scale_factor
    #     v10_offset = f.variables['v10'].add_offset
    #     v10 = f.variables['v10'][hour_index].astype(np.float32)*v10_scale+v10_offset
    #     v10fil = -32727*v10_scale+v10_offset
    #
    #     t2m_scale = f.variables['t2m'].scale_factor
    #     t2m_offset = f.variables['t2m'].add_offset
    #     t2m = f.variables['t2m'][hour_index].astype(np.float32)*t2m_scale+t2m_offset # 2m处温度
    #     t2mfil = -32727*t2m_scale+t2m_offset
    #
    #     tiem = f.variables['time'][:] # 自1900-01-01后的小时数
    #
    #     sp_scale = f.variables['sp'].scale_factor
    #     sp_offset = f.variables['sp'].add_offset
    #     sp = (f.variables['sp'][hour_index].astype(np.float32)*sp_scale+sp_offset)/100. # surface pressure, unit=hPa
    #     spfil = (-32727*sp_scale+sp_offset)/100.
    #
    #
    # # 超出范围的值赋为filval
    # u10[u10==u10fil] = filval
    # v10[v10==v10fil] = filval
    # t2m[t2m==t2mfil] = filval
    # sp[sp==spfil] = filval


    # u10 = u10[:,:,np.newaxis]
    # v10 = v10[:,:,np.newaxis]
    # t2m = t2m[:,:,np.newaxis]
    # sp = sp[:,:,np.newaxis]
    # ecdata = np.concatenate((u10,v10,t2m,sp),axis=2)
    # f.close()
    return point_t

def main():
    fpath = r'G:\work\2017-2018Tem_250.nc'
    t = read_ec(fpath)
    print(t)

if __name__ == "__main__":
    main()
