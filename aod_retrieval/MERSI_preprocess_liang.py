# -*- coding: utf-8 -*-
"""
@author: lianghongli,zhuzeyang
@time: 20190520
初始匹配数据：首先按照经纬度最近邻的原则，将一天中的MERSI表观反射率、亮温等数据和MODIS的AOD匹配
匹配条件：
"""
import h5py
import numpy as np
from scipy.spatial import cKDTree
from scipy import stats
from pathlib import Path
import read_files as rf

def get_maps(myd04files,mersifiles,ecfiles=None,meteofiles=None):
    """
    建立文件与该文件所对应时间的映射关系，方便根据时间查找文件
    :return: pm，myd21，ec的映射，key是4位时间，前三位是天数第四位1代表白天0代表夜晚，value是path的列表
    """
    myd04_map = rf.myd04_mapping(myd04files)
    mersi_map = rf.mersi_mapping(mersifiles)
    if ecfiles!=None and meteofiles!=None:
        ec_map = rf.ec_mapping(ecfiles)
        meteo_map = rf.ec_mapping(meteofiles)
        return myd04_map, mersi_map, ec_map, meteo_map
    else:
        return myd04_map, mersi_map

def get_myd04(myd04_files):
    """
    循环读取一天内的所有myd04文件,返回具有实际值的数据点
    :param myd04_files:
    :return: (数据点数，5)；每列最后两个数据是lon和lat
    """

    for index, myd04_file in enumerate(myd04_files):
        myd04, lon_lats = rf.read_MYD04(myd04_file)
        myd04 = np.concatenate((myd04, lon_lats), axis=2)

        r,c  = np.where(myd04[:,:,0] != -99.)
        if index == 0:
            myd04_valid = myd04[r, c, :].reshape((-1, 5))
        else:
            myd04_valid = np.concatenate((myd04_valid, myd04[r, c, :].reshape(-1, 5)), axis=0)

    return myd04_valid

def main():
    m4path = Path(r'H:\RS competition\raw_dataset\MYD04_L2')
    mersipath = Path(r'I:\FY3D')

    ecpath = Path(r'H:\RS competition\raw_dataset\EC\water')
    meteopath = Path(r'H:\RS competition\raw_dataset\EC')
    fsave = Path(r'H:\RS competition\training_dataset\mersi_modis_db_patch6.HDF')
    m4files = [x for x in m4path.glob('*.HDF')]
    mersifiles = [x for x in mersipath.glob('*.HDF') if '1000M' in x.name]
    ecfiles = [x for x in ecpath.glob('*.nc')]
    meteofiles = [x for x in meteopath.glob('*.nc')]
    myd04_map, mersi_map, ec_map, meteo_map = get_maps(m4files,mersifiles,ecfiles,meteofiles)
    myd04_map, mersi_map = get_maps(m4files,mersifiles)
    # print(mersi_map.keys())
    m = 3 # patch大小
    n_valid = int((m*2)**2*0.90) # 一个patch中有效值大于75%时做计算

    for day,myd04_files in myd04_map.items():
        # if day>183:
        #     continue

        try:
            fmersi = mersi_map[day]
        except:
            print('MERSI没有第{:d}天的数据'.format(day))
            continue

        print(day)
        fec = ec_map[day][0]
        fmeteo = meteo_map[day][0]
        m4_valid = get_myd04(myd04_files)
        print('第{:d}天的有效AOD数据量为'.format(day),m4_valid.shape[0])

        # 先匹配水汽，读该天的EC水汽文件，由于一个月一个文件，因此一天只有一个文件
        water,eclon_lats = rf.read_water(fec)
        ectree = cKDTree(eclon_lats.reshape(-1,2))
        _,indexs = ectree.query(m4_valid[:,-2:])
        water = water.reshape(-1,1)
        water_valid = water[indexs]

        # 匹配气象数据，包括气压、温度、风速
        meteo, meteolon_lats = rf.read_ec(fmeteo,day)
        meteotree = cKDTree(meteolon_lats.reshape(-1,2))
        _indexs = meteotree.query(m4_valid[:,-2:])
        meteo = meteo.reshape(-1,meteo.shape[-1])
        meteo_valid = meteo[indexs,:]

        # 匹配MERSI数据
        fvalid = []
        fgeovalid = []
        for p,f in enumerate(fmersi):
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
                print('该文件覆盖范围不包括中国区域或覆盖范围很小')
                continue
            else:
                fvalid.append(f)
                fgeovalid.append(fgeo)

        print('第{:d}天可能包含有效数据的MERSI文件有{:d}个'.format(day,len(fvalid)))

        # 遍历包含有效数据的文件，每个文件里面都可能包含od_valid中的有效点，若有，计算最近邻，并在此次循环结束时除去已计算的od_valid点；若无，跳过
        for k,f in enumerate(fvalid):
            xmersi,lon_lats = rf.read_MERSI(f,fgeovalid[k])
            # 四个角的经纬度，即一幅图像的边界
            boudry_lons = np.array([lon_lats[0,0,0],lon_lats[-1,0,0],lon_lats[0,-1,0],lon_lats[-1,-1,0]])
            boudry_lats = np.array([lon_lats[0,0,1],lon_lats[-1,0,1],lon_lats[0,-1,1],lon_lats[-1,-1,1]])
            rvalid = []

            for x,lo in enumerate(m4_valid[:,3]):
                if lo>np.min(boudry_lons) and lo<np.max(boudry_lons) and m4_valid[x,4]>np.min(boudry_lats) and m4_valid[x,4]<np.max(boudry_lats):
                    rvalid.append(x)

            if len(rvalid)==0:
                print('文件',f,'中不包含有效的AOD点')
                continue
            else:
                m4_temp = m4_valid[rvalid,:]
                water_temp = water_valid[rvalid]
                meteo_temp = meteo_valid[rvalid,:]
                print('文件',f,'中包含的有效AOD点的数量为：',len(rvalid))
                tree = cKDTree(lon_lats.reshape((-1,2))) #建树
                _,indices = tree.query(m4_temp[:,-2:])

                # 删除已匹配的AOD点
                m4_valid = np.delete(m4_valid,rvalid,0)


            x0 = np.full((len(indices),xmersi.shape[-1]-2),-99.)# save mean value of each variable in xmersi
            x1 = np.full((len(indices),xmersi.shape[-1]-2),-99.) # save median of each variable in xmersi
            x2 = np.full((len(indices),xmersi.shape[-1]-2),-99.) # save std of each variable in xmersi
            lonlat_mean = np.full((len(indices),2),-99.)
            lc = np.full((len(indices),1),-99.) # land cover，算众数
            mer_hour = np.full((len(indices),1),xmersi[0,0,-1])

            for n,index in enumerate(indices):
                r = int(index//lon_lats.shape[1])
                c = int(index%lon_lats.shape[1])

                if r<m or c<m:
                    # print('数据位于MERSI文件的边界，不能取四周')
                    continue

                for a in range(xmersi.shape[-1]-2):
                    temp = xmersi[r-m:r+m,c-m:c+m,a]
                    temp = temp[temp!=-99.]
                    if len(temp)<n_valid:
                        continue
                    else:
                        x0[n,a] = np.mean(temp)
                        x1[n,a] = np.median(temp)
                        x2[n,a] = np.std(temp)

                lc[n,0] = stats.mode(xmersi[r-m:r+m,c-m:c+m,-2],axis=None)[0]
                lonlat_mean[n,0] = np.mean(lon_lats[r-m:r+m,c-m:c+m,0])
                lonlat_mean[n,1] = np.mean(lon_lats[r-m:r+m,c-m:c+m,1])

            if k==0:
                X = np.concatenate((x0,x1,x2,lonlat_mean,mer_hour,m4_temp[:,1:],lc,water_temp,meteo_temp,m4_temp[:,0].reshape(-1,1)),axis=1)
            else:
                X = np.append(X,np.concatenate((x0,x1,x2,lonlat_mean,mer_hour,m4_temp[:,1:],lc,water_temp,meteo_temp,m4_temp[:,0].reshape(-1,1)),axis=1),axis=0)


        print('第{:d}天的数据匹配完成'.format(day))
        if day==182:
            with h5py.File(fsave,'w') as f:
                x = f.create_dataset(str(day),data=X,dtype=float,chunks=True,compression='gzip')
        else:
            with h5py.File(fsave,'a') as f:
                x = f.create_dataset(str(day),data=X,dtype=float,chunks=True,compression='gzip')


if __name__=="__main__":
    main()
