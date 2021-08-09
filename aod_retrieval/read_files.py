# -*- coding: utf-8 -*-
"""
读取MYD02、MYD04、EC、PM2.5数据
@author: liang
"""

from pyhdf.SD import SD
import numpy as np
from scipy.io import netcdf_file as nc
import pandas as pd
import time
import h5py

def read_MYD03(fname):
    g = SD(str(fname))
    lons = g.select('Longitude')[:]
    lats = g.select('Latitude')[:]
    sol_zenith = g.select('SolarZenith')[:].astype(np.float32)
    sol_zenith = np.cos(sol_zenith*0.01*np.pi/180.)

    return sol_zenith,lons,lats

def read_MYD021(filename):
    '''读取MYD021中的数据，计算表观反射率，所有无效的值都填充为-99
    :returns ref(7, 2030, 1354), sol_zenith,sens_zenith,razimuth(406, 271), lon_lats(406,271,2)
    '''
    f = SD(str(filename))
    filval = -99.
    ref12 = f.select('EV_250_Aggr1km_RefSB')[:].astype(np.float32) # band1,band2,shape=[2,2030,1354]
    a = 30000.
    ref12[ref12[:]>a] = filval
    ref12[ref12[:]<0] = filval
    ref37 = f.select('EV_500_Aggr1km_RefSB')[:].astype(np.float32) # band3-7,shape=[5,2030,1354]
    ref37 = np.concatenate((ref37[:2], ref37[3:]), axis=0)
    ref37[ref37[:]>a] = filval
    ref37[ref37[:]<0] = filval

    ref12_scale = f.select('EV_250_Aggr1km_RefSB').reflectance_scales
    ref12_offset = f.select('EV_250_Aggr1km_RefSB').reflectance_offsets
    ref37_scale = f.select('EV_500_Aggr1km_RefSB').reflectance_scales
    ref37_offset = f.select('EV_500_Aggr1km_RefSB').reflectance_offsets
    ref37_scale = np.append(ref37_scale[:2], ref37_scale[3:])
    ref37_offset = np.append(ref12_offset[:2], ref37_offset[3:])

    sol_zenith_scale = f.select('SolarZenith').scale_factor
    sol_zenith = f.select('SolarZenith')[:].astype(np.float32)
    sol_zenith = sol_zenith_scale*sol_zenith # 卫星天顶角，unit=degrees,shape=[406,271]

    sens_zenith_scale = f.select('SensorZenith').scale_factor
    sens_zenith = f.select('SensorZenith')[:].astype(np.float32)
    sens_zenith = sens_zenith_scale*sens_zenith # 传感器天顶角，unit=degrees,shape=[406,271]

    azimuth_scale = f.select('SolarAzimuth').scale_factor
    razimuth = azimuth_scale*(f.select('SolarAzimuth')[:].astype(np.float32) - f.select('SensorAzimuth')[:].astype(np.float32)) # 相对方位角，unit=degrees,shape=[406,271]

    # 定标
    for k in range(2):
        ref12[k][ref12[k]!=filval] = ref12[k][ref12[k]!=filval]*ref12_scale[k]+ref12_offset[k]
    for k in range(4):
        ref37[k][ref37[k]!=filval] = ref37[k][ref37[k]!=filval]*ref37_scale[k]+ref37_offset[k]

    # 计算各角度的余弦，对reflectance做余弦订正
    sol_zenith = np.cos(sol_zenith*np.pi/180.)
    sens_zenith = np.cos(sens_zenith*np.pi/180.)
    razimuth = np.cos(razimuth*np.pi/180.)

    # ref12[ref12!=filval] = ref12[ref12!=filval]/sol_zenith
    # ref37[ref37!=filval] = ref37[ref37!=filval]/sol_zenith
    lons = f.select('Longitude')[:] # shape=[406,271]
    lats = f.select('Latitude')[:] # shape=[406,271]
    lons = lons[:,:,np.newaxis]
    lats = lats[:,:,np.newaxis]
    lon_lats = np.concatenate((lons, lats), axis=2)

    day = int(filename.name.split('.')[1][5:])
    hour = int(filename.name.split('.')[2][:2])
    if hour >= 14:
        hour = 1
    else:
        hour = 0
    time = day*10+hour
    #
    ref = np.concatenate((ref12, ref37), axis=0)

    return ref,sol_zenith,sens_zenith,razimuth,lon_lats

def read_MYD04(fname,dt_or_db):
    '''读取所需的AOD相关数据，所有无效的值都填充为-99
    :returns od_land (203, 135), lon_lats (203, 135, 2), time int
    '''
    f = SD(str(fname))
    filval = -99.
    # day = int(fname.name.split('.')[1][1:])
    hour = float(fname.name.split('.')[2])/100
    # if hour > 24:
    #     day += 1
    #     hour -= 24
    # day = np.array([day])
    hour = np.array([hour])
    if dt_or_db=='db':
        od_db = f.select("Deep_Blue_Aerosol_Optical_Depth_550_Land")[:].astype(np.float32) # shape=[203,135]
        od_db[od_db<=0] = filval
        od_db[od_db>5000] = filval
        od_db_scale = f.select("Deep_Blue_Aerosol_Optical_Depth_550_Land").scale_factor
        od_db[od_db!=filval] = od_db[od_db!=filval]*od_db_scale # AOT at 0.55 micron for land,retrieved with 'deep blue' algorithm

        cf = f.select('Deep_Blue_Cloud_Fraction_Land')[:].astype(np.float32)
        scale = f.select('Deep_Blue_Cloud_Fraction_Land').scale_factor
        cf = cf*scale
        cf[cf>1.00001] = filval
        cf[cf<0.] = filval

        od = od_db[:, :, np.newaxis]
    elif dt_or_db=='dt':
        od_dt = f.select("Corrected_Optical_Depth_Land")[:][1].astype(np.float32)
        od_dt[od_dt<=-500] = filval
        od_dt[od_dt>5000] = filval
        od_dt_scale = f.select("Corrected_Optical_Depth_Land").scale_factor
        od_dt[od_dt!=filval] = od_dt[od_dt!=filval]*od_dt_scale  # AOT at 0.55 micron for land,retrieved with 'dark target' algorithm
        # 记录云检测
        cf = f.select('Aerosol_Cloud_Fraction_Land')[:].astype(np.float32)
        scale = f.select('Aerosol_Cloud_Fraction_Land').scale_factor
        cf = cf*scale
        cf[cf>1.00001] = filval
        cf[cf<0.] = filval
        od = od_dt[:, :, np.newaxis]

    # sol_zenith = f.select('Solar_Zenith')[:].astype(np.float32)
    # sol_zenith_scale = f.select('Solar_Zenith').scale_factor
    # sol_zenith = np.cos(sol_zenith*sol_zenith_scale*np.pi/180.)
    # ss_zenith = f.select('Sensor_Zenith')[:].astype(np.float32)
    # ss_zenith_scale = f.select('Sensor_Zenith').scale_factor
    # ss_zenith = np.cos(ss_zenith*ss_zenith_scale*np.pi/180.)
    # sol_azimuth = f.select('Solar_Azimuth')[:].astype(np.float32)
    # sol_azimuth_scale = f.select('Solar_Azimuth').scale_factor
    # sol_azimuth = sol_azimuth*sol_azimuth_scale
    # ss_azimuth = f.select('Sensor_Azimuth')[:].astype(np.float32)
    # ss_azimuth_scale = f.select('Sensor_Azimuth').scale_factor
    # ss_azimuth = ss_azimuth*ss_azimuth_scale
    # rel_azimuth = np.cos((sol_azimuth-ss_azimuth)*np.pi/180.)
    # scatter = f.select('Scattering_Angle')[:].astype(np.float32)
    # scatter_scale = f.select('Scattering_Angle').scale_factor
    # scatter = np.cos(scatter*scatter_scale*np.pi/180.)

    # od = f.select('AOD_550_Dark_Target_Deep_Blue_Combined')[:].astype(np.float32)
    # od_scale = f.select('AOD_550_Dark_Target_Deep_Blue_Combined').scale_factor
    # od[od<=-100] = filval
    # od[od>5000] = filval
    # od[od!=filval] = od[od!=filval]*od_scale

    # ref = f.select("Mean_Reflectance_Land")[:].astype(np.float32)
    # ref[ref<0] = filval
    # ref[ref>10000] = filval
    # ref_scale = f.select("Mean_Reflectance_Land").scale_factor
    # ref[ref!=filval] = ref[ref!=filval]*ref_scale #平均表观反射率 at 0.47,0.55,0.66,0.86,1.24,1.64,2.13 micron
    # ref = np.concatenate((ref[:4], ref[5:7]), axis=0)
    # ref2 = ref**2
    lon = f.select("Longitude")
    lon = lon[:]
    lat = f.select("Latitude")
    lat = lat[:]

    lon = lon[:, :, np.newaxis]
    lat = lat[:, :, np.newaxis]
    lon_lats = np.concatenate((lon, lat), axis=2)

    # od_dt = od_dt[:, :, np.newaxis]
    cf = cf[:,:,np.newaxis]
    # day = day[:,np.newaxis,np.newaxis]
    hour = np.tile(hour,lon.shape[:2])
    hour = hour[:,:,np.newaxis]
    # print(od_db.shape,cf.shape,hour.shape)
    # od = od[:,:,np.newaxis]
    # sol_zenith = sol_zenith[:,:,np.newaxis]
    # ss_zenith = ss_zenith[:,:,np.newaxis]
    # rel_azimuth = rel_azimuth[:,:,np.newaxis]
    # scatter = scatter[:,:,np.newaxis]
    # ref   = np.transpose(ref, (1, 2, 0))
    # ref2 = np.transpose(ref2,(1,2,0))
    # print(od_dt.shape,ref.shape,sol_zenith.shape)
    # myd04 = np.concatenate((od_db, od_dt, ref, ref2,sol_zenith,ss_zenith,rel_azimuth,scatter), axis=2)
    myd04 = np.concatenate((od, cf, hour), axis=2)
    # day = int(filename.name.split('.')[1][5:])
    # hour = int(filename.name.split('.')[2][:2])+8
    # if hour >= 14:
    #     hour = 1
    # else:
    #     hour = 0
    # time = day*10+hour

    return myd04, lon_lats

def read_ec(fname,day):
    '''读EC的气象数据，包括10m处风速、2m处温度等
    day：一年中的第几天
    :returns u10,v10,t2m,sp (124, 313, 497), lon_lats (313,497), time(124,)
    '''
    filval = -99.
    with nc(fname,'r') as f:
        lon = f.variables['longitude'][:]
        lat = f.variables['latitude'][:]

        lon, lat = np.meshgrid(lon, lat)
        lon = lon[:, :, np.newaxis]
        lat = lat[:, :, np.newaxis]
        lon_lats = np.concatenate((lon, lat), axis=2)

        struct = time.strptime('2018-'+str(day),'%Y-%j')
        date = time.strftime('%Y%m%d',struct)
        d = int(date[6:])
        hour_index = (d-1)*4+1
        # if d==31:
        #     hour_index = 122
        u10_scale = f.variables['u10'].scale_factor
        u10_offset = f.variables['u10'].add_offset
        u10 = f.variables['u10'][hour_index].astype(np.float32)*u10_scale+u10_offset # 10m处x轴（east）向风速，unit=m/s
        u10fil = -32727*u10_scale+u10_offset
        v10_scale = f.variables['v10'].scale_factor
        v10_offset = f.variables['v10'].add_offset
        v10 = f.variables['v10'][hour_index].astype(np.float32)*v10_scale+v10_offset
        v10fil = -32727*v10_scale+v10_offset

        t2m_scale = f.variables['t2m'].scale_factor
        t2m_offset = f.variables['t2m'].add_offset
        t2m = f.variables['t2m'][hour_index].astype(np.float32)*t2m_scale+t2m_offset # 2m处温度
        t2mfil = -32727*t2m_scale+t2m_offset

        tiem = f.variables['time'][:] # 自1900-01-01后的小时数

        sp_scale = f.variables['sp'].scale_factor
        sp_offset = f.variables['sp'].add_offset
        sp = (f.variables['sp'][hour_index].astype(np.float32)*sp_scale+sp_offset)/100. # surface pressure, unit=hPa
        spfil = (-32727*sp_scale+sp_offset)/100.


    # 超出范围的值赋为filval
    u10[u10==u10fil] = filval
    v10[v10==v10fil] = filval
    t2m[t2m==t2mfil] = filval
    sp[sp==spfil] = filval


    # 时间转为一年中的第几天及该天的
    # nleap =len([x for x in range(1900,2018) if is_year(x)]) # 1900-2019年中闰年的数量
    # hours = tiem-(nleap+(2018-1900)*365)*24
    # day = hours//24
    # hour = hours%24
    # if hour >= 14:
    #     hour = 1
    # else:
    #     hour = 0
    # time = day*10 + hour
    u10 = u10[:,:,np.newaxis]
    v10 = v10[:,:,np.newaxis]
    t2m = t2m[:,:,np.newaxis]
    sp = sp[:,:,np.newaxis]
    ecdata = np.concatenate((u10,v10,t2m,sp),axis=2)
    return ecdata, lon_lats

def read_water(fname):
    with nc(fname,'r') as f:
        lons = f.variables['longitude'][:]
        lats = f.variables['latitude'][:]
        water = f.variables['p55.162'][:]
        scale = f.variables['p55.162'].scale_factor
        offset = f.variables['p55.162'].add_offset
        water = water*scale+offset

        lons,lats = np.meshgrid(lons,lats)
        lons = lons[:,:,np.newaxis]
        lats = lats[:,:,np.newaxis]
        lon_lats = np.concatenate((lons,lats),axis=2)
    return water,lon_lats

def is_year(year):
    """判断是否是闰年"""
    is1 = year%100!=0 and year%4==0
    is2 = year%400==0
    if is1 or is2:
        return True
    else:
        return False

def read_pm(fname):
    '''读PM2.5数据
    :returns pm25 (24, 1605) stations (1605,) time (24,)
    '''
    df = pd.read_csv(fname, engine='python')
    df = df[df['type']=='PM2.5']
    df = df.dropna(axis=1, thresh=4) # 在列的方向上大于4个空值，删除该列
    df = df.fillna(-99)
    stations = df.columns.values[3:]

    '''
    dates = fname.name.split('_')[2][:8]
    tstruct = time.strptime(dates,'%Y%m%d')
    day = int(time.strftime('%j',tstruct))
    hour = df['hour'].values
    time = day*10 + hour
    '''

    pm25 = df.iloc[:,3:].values

    return pm25, stations

def read_station(fname):
    '''站点的位置信息'''
    df = pd.read_excel(fname)
    df = df.dropna(axis=0)
    stations = df['监测点编码'].values
    lon_lats = df.loc[:, ['经度', '纬度']].values

    return stations, lon_lats

def ec_mapping(ecfiles):
    """
    :param ecfiles:
    :return: key: 路径数据的时间，格式是4位数，前三位是天数，第四位1代表白天，0代表晚上；value是列表存储文件路径
    """
    ec_map = {}
    for i in ecfiles:
        date_1 = i.name.split('-')[0]
        date_2 = '2018' + i.name.split('-')[1][:4]
        tstruct_1 = time.strptime(date_1, '%Y%m%d')
        tstruct_2 = time.strptime(date_2, '%Y%m%d')
        day_from = int(time.strftime('%j', tstruct_1))
        day_to = int(time.strftime('%j', tstruct_2))

        for day in range(day_from, day_to+1):
            if day in ec_map:
                ec_map[day].append(i)
            else:
                ec_map[day] = []
                ec_map[day].append(i)

    return ec_map

def pm_mapping(pmfiles):
    pm_map = {}
    for i in pmfiles:
        date = i.name.split('_')[2][:8]
        tstruct = time.strptime(date, '%Y%m%d')
        day = int(time.strftime('%j', tstruct))

        pm_map[day] = i
    return pm_map

def myd04_mapping(myd04files):
    myd04_map = {}
    for i in myd04files:
        day = int(i.name.split('.')[1][5:])
        hour = int(i.name.split('.')[2][:2])+8
        # if hour >= 24:
        #     day += 1

        if day in myd04_map:
            myd04_map[day].append(i)
        else:
            myd04_map[day] = []
            myd04_map[day].append(i)
    return myd04_map

def myd21_mapping(myd21files):
    myd21_map = {}
    for i in myd21files:
        day = int(i.name.split('.')[1][5:])
        hour = int(i.name.split('.')[2][:2])+8
        if hour >= 24 :
            day += 1

        if day in myd21_map:
            myd21_map[day].append(i)
        else:
            myd21_map[day] = []
            myd21_map[day].append(i)

    return myd21_map

def read_MERSI(fname,fgeo,coef):
    '''
    读取MERSI中和MODIS相同的个通道数据并定标，做云检测。分别和MODIS1-4、6-7通道对应的MERSI的通道为
    3，4，1，2，6，7
    :param fname: string or Path object
    :return: 6个通道做完定标、角度、日地距离订正、云检测等后的反射率，几个角度
    2019/7/2改：保存所有通道的信息，包括表观反射率、亮温，以备后用
    '''
    with h5py.File(fname,'r') as f:
        # 2019年更新了定标系数，用2018年的系数重新做数据尝试,只需修改cal_coef

        date = int(fname.name.split('_')[4])
        hour = float(fname.name.split('_')[5])/100
        # if hour >= 24:
        #     hour -= 24
        hour = np.array([hour])
        # cal_coef = f['Calibration/VIS_Cal_Coeff'][:]
        cal_coef = coef
        refs14 = f['Data/EV_250_Aggr.1KM_RefSB'][:].astype(np.float) #channel 1-4

        # refs67 = f['Data/EV_1KM_RefSB'][:].astype(np.float)
        # refs67 = refs67[1:3]
        refs519 = f['Data/EV_1KM_RefSB'][:].astype(np.float) #channel 5-19
        refs = np.append(refs14,refs519,axis=0)
        ratio = f.attrs['EarthSun Distance Ratio']
        # 计算通道20的亮温
        # ch24 = f['Data/EV_250_Aggr.1KM_Emissive'][:][0].astype(np.float)*0.01
        # c1 = 1.1910427* 10**-5
        # c2 = 1.4387752
        # vc = 10000./f.attrs['Effect_Center_WaveLength'][23]
        # te = c2*vc/(np.log(1+c1*vc**3/ch24))
        # a = f.attrs['TBB_Trans_Coefficient_A'][4]
        # b = f.attrs['TBB_Trans_Coefficient_B'][4]
        # tb = te*a/b

        # Computing bightness temperature of IR channel
        ch203 = f['Data/EV_1KM_Emissive'][:].astype(np.float)
        # slope = f['Data/EV_1KM_Emissive'].Slope
        slope = [2.0E-4,2.0E-4,0.01,0.01]
        for k in range(len(slope)):
            ch203[k] = ch203[k]*slope[k]

        ch245 = f['Data/EV_250_Aggr.1KM_Emissive'][:].astype(np.float)*0.01
        emiss = np.append(ch203,ch245,axis=0)
        c1 = 1.1910427* 10**-5
        c2 = 1.4387752
        vc = 10000./f.attrs['Effect_Center_WaveLength'][19:]
        a = f.attrs['TBB_Trans_Coefficient_A']
        b = f.attrs['TBB_Trans_Coefficient_B']
        tb = np.full(emiss.shape,0.)
        for k in range(len(vc)):
            emiss[k] = c2*vc[k]/(np.log(1+c1*vc[k]**3/emiss[k]))
            tb[k] = emiss[k]*a[k]+b[k]

    with h5py.File(fgeo,'r') as g:
        lats = g['Geolocation/Latitude'][:]
        lons = g['Geolocation/Longitude'][:]
        sol_zenith = g['Geolocation/SolarZenith'][:].astype(np.float) * 0.01
        ss_zenith = g['Geolocation/SensorZenith'][:].astype(np.float)*0.01
        sol_azimuth = g['Geolocation/SolarAzimuth'][:].astype(np.float)
        ss_azimuth = g['Geolocation/SensorAzimuth'][:].astype(np.float)
        rela_azimuth = (sol_azimuth-ss_azimuth)*0.01
        lsm = g['Geolocation/LandSeaMask'][:]
        # LandSeaMask:0=shallow ocean, 1=land, 3=shallow inland water,
        dem = g['Geolocation/DEM'][:]/1000. #地表高程，单位：km
        landcover = g['Geolocation/LandCover'][:] #地表覆盖类型
        '''
        地表覆盖类型：0=water，1=evergreen needleleaf forest,2=evergreen broadleaf forest,
        3=deciduous needleleaf forest,4=decidous broadleaf forest,5=mixed forests,6=closed shrublands,
        7=open shrublands,8=woody savannas,9=savannas,10=grasslands,11=permanent wetlands,
        12=croplands,13=urban and built-up,14=cropland/natural vegetation,15=snow and ice,
        16=barren or sparsely vegetated,17=IGBP water bodies,recorded to 0 for MODIS land product
        '''

    # 计算散射角sga
    radfactor = np.pi/180.
    add1 = -np.sin(sol_zenith*radfactor) * np.sin(ss_zenith*radfactor)*np.cos(rela_azimuth*radfactor)
    add2 = np.cos(sol_zenith*radfactor)*np.cos(ss_zenith*radfactor)
    sga = np.arccos(add1+add2)

    for k in range(refs.shape[0]):
        refs[k] = refs[k] * cal_coef[k,1] + cal_coef[k,0] # 初始定标
    # for k in range(2):
    #     refs67[k] = refs67[k] * cal_coef[k+5,1] + cal_coef[k+5,0]
    # sol_zenith = np.cos(sol_zenith*radfactor) # 后面需要求平均，因此先不求余弦值
    # ss_zenith = np.cos(ss_zenith*radfactor)
    # rela_azimuth = np.cos(rela_azimuth*radfactor)
    # 做日地距离和太阳天顶角订正
    refs = refs * ratio*ratio/np.cos(sol_zenith*radfactor)/100.
    # refs67 = refs67 * ratio*ratio/sol_zenith/100.
    fil = -99
    refs[refs>1.] = fil
    refs[refs<0.] = fil
    # refs67[refs67>1.] = fil
    # refs67[refs67<0.] = fil
    # 做简单的云检测，ref0.66>0.4，且通道24的亮温小于273，则被认为是云
    # r = []
    # c = []
    # for k in range(tb.shape[0]):
    #     for j in range(tb.shape[1]):
    #         if refs14[2,k,j]>0.4 and tb[k,j]<273:
    #             r.append(k)
    #             c.append(j)
    #
    # refs14[:,r,c] = fil
    # refs67[:,r,c] = fil
    # rs,cs = np.where(tb<273)
    # refs14[:,rs,cs] = fil
    # refs67[:,rs,cs] = fil
    # 去除陆地以外的pixel
    rs,cs = np.where(lsm!=1)
    refs[:,rs,cs] = fil
    # refs67[:,rs,cs] = fil
    # refs = np.concatenate((refs14,refs67),axis=0)
    # print(refs.shape)
    # refs2 = refs**2
    refs = np.transpose(refs,(1,2,0))
    tb = np.transpose(tb,(1,2,0))
    # print(refs.shape)
    # refs2 = np.transpose(refs2,(1,2,0))

    sol_zenith = sol_zenith[:,:,np.newaxis]
    ss_zenith = ss_zenith[:,:,np.newaxis]
    rela_azimuth = rela_azimuth[:,:,np.newaxis]
    sga = sga[:,:,np.newaxis]
    dem = dem[:,:,np.newaxis]
    landcover = landcover[:,:,np.newaxis]
    hour = np.tile(hour,sol_zenith.shape)
    print(sol_zenith.shape)
    # print(refs.shape,tb.shape,sol_zenith.shape,ss_zenith.shape,rela_azimuth.shape,sga.shape,dem.shape,landcover.shape,hour.shape)
    xmersi = np.concatenate((refs,tb,sol_zenith,ss_zenith,rela_azimuth,sga,dem,landcover,hour),axis=2) # shape=(nr,nc,32)
    lons = lons[:,:,np.newaxis]
    lats = lats[:,:,np.newaxis]
    lon_lats = np.concatenate((lons,lats),axis=2)

    return xmersi,lon_lats

def mersi_mapping(mersifiles):
    mersi_map = {}
    for i in mersifiles:
        date = i.name.split('_')[4]
        # hour = int(i.name.split('_')[5])+8
        tstruct = time.strptime(date, '%Y%m%d')
        day = int(time.strftime('%j', tstruct))
        # if hour >= 24:
        #     day += 1

        if day in mersi_map:
            mersi_map[day].append(i)
        else:
            mersi_map[day] = []
            mersi_map[day].append(i)
    return mersi_map


def thday(nday):
    if nday>=182 and nday<212:
        day = nday-182
    elif nday>=212 and nday<243:
        day = nday-212
    elif nday>=243 and nday<273:
        day = nday-243
    elif nday>=273 and nday<304:
        day = nday-273
    elif nday>=304 and nday<334:
        day = nday-304
    elif nday>=334 and nday<=365:
        day = nday-334

    return day
