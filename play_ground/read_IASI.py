# -*- coding: utf-8 -*-
"""
@ Time: 201909
@ author: LiangHongli
@ Mail: i.hong.li@foxmail.com
"""
from datetime import datetime
import os
import sys

import harp
import coda
# from PB import pb_sat
# from PB.pb_time import metop_ymd2seconds
import numpy as np

# IASI的Level1数据包含的变量

class CLASS_IASI_L1():

    def __init__(self, BandLst):

        # 字典类型物理量
        self.Tbb = {}
        self.Rad = {}

        # 二维矩阵
        self.Lons = []
        self.Lats = []
        self.Time = []

        self.satAzimuth = []
        self.satZenith = []
        self.sunAzimuth = []
        self.sunZenith = []

        # 光谱信息
        self.wavenumber = []
        self.radiance = []

    def Load(self, L1File):

        print(u'读取 LEO所有数据信息......')
        if not os.path.isfile(L1File):
            print('Error: %s not found'  % L1File)
            sys.exit(1)

        try:
            # products = harp.import_product(L1File)
            # print(products)
            cursor = coda.Cursor()
            coda.cursor_set_product(cursor,L1File)
            fp = coda.open(L1File)
        except IOError:
            print('Error: open file error')
            return

        try:
            # EPS = EUMETSAT Polar System atmospheric products (GOME-2 and IASI)
            # EPS = EUMETSAT极地大气系统产品（GOME-2和IASI）'
            # 获取文件头信息
            product_class = coda.get_product_class(fp)
            product_type = coda.get_product_type(fp)
            product_version = coda.get_product_version(fp)
            product_format = coda.get_product_format(fp)
            # product_size = coda.get_product_file_size(fp)
            product_size = coda.get_size(fp)
            print('product_class: ', product_class)
            print('product_type: ', product_type)
            print('product_version: ', product_version)
            print('product_format: ', product_format)
            print('product_size: ', product_size)
            print('available product fields: ',coda.get_field_available(fp))
            print('filed names: ',coda.get_field_names(fp))
            products = coda.fetch(fp)
            print('coda products: ', products)
            record = coda.fetch(fp,"MDR")
            SAT_angle = coda.fetch(fp, 'MDR', -1, 'MDR', 'GGeoSondAnglesMETOP')
            SUN_angle = coda.fetch(fp, 'MDR', -1, 'MDR', 'GGeoSondAnglesSUN')

            all_sun_angle = []
            all_sat_angle = []

            for i in range(len(SAT_angle)):
                tmp_sat = SAT_angle[i].reshape(-1)
                tmp_sun = SUN_angle[i].reshape(-1)
                if len(all_sat_angle) == 0:
                    all_sat_angle = tmp_sat
                    all_sun_angle = tmp_sun
                else:
                    all_sat_angle = np.concatenate((all_sat_angle, tmp_sat), 0)
                    all_sun_angle = np.concatenate((all_sun_angle, tmp_sun), 0)

            iasiLen = len(record.longitude)
            self.satZenith = (all_sat_angle[0::2]).reshape(iasiLen, 1)
            self.satAzimuth = (all_sat_angle[1::2]).reshape(iasiLen, 1)
            self.sunZenith = (all_sun_angle[0::2]).reshape(iasiLen, 1)
            self.sunAzimuth = (all_sun_angle[1::2]).reshape(iasiLen, 1)

            self.Lons = (record.longitude).reshape(iasiLen, 1)
            self.Lats = (record.latitude).reshape(iasiLen, 1)

            self.radiance = record.spectral_radiance * 10 ** 7

            # 暂时取一个观测的光谱波数
            self.wavenumber = record.wavenumber[0, :]

            v_ymd2seconds = np.vectorize(metop_ymd2seconds)
            T1 = v_ymd2seconds(record.time)
            self.Time = T1.reshape(iasiLen, 1)

        except Exception as e:
            print(str(e))
            sys.exit(1)
        finally:
            coda.close(fp)

    def get_rad_tbb(self, D1, bandLst):
        '''
        D1是目标类的实例
        '''
        # iasi 的光谱波数范围
        WaveNum2 = self.wavenumber
        for Band in bandLst:
            WaveNum1 = D1.waveNum[Band]
            WaveRad1 = D1.waveRad[Band]
            # WaveRad2 = pb_sat.spec_interp(WaveNum1, WaveRad1, WaveNum2) #插值
            # newRad = pb_sat.spec_convolution(WaveNum2, WaveRad2, self.radiance) #光谱卷积
            # tbb = pb_sat.planck_r2t(
            #     newRad, D1.WN[Band], D1.TeA[Band], D1.TeB[Band]) #普朗克逆函数，求亮温

            # self.Tbb[Band] = tbb.reshape(tbb.size, 1)
            # self.Rad[Band] = newRad.reshape(newRad.size, 1)

def main():
    T1 = datetime.now()
    BandLst = ['CH_20', 'CH_21', 'CH_22', 'CH_23', 'CH_24', 'CH_25']
    L1File = r'G:\IASI_L1\IASI_xxx_1C_M01_20181215235053Z_20181215235357Z_N_O_20181216001746Z__20181216002024'
    iasi1 = CLASS_IASI_L1(BandLst)
    iasi1.Load(L1File)



if __name__ == '__main__':
    main()
