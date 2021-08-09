'''
@ ! D:/Python366
@ -*- coding:utf-8 -*-
@ Time: 2019/7/14, 22:01
@ Author: LiangHongli
@ Mail: l.hong.li@foxmail.com
@ File: drawing_mapping_funcs.py
@ Software: PyCharm
常用的画散点、折线和投影图的函数
'''
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib as mpl
import numpy as np
from mpl_toolkits.basemap import Basemap

def plot_scatter(x,y,xlabel,ylabel,mse=None,mae=None,r2=None):
    font = {
        'family': 'Times New Roman',
        'size': 10,
        'weight': 'light'
    }
    mpl.rc('font', **font)

    # plt.scatter(x,y,c=y,s=1.2) # 为显示数据密度，颜色标为y的色阶
    plt.hist2d(x,y,bins=200,norm=LogNorm())
    plt.plot(x,x,'k-',linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0,np.max(x)])
    plt.ylim([0,np.max(x)])
    if mse!=None:
        xx = np.max(x)*0.05
        yy = np.max(y)*0.9
        ss = 'MSE='+str(mse)[:5]
        plt.text(xx,yy,ss)

    if mae!=None:
        xx = np.max(x)*0.05
        yy = np.max(y)*0.8
        ss = 'MAE='+str(mae)[:5]
        plt.text(xx,yy,ss)

    if r2!=None:
        xx = np.max(x)*0.05
        yy = np.max(y)*0.7
        ss = '${R^2}$='+str(r2)[:5]
        plt.text(xx,yy,ss)

    plt.show()

def mapping(lons,lats,data,title,latb,latu,lonl,lonr,label=None,vmin=None,vmax=None):
    '''
    根据经纬度和（或）数据画投影图
    :param lons: 经度，向量或矩阵
    :param lats: 纬度，向量或矩阵
    :param data: 需要绘出的数据，向量或矩阵，和lons、lats保持相同shape
    :param title: 标题，string
    :param latb: 图形显示的底边纬度
    :param latu: 图形显示的顶边纬度
    :param lonl: 图形显示的左边经度
    :param lonr: 图形显示的左边经度
    :param label: colorbar的标签
    :param vmin: 数据显示的最小值
    :param vmax: 数据显示的最大值
    :return: projected figure
    '''
    m = Basemap(llcrnrlat=latb,urcrnrlat=latu,urcrnrlon=lonr,llcrnrlon=lonl,lat_0=0.) #使用默认投影方式
    font = {'family':'Times New Roman',
            'weight':1,
            'size':10}
    mpl.rc('font',**font)
    meridians = np.arange(75,136,10) # 显示的经度范围和经线
    parallels = np.arange(20,53,10) # 显示的纬度范围和纬线
    m.drawmeridians(meridians,labels=[0,0,0,1],color='w',linewidth=0.5)
    m.drawparallels(parallels,labels=[1,0,0,0],color='w',linewidth=0.5)
    m.drawmapboundary(color='w',fill_color='k')
    m.drawlsmask(ocean_color='k')
    m.readshapefile(r'H:\RS competition\china\china',name='china',linewidth=0.6,color='w')
    # nm = Normalize(vmin=0.0,vmax=1.5,clip=True)
    # cs = m.pcolormesh(lons,lats,data,vmin=vmin,vmax=vmax,cmap='jet',latlon=True)
    cs = m.scatter(lons,lats,s=1,c=data,cmap='jet',marker=',',latlon=True)
    plt.title(title)
    if label!=None:
    #     m.colorbar(cs,location='right',size='3%',pad='3%')
    # else:
        m.colorbar(cs,location='right',size='3%',pad='3%',label=label)
    # plt.show()
