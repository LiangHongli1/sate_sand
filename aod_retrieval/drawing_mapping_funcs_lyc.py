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
from matplotlib.colors import LogNorm,Normalize
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np
from scipy import optimize
from mpl_toolkits.basemap import Basemap

#直线方程函数
def f_1(x, A, B):
    return A*x + B

def plot_scatter(x,y,hist_or_plot,day_or_month,xlabel,ylabel,mse=None,r2=None,mean=None,bias=None):
    font = {
        'family': 'Times New Roman',
        'size': 13,
        'weight': 'light'
    }
    mpl.rc('font', **font)
    A1, B1 = optimize.curve_fit(f_1, x, y)[0]
    plt.plot(x, f_1(x,A1,B1), 'r-', linewidth=1.2) #拟合线
    xx = 1.5*0.05
    yy = 1.5*0.93
    plt.text(xx,yy,'y = {:.3f}*x + {:.3f}'.format(A1,B1))

    if hist_or_plot=='hist':
        if day_or_month=='day':
            plt.hist2d(x,y,bins=200,cmap='jet',cmin=2,norm=Normalize())
        else:
            plt.hist2d(x,y,bins=200,cmap='jet',cmin=10,norm=Normalize())

        plt.colorbar(pad=0.01)
        plt.plot(x,x,'k-',linewidth=1.2)
        plt.legend(['regression line','1:1 line'],loc=(0.04,0.6))

    elif hist_or_plot=='plot':
        # 画上下的误差限
        y1 = x+0.05+0.15*x
        y2 = x-0.05-0.15*x
        # print('ground-truth:',x[:10])
        # print('y1,y2:',y1[:10],y2[:10])
        index1 = np.where(x==np.max(x))[0][0]
        index2 = np.where(x==np.min(x))[0][0]
        mask = np.logical_and(y>=y2,y<=y1)

        num_valid = len(x[mask])/len(x)*100 # 误差范围内的点
        num_err_up = len(x[y>y1])/len(x)*100 # 超出误差上界的点
        num_err_down = len(x[y<y2])/len(x)*100 # 超出误差下界的点

        plt.plot((x[index2],x[index1]),(y1[index2],y1[index1]),c='m',ls='--',linewidth=1.) # 误差上限 color='m',linestyle='--'
        plt.plot((x[index2],x[index1]),(y2[index2],y2[index1]),c='m',ls='--',linewidth=1.) # 误差下线
        plt.text(xx,1.5*0.88,'N = {}'.format(len(x)))
        plt.text(xx,1.5*0.83,'Num_valid = {:.2f}%'.format(num_valid))
        plt.text(xx,1.5*0.78,'Num_error + = {:.2f}%'.format(num_err_up))
        plt.text(xx,1.5*0.73,'Num_error - = {:.2f}%'.format(num_err_down))
        plt.legend(['regression line','Exp error +','Exp error -'],loc=(0.05,0.58))
        # 画竖直的误差线
        bins = np.arange(0,1.5,0.15)
        xbins = np.zeros((bins.shape[0],3))
        for k in range(bins.shape[0]-1):
            mask = np.logical_and(x>=bins[k],x<=bins[k+1])
            xbins[k,0] = np.mean(x[mask])
            xbins[k,1] = np.mean(y[mask])
            xbins[k,-1] = np.std(y[mask])

        # print(xbins)
        print(xbins[:,-1])
        plt.errorbar(xbins[:,0],xbins[:,1],yerr=xbins[:,-1],fmt='o')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0,1.5])
    plt.ylim([0,1.5])

    if mse!=None and hist_or_plot=='hist':
        yy = 1.5*0.88
        ss = 'RMSE = '+'{:.3f}'.format(np.sqrt(mse))  #均方根误差
        plt.text(xx,yy,ss)

    if r2!=None and hist_or_plot=='hist':
        yy = 1.5*0.83
        ss = 'R = '+'{:.3f}'.format(np.sqrt(r2))
        plt.text(xx,yy,ss)

    if mean!=None and hist_or_plot=='hist':
        yy = 1.5*0.78
        ss = 'mean = '+'{:.3f}'.format(mean)
        plt.text(xx,yy,ss)

    if bias!=None and hist_or_plot=='hist':
        yy = 1.5*0.73
        ss = 'bias = '+'{:.3f}'.format(bias)
        plt.text(xx,yy,ss)

    # plt.show()

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
    m.readshapefile(r'G:\RS competition\china\china',name='china',linewidth=1,color='w')
    # nm = Normalize(vmin=0.0,vmax=1.5,clip=True)
    # cs = m.pcolormesh(lons,lats,data,vmin=vmin,vmax=vmax,cmap='jet',latlon=True)
    cs = m.scatter(lons,lats,s=0.5,c=data,cmap='jet',marker=',',latlon=True)
    plt.title(title)
    if label!=None:
    #     m.colorbar(cs,location='right',size='3%',pad='3%')
    # else:
        m.colorbar(cs,location='right',size='3%',pad='3%',label=label)
    # plt.show()

