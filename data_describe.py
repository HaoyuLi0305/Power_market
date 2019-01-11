# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:45:14 2018

@author: mr_li
"""

import pandas as pd 
import os 
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time

names = locals()
os.getcwd()
#导入数据
filedir=['D:\load Forecasting\stat_spilt','D:\load Forecasting\zone_spilt']
for dirname in filedir:
    filelist=os.listdir(dirname)
    os.chdir(dirname)
    for filename in filelist:
        file=re.sub(".csv","",filename)
        names[file]=pd.read_csv(filename)


Load_history_series=pd.read_csv("D:\load Forecasting\Load_history_series.csv",index_col=0)
load_history_summary1=pd.read_csv("D:\load Forecasting\Load_history_summary1.csv",index_col=0)
temperature_history_series=pd.read_csv("D:\\load Forecasting\\temperature_history_series.csv",index_col=0)
Daily_load=pd.read_csv("D:\\load Forecasting\\Daily_load.csv")
Daily_temp=pd.read_csv("D:\\load Forecasting\\Daily_temp.csv")

data_hour=pd.concat([Load_history_series,temperature_history_series],axis=1)
#data_hour=data.dropna()
##调整日期
Daily_load.date=pd.to_datetime(Daily_load.date)
Daily_temp.date=pd.to_datetime(Daily_temp.date)


data_day=pd.concat([Daily_load,Daily_temp.drop(['date'],axis=1)],axis=1)
#data_day=pd.concat([Daily_load,Daily_temp],axis=1)

###############################画图
data_day=data_day[data_day.zone1!=0][0:1586]
colors = plt.cm.BuPu(np.linspace(1, 0.5, 20))
plt.style.use('ggplot')
plt.figure(figsize=(15,12))
zones_list=[]
for i in range(0,20):
    j=i+1
    zones_list.append('zone%s' % j)
    plt.subplot(5,4,1+i)
    plt.plot(data_day[zones_list[i]],color=colors[i])
    plt.title(zones_list[i],loc='right')
    plt.tight_layout()
plt.savefig("D:\\load Forecasting\\plot\\20zones_dailyload.jpg")

##############zone1分布、
plt.style.use('ggplot')
plt.figure(figsize=(16,9))
plt.hist(data_day['zone1'],bins=20,color='#B08FC7')
plt.title('Distribution of zone1 load',fontsize=18)
plt.savefig("D:\\load Forecasting\\plot\\zone1distrubution.jpg")





#############################################################load1
###具体分解load1图
data_day=data_day[data_day.zone1!=0][0:1586]
spilt_day='2004-12-31'
plt.style.use('ggplot')
fig,(ax2,ax1)=plt.subplots(1,2,gridspec_kw = {'width_ratios':[1, 3]},figsize=(16, 9))
#fig,ax1=plt.subplots(1,1)
ax1.set_ylabel('zoad1',fontsize=12)
ax2.set_ylabel('zone1',fontsize=12)
ax1.set_yticks([int('%d00000'%i) for i in range(12)])
ax2.set_yticks([int('%d00000'%i) for i in range(12)])
ax1.set_xticks([datetime.date(i,j,1) for i in range(2004,2009) for j in [1,7]])
ax1.set_title('Daily load of zone1')
ax2.set_xticks([datetime.date(i,j,1) for i in [2004,2005] for j in [1,12]])
#ax1.set_xticklabels('')
ax1.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2004,2009) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in [2004] for j in [1,12]])
#ax1.plot(data_day['date'].astype(datetime.datetime).values,data_day['zone1'],color='#B08FC7')
ax1.plot(data_day[data_day['date']<spilt_day]['date'].astype(datetime.datetime).values,data_day[data_day['date']<spilt_day]['zone1'],color='#B08FC7')
ax1.plot(data_day[data_day['date']>=spilt_day]['date'].astype(datetime.datetime).values,data_day[data_day['date']>=spilt_day]['zone1'],color='#8FBAC8')
         
ax2.plot(data_day[data_day['date']<spilt_day]['date'].astype(datetime.datetime).values,data_day[data_day['date']<spilt_day]['zone1'],color='#B08FC7')
#fig.tight_layout()
plt.show()
plt.savefig("D:\\load Forecasting\\plot\\cycload1.jpg")

####################################################barplot
###mode function众数
from scipy import stats
 
def mode(nums):
    from scipy import stats
    mode=stats.mode(nums)[0][0]
    return(mode)
#season    
season_list=['Fall','Spring','Summer','Winter']
weekday_list=['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
hour_list=range(0,24)   
    
################week and season
load_history_summary1['sum_load']=np.sum(load_history_summary1.iloc[:,0:20],1)
week_season=pd.pivot_table(load_history_summary1,index=["weekday"],values=['zone1'],
               columns=["season"],aggfunc=[np.sum])
index=np.arange(1,8)
plt.style.use('ggplot')
plt.figure(figsize=(16,9))
plt.suptitle('zone1 Season Vs Weekday',fontsize=18)
for i in range(0,4):
    plt.subplot(2,2,1+i)
    plt.bar(index,week_season.iloc[:,i],color='#20B2AA')
    plt.xticks(np.arange(len(weekday_list))+1,weekday_list,rotation=40)        
    plt.ylabel(season_list[i])
plt.savefig('D:\\load Forecasting\\plot\\zone1SeasonVsWeekday.jpg')

#weekday
week_load=pd.pivot_table(load_history_summary1,index=["weekday"],values=['zone1'],
               aggfunc=[np.sum])
plt.figure(figsize=(16,9))
plt.suptitle('zone1 Weekday',fontsize=18)
index=np.arange(1,8)
plt.style.use('ggplot')
plt.bar(index,week_load.iloc[:,0],color='#87CEFA')
plt.xticks(np.arange(len(weekday_list))+1,weekday_list,rotation=40)
plt.savefig('D:\\load Forecasting\\plot\\zone1Weekday.jpg')


#season
plt.figure(figsize=(16,9))
plt.suptitle('zone1 Season',fontsize=18)
season_load=pd.pivot_table(load_history_summary1,index=["season"],values=['zone1'],
               aggfunc=[np.sum])
index=np.arange(1,5)
plt.style.use('ggplot')
#width=1.5
plt.bar(index,season_load.iloc[:,0],label='season load',color='#B0C4DE')
plt.xticks(np.arange(len(season_list))+1,season_list,rotation=40)
plt.savefig('D:\\load Forecasting\\plot\\zone1season.jpg')

##hour
plt.figure(figsize=(16,9))
plt.suptitle('zone1 Hour',fontsize=18)
hour_load=pd.pivot_table(load_history_summary1,index=["hour"],values=["zone1"],aggfunc=[np.sum])
index=np.arange(1,25)
plt.style.use('ggplot')
plt.bar(index,hour_load.iloc[:,0],color='#87CEFA')
plt.xticks(np.arange(len(index))+1,index-1,rotation=40)   
plt.savefig('D:\\load Forecasting\\plot\\zone1Hour.jpg')
        

##hourvsseason
hour_season=pd.pivot_table(load_history_summary1,index=["hour"],values=['zone1'],
               columns=["season"],aggfunc=[np.sum])
index=np.arange(1,25)
plt.style.use('ggplot')
plt.figure(figsize=(16,9))
plt.suptitle('zone1 Hour Vs Season',fontsize=18)
for i in range(0,4):
    plt.subplot(2,2,1+i)
    plt.bar(index,hour_season.iloc[:,i],color='#20B2AA')
    plt.ylabel(season_list[i])
    plt.xticks(np.arange(len(index))+1,index-1,rotation=40) 
plt.savefig('D:\\load Forecasting\\plot\\zone1HourVsSeason.jpg')

    
###weekdayvshour
hour_season=pd.pivot_table(load_history_summary1,index=["hour"],values=['zone1'],
               columns=["weekday"],aggfunc=[np.sum])    
index=np.arange(1,25)
plt.figure(figsize=(16,9))
plt.suptitle('zone1 Hour Vs Weekday',fontsize=18)
plt.style.use('ggplot')
for i in range(0,7):
    plt.subplot(4,2,1+i)
    plt.bar(index,hour_season.iloc[:,i],color='#B0C4DE')
    plt.ylabel(weekday_list[i])
    plt.xticks(np.arange(len(index))+1,index-1,rotation=40)
plt.savefig('D:\\load Forecasting\\plot\\zone1HourVsWeekday.jpg')

##   
######################相关系数图
data=pd.concat([Load_history_series,temperature_history_series],axis=1).dropna()
cor_spear=data.corr("spearman")
cor_pearson=data.corr()
cor_kendall=data.corr('kendall')


plt.style.use('ggplot')
plt.subplots(figsize=(16, 9))
sns.heatmap(cor_spear, vmax=1, square=True, cmap="Blues")
plt.title(r'correlation of load & temperature')
plt.show()
plt.savefig("D:\\load Forecasting\\plot\\correlation.jpg")