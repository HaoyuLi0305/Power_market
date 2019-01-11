# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 10:59:53 2018

@author: de3ll
"""
####处理原始数据集
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


Load_history=pd.read_csv('D://load Forecasting//Load_history.csv', sep=',',thousands=',')
temperature_history=pd.read_csv('D://load Forecasting//temperature_history.csv', sep=',',thousands=',')
weights=pd.read_csv('D://load Forecasting//weights.csv', sep=',',thousands=',')
test=pd.read_csv('D://load Forecasting//test.csv', sep=',',thousands=',')
temperature_history['date']=pd.to_datetime(temperature_history.loc[:,('year','month','day')])


Load_history.info()
##存在object 不可画图计算
Load_history = Load_history.convert_objects(convert_numeric=True)



Load_history.describe()
#缺失率

for i in Load_history.columns:
   na_rate=len(Load_history[i][pd.notnull(Load_history[i])])/len(Load_history[i])
   print('na% of '+str(i)+' is '+str(na_rate))
#日负荷电量
hour=['h%s' % x  for x in range(1,25) ]
zone=['zone%s' % x for x in range(1,21)]
Load_history['daily_load']=Load_history[hour].sum(axis=1)
Load_history.columns
#Daily_load=Load_history.dropna()[['zone_id', 'year', 'month', 'day','daily_load']]
Daily_load=Load_history[['zone_id', 'year', 'month', 'day','daily_load']]
Daily_load['date']=pd.to_datetime(Daily_load.loc[:,('year','month','day')])
Daily_load.set_index(["date"], inplace=True)
Daily_load=Daily_load.drop(['year','month','day'],axis=1)
Daily_load['zone_id'].astype("category")
table=pd.pivot_table(Daily_load,values='daily_load',columns='zone_id',index=Daily_load.index)
table.columns=zone 
cor=table.corr()
table.to_csv('J:\\load Forecasting\\Daily_load.csv', index = True)

##日均temperature
stat=['stat%s' % x for x in range(1,12)]
temperature_history['daily_temp']=temperature_history[hour].mean(axis=1)
#Daily_temp=temperature_history.dropna()[['station_id', 'year', 'month', 'day','daily_temp']]
Daily_temp=temperature_history[['station_id', 'year', 'month', 'day','daily_temp']]
Daily_temp['date']=pd.to_datetime(Daily_temp.loc[:,('year','month','day')])
Daily_temp.set_index(["date"], inplace=True)
Daily_temp=Daily_temp.drop(['year','month','day'],axis=1)
Daily_temp['station_id'].astype("category")
table2=pd.pivot_table(Daily_temp,values='daily_temp',columns='station_id',index=Daily_temp.index)
table2.columns=stat
table2.to_csv('J:\\load Forecasting\\Daily_temp.csv',encoding = 'utf-8', index = True)

data=pd.concat([table,table2],axis=1)
cor=data.corr('spearman')

#按区分拆数据集  
names = locals()
for j in range(1,21):
  #names['zone_load%s'% j]=Load_history[Load_history['zone_id']==j]
  ##去除空置
  names['zone_load%s'% j]=Load_history[Load_history['zone_id']==j]#.dropna(axis=0)
  #添加日期列
  names['zone_load%s'% j]['date']=pd.to_datetime(names['zone_load%s'% j].loc[:,('year','month','day')])
  names['zone_load%s'% j].set_index(["date"], inplace=True)
  names['zone_load%s'% j]=names['zone_load%s'% j].drop(['zone_id','year','month','day'],axis=1)
  #nesdir='J:\\load Forecasting\\zone_spilt\\zone_load%s.csv'% j
  #names['zone_load%s'% j].to_csv(nesdir,encoding = 'utf-8', index = True)
length=zone_load1.shape[0]

for k in range(1,21):    
    names['zone_load%s_series' % k]=np.asarray(names['zone_load%s'% k].drop(['daily_load'],axis=1)).reshape((length*24,1))
#拆分数据集时间
timelist=[]
for i in range(0,length):
    for j in range(0,24):
        timelist.append(zone_load2.index[i]+datetime.timedelta(hours=j))
#合并数据
#    'datetime64[ns]'    
coumns_name=['zone'+str(x) for x in range(1,21)]
Load_history_series=pd.DataFrame(columns=coumns_name,index=timelist)
for j in range(0,20):
    k=j+1
    Load_history_series.iloc[:,j]=names['zone_load%s_series' % k]
##
##把为na的日期分割出来

split_date=Load_history_series[pd.isnull(Load_history_series['zone1'])].index
#Load_history_series=Load_history_series.dropna(axis=0)
##保存数据
Load_history_series.to_csv('J:\load Forecasting\Load_history_series.csv', encoding = 'utf-8', index = True)

price_corr=Load_history_series.corr()

plt.subplots(figsize=(9, 9))
sns.heatmap(price_corr, vmax=1, square=True, cmap="Blues")
plt.title('correlation of zone')
plt.show()


##temperature
names = locals()
for j in range(1,12):
  #names['zone_load%s'% j]=Load_history[Load_history['zone_id']==j]
  ##去除空置
  names['stat_load%s'% j]=temperature_history[temperature_history['station_id']==j]#.dropna(axis=0)
  #添加日期列
  names['stat_load%s'% j]['date']=pd.to_datetime(names['stat_load%s'% j].loc[:,('year','month','day')])
  names['stat_load%s'% j].set_index(["date"], inplace=True)
  names['stat_load%s'% j]=names['stat_load%s'% j].drop(['station_id','year','month','day'],axis=1)
  #nesdir='J:\\load Forecasting\\stat_spilt\\stat_load%s.csv'% j
  #names['stat_load%s'% j].to_csv(nesdir,encoding = 'utf-8', index = True)
length=stat_load1.shape[0]

for k in range(1,12):    
    names['stat_load%s_series' % k]=np.asarray(names['stat_load%s'% k].drop(['daily_temp'],axis=1)).reshape((length*24,1))
#拆分数据集时间
timelist=[]
for i in range(0,length):
    for j in range(0,24):
        timelist.append(stat_load1.index[i]+datetime.timedelta(hours=j))

coumns_name=['stat'+str(x) for x in range(1,12)]
temperature_history_series=pd.DataFrame(columns=coumns_name,index=timelist)
for j in range(0,11):
    k=j+1
    temperature_history_series.iloc[:,j]=names['stat_load%s_series' % k]
#temperature_history_series_nona=temperature_history_series.dropna(axis=0)

temperature_history_series.to_csv('J:\\load Forecasting\\temperature_history_series.csv', encoding = 'utf-8', index = True)

load_history_summary=pd.concat([Load_history_series,temperature_history_series],axis=1)
load_history_summary['date']=load_history_summary.index
for i in range(load_history_summary.shape[0]+1):
    times=datetime.datetime.strptime(load_history_summary.index[i],"%Y-%m-%d %H:%M:%S")
    load_history_summary['year'][i]=times.year
    load_history_summary['month'][i]=times.month
    load_history_summary['day'][i]=times.day
    load_history_summary['hour'][i]=times.hour
    load_history_summary['weekday'][i]=times.weekday

load_history_summary.to_csv('D:\\load Forecasting\\load_history_summary.csv', encoding = 'utf-8', index = True)
