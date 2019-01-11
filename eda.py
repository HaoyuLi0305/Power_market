# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 08:47:44 2018

@author: de3ll
"""

import pandas as pd 
import os 
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time

load_history_summary=pd.read_csv("D:\load Forecasting\Load_history_summary.csv",index_col=0)

##处理loadsummary数据,特征化，输入数据
year=[]
month=[]
day=[]
hour=[]
weekday=[]
season=[]

for i in range(load_history_summary.shape[0]):
        times=datetime.datetime.strptime(load_history_summary.index[i],"%Y-%m-%d %H:%M:%S")
    year.append(times.year)
    month.append(times.month)
    day.append(times.day)
    weekday.append(times.weekday())
    hour.append(times.hour)
    if times.month*100+times.day in range(316,615):
        season.append('Spring')
    elif times.month*100+times.day in range(616,915):
        season.append('Summer') 
    elif times.month*100+times.day in range(916,1215):
        season.append('Fall')
    else:
        season.append('Winter')
    
load_history_summary['month']=month
load_history_summary['day']=day
load_history_summary['year']=year
load_history_summary['hour']=hour
load_history_summary['weekday']=weekday
load_history_summary['season']=season


load_history_summary['business_day']=1
for i in [5,6]:
    load_history_summary['business_day'][load_history_summary['weekday']==i]=0
    
load_history_summary.to_csv('D:\\load Forecasting\\load_history_summary1.csv', encoding = 'utf-8', index = True)

###






    