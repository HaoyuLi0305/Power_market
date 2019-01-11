# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 15:23:10 2018

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
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search


load_data=pd.read_csv('D:\\load Forecasting\\Load_history_augmented.csv',index_col=0)
load_data.columns
load1=load_data[['year', 'month', 'day', 'holiday', 'weekend', 'business_day', 'season',
       'day_of_week', 'hour', 'load1', 'temp1', 'temp2', 'temp3', 'temp4', 'temp5', 'temp6', 'temp7',
       'temp8', 'temp9', 'temp10', 'temp11']]
load1=load1.dropna()
cate_list=['holiday', 'weekend', 'business_day', 'season','day_of_week','hour','month']
#cate_list=['holiday', 'weekend', 'business_day', 'season','day_of_week']
#定性变量转化虚拟变量
for i in cate_list:
    dummy=pd.get_dummies(load1[i],prefix=i)
    load1=pd.concat([load1,dummy],1)
load1=load1.drop(cate_list,axis=1)         
##############################################################分割数据集

split_num=int(38070*0.95)
load1_train=load1.iloc[0:split_num,:]
load1_test=load1.iloc[split_num:38070,:]
load1_test2=load1[37089-48:37089]
train_x, test_x, train_y, test_y = train_test_split(load1_train.drop('load1',1), load1_train['load1'],train_size=0.9, random_state=133)
#Dtrain=xgb.DMatrix(train_x,train_y)
#Dtest=xgb.DMatrix(test_x,test_y)

#######################################################cv调参
xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='reg:gamma',
 nthread=4,
 scale_pos_weight=1,
 seed=1024)
    
  


#####parameter 1max_depth 

xgb_param = xgb1.get_xgb_params()
cvresult = xgb.cv(xgb_param, Dtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
        metrics='rmse', early_stopping_rounds=50)
xgb1.set_params(n_estimators=cvresult.shape[0])






param_test1 = {
 'max_depth':[3,4,5,6,7],
 'min_child_weight':[3,4,5,6,7]
}
gsearch1 = GridSearchCV(estimator = XGBRegressor(
 learning_rate =0.1,
 n_estimators=1000,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='reg:gamma',
 nthread=4,
 scale_pos_weight=1,
 seed=27),  param_grid = param_test1,  scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch1.fit(train_x,train_y)

gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_

# gsearch1.best_params_
#Out[82]: {'max_depth': 7, 'min_child_weight': 4}

xgb1.set_params(max_depth= 7, min_child_weight=4)


##############parameter 2 gamma
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator =XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=7, min_child_weight=4, missing=None, n_estimators=1000,
       n_jobs=1, nthread=4, objective='reg:gamma', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1024,
       silent=True, subsample=0.8), param_grid = param_test3, scoring='neg_mean_squared_error',iid=False, cv=5)

gsearch3.fit(train_x,train_y)
gsearch3.grid_scores_, gsearch3.best_params_,     gsearch3.best_score_
##{'gamma': 0.0}

###subsample &colsample_bytree
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(estimator =XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=7, min_child_weight=4, missing=None, n_estimators=1000,
       n_jobs=1, nthread=4, objective='reg:gamma', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1024,
       silent=True, subsample=0.8), param_grid = param_test4, scoring='neg_mean_squared_error',iid=False, cv=5)

gsearch4.fit(train_x,train_y)
gsearch4.grid_scores_, gsearch4.best_params_,     gsearch4.best_score_
 #{'colsample_bytree': 0.9, 'subsample': 0.7}
####0.05步长
param_test5 = {
 'subsample':[i/100.0 for i in range(86,96,2)],
 'colsample_bytree':[i/100.0 for i in range(66,76,2)]
}

gsearch5 = GridSearchCV(estimator =XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=7, min_child_weight=4, missing=None, n_estimators=1000,
       n_jobs=1, nthread=4, objective='reg:gamma', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1024,
          
       silent=True, subsample=0.7), param_grid = param_test5, scoring='neg_mean_squared_error',iid=False, cv=5)

gsearch5.fit(train_x,train_y)
gsearch5.best_params_
# {'colsample_bytree': 0.7, 'subsample': 0.86}
xgb1.set_params(colsample_bytree=0.7, subsample= 0.86)


####正则化
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

gsearch6 = GridSearchCV(estimator =xgb1, param_grid = param_test6, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch6.fit(train_x,train_y)
gsearch6.best_params_
#{'reg_alpha': 0.1}
param_test7 = {
 'reg_alpha':[i/1000 for i in range(50,150,10)]
}
gsearch7 = GridSearchCV(estimator =xgb1, param_grid = param_test7, scoring='neg_mean_squared_error',iid=False, cv=5)
gsearch7.fit(train_x,train_y)
gsearch7.best_params_
#{'reg_alpha': 0.09}
xgb1.set_params(reg_alpha= 0.09)

#parameters of algorithm
xgb1=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.7, gamma=0, learning_rate=0.5, max_delta_step=0,
       max_depth=7, min_child_weight=4, missing=None, n_estimators=1000,
       n_jobs=1, nthread=4, objective='reg:gamma', random_state=0,
       reg_alpha=0.09, reg_lambda=1, scale_pos_weight=1, seed=1048,
       silent=True, subsample=0.86)
###################

 #Fit the algorithm on the data

xgb1.fit(train_x, train_y,eval_metric='rmse')

### train vs test
pre = xgb1.predict(test_x)
plt.figure(figsize=(16,9))
plt.style.use('ggplot')
plt.plot(pre,label='predict_load1')
plt.plot(np.array(test_y),label='test_load1')
plt.title('Test_prediction  MAE=%s' % str(np.sum(abs(pre-test_y))/len(pre)))
plt.legend(loc='upper left')
plt.savefig("D:\\load Forecasting\\plot\\TraiVsTest.jpg")


#####Newdata
pre1=xgb1.predict(load1_test.drop('load1',1))
plt.figure(figsize=(16,9))
plt.style.use('ggplot')
plt.plot(pre1,label='predict_load1')
plt.plot(np.array(load1_test['load1']),label='real_load1')
plt.title('Newdata_prediction  MAE=%s' % str(np.sum(abs(pre1-load1_test['load1']))/len(pre1)))
plt.legend(loc='upper left',fontsize=20)
plt.savefig("D:\\load Forecasting\\plot\\Newdata_predict.jpg")

####NEW DATe
pre2=xgb1.predict(load1_test2.drop('load1',1))
pre2=pd.DataFrame(pre2,index=load1_test2.index)

plt.figure(figsize=(25,12))
plt.style.use('ggplot')
plt.suptitle('Newday_predict',fontsize=18)
mult_list=[5,8,15,18,23,30]
for i in range(0,6):
    plt.subplot(2,3,1+i)
    daynum=24*mult_list[i]+2
    plt.plot(np.array(pre1[daynum:daynum+24]),label='predict_load1')
    plt.plot(np.array(load1_test.ix[daynum:daynum+24,'load1']),label='real_load1')
    plt.xticks(range(0,24))
    date=load1_test.index[daynum]
    
    plt.title('Date%s load MAE=%s' % (date.split()[0],str(np.sum(np.abs(pre1[daynum:daynum+24]-load1_test.ix[daynum:daynum+24,'load1']))/24)))
    plt.legend(loc='upper left')


plt.savefig("D:\\load Forecasting\\plot\\Newday_predict1.jpg")





#


#
plt.plot(pre2)
plt.plot(np.array(load1_test['load1']))

