# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 18:06:26 2018

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
cate_list=['holiday', 'weekend', 'business_day', 'season','day_of_week']

for i in cate_list:
    dummy=pd.get_dummies(load1[i],prefix=i)
    load1=pd.concat([load1,dummy],1)
load1=load1.drop(cate_list,axis=1)


load1=load1.iloc[0:39414,]

missing_day_list=load1[load1.isnull().values==True].index
load1_1=load1.iloc[0:10319,]


train_x, test_x, train_y, test_y = train_test_split(load1_1.drop('load1',1), load1_1['load1'],train_size=0.8, random_state=33)
Dtrain=xgb.DMatrix(train_x,train_y)
Dtest=xgb.DMatrix(test_x,test_y)

###调参cv
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
if useTrainCV:
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        metrics='rmse', early_stopping_rounds=early_stopping_rounds, show_progress=False)
    alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='rmes')
dtrain_predictions = alg.predict(dtrain[predictors])

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
 seed=27)







#Predict training set:
dtrain_predictions = alg.predict(dtrain[predictors])
dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=140, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27), 
 param_grid = param_test1,     scoring='rmes',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_


















param = {'silent':1, 'objective':'reg:gamma', 'booster':'gbtree', 'base_score':3}
watchlist = [(Dtest, 'eval'), (Dtrain, 'train')]
num_round = 50

# training and evaluation
bst = xgb.train(param,Dtrain, num_round, watchlist)
preds = bst.predict(Dtest)

#######
load1=load1.dropna()
