# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:46:40 2018

@author: mr_li
"""

import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

boston=datasets.load_boston()
x=boston.data
y=boston.target

train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x, y,
                                                                    train_size=0.8, random_state=33)
ss_x = preprocessing.StandardScaler()
train_x_disorder = ss_x.fit_transform(train_x_disorder)


test_x_disorder = ss_x.transform(test_x_disorder)   

ss_y = preprocessing.StandardScaler()
train_y_disorder = ss_y.fit_transform(train_y_disorder.reshape(-1, 1))
test_y_disorder=ss_y.transform(test_y_disorder.reshape(-1, 1))

model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
Dtrain=xgb.DMatrix(train_x_disorder,train_y_disorder)
Dtest=xgb.DMatrix(test_x_disorder,test_y_disorder)
param = {'silent':1, 'objective':'reg:gamma', 'booster':'gbtree', 'base_score':3}
watchlist = [(Dtest, 'eval'), (Dtrain, 'train')]
num_round = 50

# training and evaluation
bst = xgb.train(param,Dtrain, num_round, watchlist)
preds = bst.predict(Dtest)
labels = Dtest.get_label()
print('test deviance=%f' % (2 * np.sum((labels - preds) / preds - np.log(labels) + np.log(preds))))

plt.plot(labels)
plt.plot(preds)