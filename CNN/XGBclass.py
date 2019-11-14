# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:34:21 2017

@author: 李 帆
"""
import numpy as np
import scipy.io as sio
#from pandas import DataFrame
from xgboost.sklearn import XGBClassifier 
import xgboost as xgb
import time 
from sklearn.model_selection import GridSearchCV 

start_time = time.time()



inv3_trainfea = 'invfc_layer_fea.mat'
inv3_trainlab = 'invfc_layer_fea_lab.mat'
inv3_teatfea = 'invtestfc_layer_fea.mat'
inv3_testlab = 'invtestfc_layer_fea_lab.mat'

dic_train_feature = sio.loadmat(inv3_trainfea)
dic_train_lab = sio.loadmat(inv3_trainlab)

dic_test_feature = sio.loadmat(inv3_teatfea)
dic_test_lab = sio.loadmat(inv3_testlab)


train_feature = dic_train_feature['invfc_feature']
train_lab_dis = dic_train_lab['invfc_feature_lab']
train_lab = np.argmax(train_lab_dis,axis=1)

test_feature = dic_test_feature['invtestfc_feature']
test_lab_dis = dic_test_lab['invtestfc_feature_lab']
test_lab = np.argmax(test_lab_dis,axis=1)
#print(type(train_feature))

xgb_train = xgb.DMatrix(train_feature, label=train_lab)
xgb_test = xgb.DMatrix(test_feature,label=test_lab)
#xgb1 = XGBClassifier(
# learning_rate =0.1,
# n_estimators=85,
# max_depth=3,
# min_child_weight=1,
# gamma=0,
# subsample=0.8,
# colsample_bytree=0.8,
# objective= 'binary:logistic',
## nthread=4,
# scale_pos_weight=1,
# seed=27)

param_test1 = {
 'max_depth':range(5,20,3),
 'min_child_weight':range(1,6,2)
}

#param_test2 = {
# 'min_child_weight':[]
#}

#param_test3 = {
# 'gamma':[i/10.0 for i in range(0,5)]
#}
#
#param_test4 = {
# 'subsample':[i/10.0 for i in range(6,10)],
# 'colsample_bytree':[i/10.0 for i in range(6,10)]
#}
#
#param_test5 = {
# 'subsample':[i/100.0 for i in range(75,90,5)],
# 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
#}
#
#param_test6 = {
# 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
#}
#
#param_test7 = {
# 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
#}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=100, max_depth=3,
 min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', num_class=85,scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train_feature,test_lab)
print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)




