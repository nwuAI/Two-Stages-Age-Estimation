# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:52:57 2017

@author: 李 帆
"""
import numpy as np
#import scipy.io as sio
#from pandas import DataFrame
#from xgboost.sklearn import XGBClassifier 
import xgboost as xgb
#import time 
#from sklearn.model_selection import GridSearchCV 
import h5py
from sklearn.model_selection import train_test_split

#start_time = time.time()

#feature = 'sub5_LGP_result.mat'
#feature = h5py.File(feature)
#feature = np.transpose(feature['sub5_LGP23'])
##train_feature = train_feature[0:937]
#print(feature.shape)

feature1 = 'DLGP123_result.mat'
feature1 = h5py.File(feature1)
feature1 = np.transpose(feature1['DLGP123'])
#train_feature = train_feature[0:937]
print(feature1.shape)

feature2 = 'hsv_fea_result.mat'
feature2 = h5py.File(feature2)
feature2 = np.transpose(feature2['hsv_fea'])
#train_feature = train_feature[0:937]
print(feature2.shape)

feature = np.hstack((feature1,feature2))
print(feature.shape)


label = 'LGP_label.mat'
label = h5py.File(label) 
label = np.transpose(label['label'])
#train_label = train_label[0:937]
print(label.shape)

train_feature,test_feature,train_label,test_label=train_test_split(feature,label,test_size=0.2,random_state=1)

#dic_train_feature = h5py.File(inv3_trainfea)
  #matlab保存的文件使用h5py  python保存的文件使用sio.mat
xgb_train = xgb.DMatrix(train_feature, label=train_label)
xgb_test = xgb.DMatrix(test_feature,label=test_label)

#def tran(input):
#    a = []
#    for i in input:
#        if i>=0.6:
#            b = 1
#        else:
#            b = 0
#        a.append(b)
#    return a
#    
###参数
param={
'booster':'gbtree',
'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.01, # 如同学习率
'min_child_weight':1, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':5, # 构建树的深度，越大越容易过拟合
'gamma':0.2,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样 
'lambda':5,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
#'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
#'objective': 'multi:softmax', #多分类的问题
'objective': 'binary:logistic',
#'num_class':10, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子
'eval_metric': 'auc'#auc适用于二分类，merror是多分类的错误率
}
#plst = list(param.items())
num_rounds = 5000 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]
#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
#model = xgb.train(param, xgb_train, num_rounds, watchlist,obj=cross_ent,feval=mae_fun)
model = xgb.train(param, xgb_train, num_rounds, watchlist)
#model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit",model.best_ntree_limit)
y_pred = model.predict(xgb_test,ntree_limit=model.best_ntree_limit) 
#arr = tran(y_pred)
#test_label = test_label.flatten()
#accury = np.mean(arr == test_label)
#print(accury)
#输出运行时长
#cost_time = time.time()-start_time
#print("xgboost success!",'\n',"cost time:",cost_time,"(s)......")
#
#model.save_model('./model/xgb.model') # 用于存储训练出的模型
#
#model.load_model('./model/xgb.model') # load data



