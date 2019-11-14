# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:52:57 2017

@author: 李 帆
"""
import numpy as np
import scipy.io as sio
#from pandas import DataFrame
#from xgboost.sklearn import XGBClassifier 
import xgboost as xgb
import time 
from sklearn.model_selection import GridSearchCV 
import h5py

start_time = time.time()



#inv3_trainfea = 'invfc_layer_fea.mat'
#inv3_trainlab = 'invfc_layer_fea_lab.mat'
#inv3_teatfea = 'invtestfc_layer_fea.mat'
#inv3_testlab = 'invtestfc_layer_fea_lab.mat'

inv3_trainfea = 'train_decmorlbp.mat'
inv3_trainlab = 'invfc_layer_fea_lab.mat'
inv3_teatfea = 'test_decmorlbp.mat'
inv3_testlab = 'invtestfc_layer_fea_lab.mat'

dic_train_feature = h5py.File(inv3_trainfea)
dic_train_lab = sio.loadmat(inv3_trainlab)

dic_test_feature = h5py.File(inv3_teatfea)
dic_test_lab = sio.loadmat(inv3_testlab)


train_feature =np.transpose( dic_train_feature['train_feature'])
train_lab_dis = dic_train_lab['invfc_feature_lab']
train_lab = np.argmax(train_lab_dis,axis=1)

test_feature = np.transpose(dic_test_feature['test_feature'])
test_lab_dis = dic_test_lab['invtestfc_feature_lab']
test_lab = np.argmax(test_lab_dis,axis=1)

xgb_train = xgb.DMatrix(train_feature, label=train_lab)
xgb_test = xgb.DMatrix(test_feature,label=test_lab)
y=xgb_test.get_label() 

#def cross_ent(y_pred, y):  
#    #p = 1.0 / (1.0 + np.exp(-y_hat)) 
#    g = -(y.get_label() / y_pred)
##    p = 1.0 / (1.0 + np.exp(-y_pred))  
##    g = p - y.get_label() 
#    h = y.get_label() / (y_pred**2)
##    h = p * (1.0-p)  
#    return g, h  
##    g = y_hat - y.get_label()
###    h = y.get_label() / (y_hat**2)  
##    h = y.get_label() / y_hat
##    return g, h 
#
#def mae_fun(y_pred,y):
#    return 'mae ',np.mean(abs(np.subtract(y_pred,y.get_label())))

##参数
param={
'booster':'gbtree',
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.1, # 如同学习率
'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':10, # 构建树的深度，越大越容易过拟合
'gamma':0.2,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样 
'lambda':6,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
#'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
'objective': 'multi:softmax', #多分类的问题
'num_class':85, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子
#'eval_metric': 'auc'
}
#plst = list(param.items())
num_rounds = 50 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]
#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
#model = xgb.train(param, xgb_train, num_rounds, watchlist,obj=cross_ent,feval=mae_fun)
model = xgb.train(param, xgb_train, num_rounds, watchlist)
#model.save_model('./model/xgb.model') # 用于存储训练出的模型
print("best best_ntree_limit",model.best_ntree_limit)
y_pred = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
#mae = np.mean(abs(np.subtract(y_pred,np.argmax(test_lab, 1))))
print(y_pred.shape)
print(type(y_pred))
mae = np.mean(abs(np.subtract(y_pred,test_lab)))
print(mae)
#print ('error=%f' % (  sum(1 for i in range(len(y_pred)) if int(y_pred[i]>0.5)!=y_test[i]) /float(len(y_pred))))  
#输出运行时长
cost_time = time.time()-start_time
print("xgboost success!",'\n',"cost time:",cost_time,"(s)......")

#model.save_model('./model/xgb.model') # 用于存储训练出的模型

#model.load_model('./model/xgb.model') # load data



