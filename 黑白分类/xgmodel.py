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
import pickle
import  random
import time 
import h5py
#

start_time = time.time()


#def batch_features_labels(features, labels, batch_size):
#    """
#    Split features and labels into batches
#    """
#    retuen features[0:batch_size],labels[0:batch_size]
#    for start in range(0, len(features), batch_size):
#        end = min(start + batch_size, len(features))
#        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
#    labels = np.argmax(labels,1)
#    num = len(labels)
#    arr = np.zeros((num, 1))
#    for i in range(num):
#        arr[i][0] = labels[i]
#    np.reshape(features,(2500,150528))
#    ind = [i for i in range(len(features))]
#    random.shuffle(ind)
#    features = features[ind]
#    labels = labels[ind]

    # Return the training data in batches of size <batch_size> or less
    return features[0:batch_size],labels[0:batch_size]
	
def load_preprocess_test_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_test_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
#    labels = np.argmax(labels,1)
#    num = len(labels)
#    arr = np.zeros((num, 1))
#    for i in range(num):
#        arr[i][0] = labels[i]
#    ind = [i for i in range(len(features))]
#    random.shuffle(ind)
#    features = features[ind]
#    labels = labels[ind]

    # Return the training data in batches of size <batch_size> or less
    return features[1200:batch_size],labels[1200:batch_size]
    #return batch_features_labels(features, labels, batch_size)
    
trainfea = 'bwclasslbp1.mat'
testfea = 'bwclass_testlbp1.mat'

dic_train_feature = h5py.File(trainfea)
dic_test_feature = h5py.File(testfea)

train_feature1 = np.transpose(dic_train_feature['dec_LBP_Feature'])
test_feature1 = np.transpose(dic_test_feature['dec_LBP_Feature'])
test_feature1 = test_feature1[1200:1300]
	
train_feature2,	train_lab = load_preprocess_training_batch(1,2500)
test_feature2, 	test_lab = load_preprocess_test_batch(1,1300)

train_feature = np.column_stack((train_feature1,train_feature2))
test_feature = np.column_stack((test_feature1,test_feature2))

xgb_train = xgb.DMatrix(train_feature, label=train_lab)
xgb_test = xgb.DMatrix(test_feature,label=test_lab)
#
def tran(input):
    a = []
    for i in input:
        if i>=0.6:
            b = 1
        else:
            b = 0
        a.append(b)
    return a
    

#def cross_ent(y_hat, y):  
#    p = 1.0 / (1.0 + np.exp(-y_hat)) 
#    #g = -(y / y_hat)
##    p = 1.0 / (1.0 + np.exp(-y_hat))  
#    g = p - y.get_label()  
#    h = p * (1-p)   
#    return g, h  
##    g = y_hat - y.get_label()
###    h = y.get_label() / (y_hat**2)  
##    h = y.get_label() / y_hat
##    return g, h 
#
#def mae_fun(y_hat,y):
#    return 'mae ',np.mean(abs(np.subtract(y_hat,y.get_label())))

##参数
param={
'booster':'gbtree',
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
#'nthread':7,# cpu 线程数 默认最大
'eta': 0.01, # 如同学习率
'min_child_weight':1, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
'max_depth':3, # 构建树的深度，越大越容易过拟合
'gamma':1.0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
'subsample':0.8, # 随机采样训练样本
'colsample_bytree':0.8, # 生成树时进行的列采样 
'lambda':100,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
#'alpha':0, # L1 正则项参数
#'scale_pos_weight':1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。
'objective': 'binary:logistic', #多分类的问题
#'num_class':85, # 类别数，多分类与 multisoftmax 并用
'seed':1000, #随机种子
#'eval_metric': 'auc'
}
plst = list(param.items())
num_rounds = 1000 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]

#训练模型并保存
##early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
###model = xgb.train(param, xgb_train, num_rounds, watchlist,obj=cross_ent,feval=mae_fun)
model = xgb.train(param, xgb_train, num_rounds, watchlist)
print("best best_ntree_limit",model.best_ntree_limit)
y_pred = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
print(y_pred)
arr = tran(y_pred)
test_lab = test_lab.flatten()
accury = np.mean(arr == test_lab)
print(accury)

###print ('error=%f' % (  sum(1 for i in range(len(y_pred)) if int(y_pred[i]>0.5)!=y_test[i]) /float(len(y_pred))))  
#输出运行时长
cost_time = time.time()-start_time
print("xgboost success!",'\n',"cost time:",cost_time,"(s)......")

#model.save_model('./model/xgb.model') # 用于存储训练出的模型
#
#model.load_model('./model/xgb.model') # load data



