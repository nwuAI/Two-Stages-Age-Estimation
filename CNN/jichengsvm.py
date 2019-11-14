
import  scipy.io as sio
from sklearn import svm
import  numpy as np
import h5py

#load train data
load_a = 'invfc_layer_fea.mat'
load_data = h5py.File(load_a)
x_train = load_data['invfc_feature']


load_b = 'invfc_layer_fea_lab.mat'
load_data1 = sio.loadmat(load_b)
y_train_dis = load_data1['invfc_feature_lab']
y_train = np.argmax(y_train_dis,axis=1)

#load test data
load_c = 'invtestfc_layer_fea.mat'
load_data2 = h5py.File(load_c)
x_test = load_data2['invtestfc_feature']

load_d = 'invtestfc_layer_fea_lab.mat'
load_data1 = sio.loadmat(load_d)
y_test_dis = load_data1['invtestfc_feature_lab']
y_test = np.argmax(y_test_dis,axis=1)

#train model
clf = svm.SVC(C=0.8, kernel='rbf', gamma='auto', decision_function_shape='ovr')
#clf = svm.SVR(C=1.0, epsilon=0.2)
clf.fit(x_train,y_train)

#train mae
y_hat = clf.predict(x_train)
#abs = np.abs(np.subtract(np.argmax(y_hat, 1), np.argmax(y_train, 1)))
abs = np.abs(np.subtract(y_hat, y_train))
MAE = np.mean(abs)
print('Train_MAE {:.6f}'.format(MAE))

#test mae
y_hat_test = clf.predict(x_test)
absl = np.abs(np.subtract(y_hat_test, y_test))
MAEtest = np.mean(absl)
print('Test_MAE {:.6f}'.format(MAEtest))

'''
a = np.array([[1,-1,-2,-7,8,9]])
print(type(a))
c = np.abs(a)
print(c)
d = np.mean(c)
print(d)
'''
