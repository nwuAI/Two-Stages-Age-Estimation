# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:15:51 2017

@author: 李 帆
"""

import numpy as np
import scipy.io as sio

a =  np.array([[1,2,3],
               [4,5,6],
               [7,8,9]])



save_fn = 'fc_layer_fea.mat'
save_fn_lab = 'fc_layer_fea_lab.mat'

#sio.savemat(save_fn, {'array': a})

load_data = sio.loadmat(save_fn)
load_matrix = load_data['fc_feature']
load_matrix_row = load_matrix[15]
print(load_matrix)
