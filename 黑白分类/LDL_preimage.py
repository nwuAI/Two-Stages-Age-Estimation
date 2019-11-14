# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 17:10:03 2017

@author: 李帆
"""

import numpy as np
import pickle
from scipy import stats

'''
归一化输入
'''


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (100, 100, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    a = 0
    b = 1
    grayscale_min = np.min(x)
    grayscale_max = np.max(x)
    return a + ((x - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min)

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    x = x.astype(int)
    num_labels = x.shape[0]
    x_one_hot = np.zeros((num_labels,2))

    for i in np.arange(num_labels):
        x_one_hot[i][x[i]] = 1
    return x_one_hot

def load_image(batch_id):
    with open('data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = np.array(batch['data']).reshape((len(batch['data']), 3, 224, 224)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels


def _preprocess_and_save(normalize,one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

def load_testimage(test_batch_id):
    with open('test_batch_' + str(test_batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    test_features = np.array(batch['data']).reshape((len(batch['data']), 3, 224, 224)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    return test_features, test_labels


n_batches = 8
#valid_features = []
#valid_labels = []

for batch_i in range(1, n_batches + 1):
    features, labels = load_image(batch_i)
    #validation_count = int(len(features) * 0.05)

    _preprocess_and_save(
        normalize,
		one_hot_encode,
        features,
        labels,
        #features[:-validation_count],
        #labels[:-validation_count],
        'preprocess_batch_' + str(batch_i) + '.p')

    #valid_features.extend(features[-validation_count:])
    #valid_labels.extend(labels[-validation_count:])


n_testbatches = 2


for test_batch_i in range(1, n_testbatches+ 1):
    test_features, test_labels = load_testimage(test_batch_i )

    _preprocess_and_save(
        normalize,
		one_hot_encode,
        test_features,
        test_labels,
        'preprocess_test_' + str(test_batch_i) + '.p')








