# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:54:53 2017

@author: 李帆
"""

import pickle
import tensorflow as tf
import scipy.io as sio
import numpy as np
# Load the Preprocessed Validation data
test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
save_model_path = './vgg_image_classification'


def neural_net_image_input(image_shape):
    image_input = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],
                                 name='x')
    return image_input


def neural_net_label_input(n_classes):
    label_input = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    return label_input


def neural_net_keep_prob_input():
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return keep_prob


def convlayer(x_tensor, conv_num_outputs, conv_ksize, conv_strides):
    w = tf.Variable(
        tf.truncated_normal([conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs],
                            stddev=1e-1))
    b = tf.Variable(tf.constant(0.0, shape=[conv_num_outputs], dtype=tf.float32))
    x = tf.nn.conv2d(x_tensor, w, [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    # x = tf.nn.lrn(x,4,bias = 1.0,alpha=0.001/9.0,beta = 0.75)
    return x


def maxpoollayer(x_tensor, pool_ksize, pool_strides):
    x = tf.nn.max_pool(x_tensor, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1],
                       padding='SAME')
    # x = tf.nn.lrn(x,4,bias = 1.0,alpha=0.001/9.0,beta = 0.75)
    # x=tf.nn.relu(x)
    return x


def flatten(x_tensor):
    x_shape = x_tensor.get_shape().as_list()
    x_tensor = tf.reshape(x_tensor, shape=[-1, x_shape[1] * x_shape[2] * x_shape[3]])

    return x_tensor


def fully_conn(x_tensor, num_outputs):
    batch, size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size, num_outputs], stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs], stddev=0.05))
    fc1 = tf.matmul(x_tensor, w)
    fc1 = tf.add(fc1, b)
    fc1 = tf.nn.relu(fc1)
    return fc1


def output(x_tensor, num_outputs):
    batch, size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size, num_outputs], stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs], stddev=0.05))
    fc1 = tf.add(tf.matmul(x_tensor, w), b)
    return fc1


tf.reset_default_graph()
# Inputs
x = neural_net_image_input((100, 100, 3))
y = neural_net_label_input(85)
keep_prob = neural_net_keep_prob_input()

conv1_1 = convlayer(x, 64, (3, 3), (1, 1))
conv1_2 = convlayer(conv1_1, 64, (3, 3), (1, 1))
pool1 = maxpoollayer(conv1_2, (2, 2), (2, 2))

conv2_1 = convlayer(pool1, 128, (3, 3), (1, 1))
con2_2 = convlayer(conv2_1, 128, (3, 3), (1, 1))
pool2 = maxpoollayer(con2_2, (2, 2), (2, 2))

conv3_1 = convlayer(pool2, 256, (3, 3), (1, 1))
conv3_2 = convlayer(conv3_1, 256, (3, 3), (1, 1))
conv3_3 = convlayer(conv3_2, 256, (3, 3), (1, 1))
pool3 = maxpoollayer(conv3_3, (2, 2), (2, 2))

conv4_1 = convlayer(pool3, 512, (3, 3), (1, 1))
conv4_2 = convlayer(conv4_1, 512, (3, 3), (1, 1))
conv4_3 = convlayer(conv4_2, 512, (3, 3), (1, 1))
pool4 = maxpoollayer(conv4_3, (2, 2), (2, 2))

conv5_1 = convlayer(pool4, 512, (3, 3), (1, 1))
conv5_2 = convlayer(conv5_1, 512, (3, 3), (1, 1))
conv5_3 = convlayer(conv5_2, 512, (3, 3), (1, 1))
pool5 = maxpoollayer(conv5_3, (2, 2), (2, 2))

con = flatten(pool5)

fc1 = fully_conn(con, 4096)
fc1_1 = tf.nn.dropout(fc1, keep_prob)
fc2 = fully_conn(fc1_1, 4096)
fc2_2 = tf.nn.dropout(fc2, keep_prob)
fc3 = fully_conn(fc2_2, 1000)
fc3_3 = tf.nn.dropout(fc3, keep_prob)

logits = output(fc3_3, 85)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_model_path)
    test_fea = sess.run(fc2, feed_dict={x: test_features, keep_prob: 1.0})

print('正在存储')

save_fea = 'vgg_test_fea.mat'
save_fea_lab = 'vgg_test_fea_lab.mat'
sio.savemat(save_fea,{'vgg_test_feature':test_fea})
sio.savemat(save_fea_lab,{'vgg_test_lab':test_labels})
print('存储完毕')