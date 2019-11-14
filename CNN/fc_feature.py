# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:54:53 2017

@author: 李帆
"""

import pickle
import tensorflow as tf
import numpy as np
import scipy.io as sio

save_model_path = './image_classification'


def neural_net_image_input(image_shape):
    image_input = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]],name='x')
    return image_input


def neural_net_label_input(n_classes):
    label_input = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y')
    return label_input


def neural_net_keep_prob_input():
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return keep_prob


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    w = tf.Variable(
        tf.truncated_normal([conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs],
                            stddev=0.05))
    b = tf.Variable(tf.truncated_normal([conv_num_outputs], stddev=0.05))
    x = tf.nn.conv2d(x_tensor, w, [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1],
                       padding='SAME')
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

def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


tf.reset_default_graph()

# Inputs
x = neural_net_image_input((100, 100, 3))
y = neural_net_label_input(85)
keep_prob = neural_net_keep_prob_input()

con1 = conv2d_maxpool(x, 32, (3, 3), (2, 2), (2, 2), (2, 2))
con2 = conv2d_maxpool(con1, 64, (3, 3), (1, 1), (2, 2), (2, 2))
con3 = conv2d_maxpool(con2, 128, (2, 2), (1, 1), (2, 2), (2, 2))

fla = flatten(con3)

fc1 = fully_conn(fla, 1024)
fc1_1 = tf.nn.dropout(fc1, keep_prob)
fc2 = fully_conn(fc1_1, 512)
fc2_2 = tf.nn.dropout(fc2, keep_prob)
fc3 = fully_conn(fc2_2, 128)
fc = tf.nn.dropout(fc3, keep_prob)

logits = output(fc, 85)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
#optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
# correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
saver = tf.train.Saver()
batch_size = 64

with tf.Session() as sess:
    saver.restore(sess, save_model_path)
    n_batches = 17
    batch_i = 1
    print('正在运行')
    fea = np.zeros((1, 512))
    fea_lab = np.zeros((1, 85))
    for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
        batch_fea = sess.run(fc2, feed_dict={x: batch_features,keep_prob: 1.0})
        fea = np.row_stack((fea, batch_fea))
        fea_lab = np.row_stack((fea_lab, batch_labels))
    fea = np.delete(fea, 0, axis=0)
    fea_lab = np.delete(fea_lab, 0, axis=0)
    for batch_i in range(2, n_batches + 1):
        for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
            batch_fea = sess.run(fc2, feed_dict={x: batch_features,keep_prob: 1.0})
            fea = np.row_stack((fea, batch_fea))
            fea_lab = np.row_stack((fea_lab, batch_labels))
    print('运行完毕')

print('正在存储')

save_fea = 'fc_layer_fea.mat'
save_fea_lab = 'fc_layer_fea_lab.mat'
sio.savemat(save_fea,{'fc_feature':fea})
sio.savemat(save_fea_lab,{'fc_feature_lab':fea_lab})
print('存储完毕')


