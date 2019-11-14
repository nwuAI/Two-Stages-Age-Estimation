# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:54:53 2017

@author: 李帆
"""

import pickle
import tensorflow as tf
import numpy as np

# Load the Preprocessed Validation data
test_features, test_labels = pickle.load(open('preprocess_test.p', mode='rb'))
save_model_path = './image_classification'

def neural_net_image_input(image_shape):
 
    image_input = tf.placeholder(dtype=tf.float32,shape=[None,image_shape[0],image_shape[1],image_shape[2]],name='x')
    return image_input


def neural_net_label_input(n_classes):
  
    label_input = tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name='y')
    return label_input


def neural_net_keep_prob_input():
   
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
    return keep_prob

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    
    w = tf.Variable(tf.truncated_normal([conv_ksize[0],conv_ksize[1],x_tensor.get_shape().as_list()[3],conv_num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([conv_num_outputs],stddev=0.05))
    x=tf.nn.conv2d(x_tensor,w,[1,conv_strides[0],conv_strides[1],1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    x=tf.nn.relu(x)
    x=tf.nn.max_pool(x,[1,pool_ksize[0],pool_ksize[1],1],[1,pool_strides[0],pool_strides[1],1],padding='SAME')
    #x=tf.nn.relu(x)
    return x    


def flatten(x_tensor):
    
    x_shape = x_tensor.get_shape().as_list()
    x_tensor = tf.reshape(x_tensor,shape=[-1,x_shape[1]*x_shape[2]*x_shape[3]])

    return x_tensor


def fully_conn(x_tensor, num_outputs):
    
    batch,size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size,num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.05))
    fc1 = tf.matmul(x_tensor,w)
    fc1 = tf.add(fc1,b)
    fc1 = tf.nn.relu(fc1)
    return fc1
    
    
def output(x_tensor, num_outputs):
    
    batch,size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size,num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.05))
    fc1 = tf.add(tf.matmul(x_tensor,w),b)
    return fc1

tf.reset_default_graph()

# Inputs
x = neural_net_image_input((100, 100, 3))
y = neural_net_label_input(85)
keep_prob = neural_net_keep_prob_input()

con = conv2d_maxpool(x, 32, (3, 3), (2, 2), (2, 2), (2, 2))
con = conv2d_maxpool(con, 64, (3, 3), (1, 1), (2, 2), (2, 2))
con = conv2d_maxpool(con, 128, (2, 2), (1, 1), (2, 2), (2, 2))

con = flatten(con)

fc1 = fully_conn(con, 1024)
fc1_1 = tf.nn.dropout(fc1, keep_prob)
fc2 = fully_conn(fc1_1, 512)
fc2_2 = tf.nn.dropout(fc2, keep_prob)
fc3 = fully_conn(fc2_2, 128)
fc = tf.nn.dropout(fc3, keep_prob)

logits = output(fc, 85)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

abs11 = tf.abs(tf.subtract(tf.argmax(logits, 1), tf.argmax(y, 1)))
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,save_model_path)
    MAEs = sess.run(abs11, feed_dict={x: test_features, y: test_labels, keep_prob: 1.0})
    #mae_error = sess.run(mae,feed_dict={x:valid_features,y:valid_labels,keep_prob:1.0})
MAE = np.mean(MAEs)
print('MAE {:.6f}'.format(MAE))

