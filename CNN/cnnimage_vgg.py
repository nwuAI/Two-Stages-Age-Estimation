# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 10:42:16 2017

@author: 李帆
"""

import pickle
import tensorflow as tf
from keras.layers.normalization import BatchNormalization

# Load the Preprocessed Validation data
#valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

def neural_net_image_input(image_shape):
 
    image_input = tf.placeholder(dtype=tf.float32,shape=[None,image_shape[0],image_shape[1],image_shape[2]],name='x')
    return image_input


def neural_net_label_input(n_classes):
  
    label_input = tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name='y')
    return label_input


def neural_net_keep_prob_input():
   
    keep_prob = tf.placeholder(dtype=tf.float32,name='keep_prob')
    return keep_prob

'''
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    
    w = tf.Variable(tf.truncated_normal([conv_ksize[0],conv_ksize[1],x_tensor.get_shape().as_list()[3],conv_num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([conv_num_outputs],stddev=0.05))
    x=tf.nn.conv2d(x_tensor,w,[1,conv_strides[0],conv_strides[1],1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    x=tf.nn.relu(x)
    x=tf.nn.max_pool(x,[1,pool_ksize[0],pool_ksize[1],1],[1,pool_strides[0],pool_strides[1],1],padding='SAME')
    #x = tf.nn.lrn(x,4,bias = 1.0,alpha=0.001/9.0,beta = 0.75)
    #x=tf.nn.relu(x)
    return x    
'''


def convlayer(x_tensor, conv_num_outputs, conv_ksize, conv_strides):
    w = tf.Variable(
        tf.truncated_normal([conv_ksize[0], conv_ksize[1], x_tensor.get_shape().as_list()[3], conv_num_outputs],
                            stddev=1e-1))
    b = tf.Variable(tf.constant(0.0 , shape = [conv_num_outputs], dtype=tf.float32))
    x = tf.nn.conv2d(x_tensor, w, [1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    x = tf.nn.elu(x)
	x = BatchNormalization(momentum=0.9, epsilon=0.001)(x)
    # x = tf.nn.lrn(x,4,bias = 1.0,alpha=0.001/9.0,beta = 0.75)
    return x




def maxpoollayer(x_tensor, pool_ksize, pool_strides):

    x = tf.nn.max_pool(x_tensor, [1, pool_ksize[0], pool_ksize[1], 1], [1, pool_strides[0], pool_strides[1], 1],
                       padding='SAME')
    # x = tf.nn.lrn(x,4,bias = 1.0,alpha=0.001/9.0,beta = 0.75)
    x=tf.nn.elu(x)
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
    fc1 = tf.nn.elu(fc1)
    return fc1
    
    
def output(x_tensor, num_outputs):
    
    batch,size = x_tensor.get_shape().as_list()
    w = tf.Variable(tf.truncated_normal([size,num_outputs],stddev=0.05))
    b = tf.Variable(tf.truncated_normal([num_outputs],stddev=0.05))
    fc1 = tf.add(tf.matmul(x_tensor,w),b)
    return fc1    

def conv_net(x, keep_prob):

    conv1_1 = convlayer(x,64,(3,3),(1,1))
    conv1_2 = convlayer(conv1_1,64,(3,3),(1,1))
    pool1 = maxpoollayer(conv1_2,(2,2),(2,2))


    conv2_1 = convlayer(pool1,128,(3,3),(1,1))
    con2_2 = convlayer(conv2_1, 128, (3, 3), (1, 1))
    pool2 = maxpoollayer(con2_2, (2, 2), (2, 2))


    conv3_1 = convlayer(pool2,256,(3,3),(1,1))
    conv3_2 = convlayer(conv3_1,256,(3,3),(1,1))
    conv3_3 = convlayer(conv3_2,256,(3,3),(1,1))
    pool3 = maxpoollayer(conv3_3,(2,2),(2,2))


    conv4_1 = convlayer(pool3,512,(3,3),(1,1))
    conv4_2 = convlayer(conv4_1,512,(3,3),(1,1))
    conv4_3 = convlayer(conv4_2,512,(3,3),(1,1))
    pool4 = maxpoollayer(conv4_3,(2,2),(2,2))


    conv5_1 = convlayer(pool4,512,(3,3),(1,1))
    conv5_2 = convlayer(conv5_1,512,(3,3),(1,1))
    conv5_3 = convlayer(conv5_2,512,(3,3),(1,1))
    pool5 = maxpoollayer(conv5_3,(2,2),(2,2))

    con = flatten(pool5)

    fc1 = fully_conn(con, 4096)
    fc1_1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = fully_conn(fc1_1, 2048)
    fc2_2 = tf.nn.dropout(fc2, keep_prob)
    fc3 = fully_conn(fc2_2, 1000)
    fc3_3 = tf.nn.dropout(fc3, keep_prob)

    out = output(fc3_3,85)
    
    return out

tf.reset_default_graph()
# Inputs
x = neural_net_image_input((224, 224, 3))
y = neural_net_label_input(85)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
#optimizer = tf.train.AdamOptimizer().minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)


# Accuracy
#correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
mae = tf.reduce_mean(tf.abs(tf.subtract(tf.argmax(logits, 1), tf.argmax(y, 1))))


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    
    session.run(optimizer,feed_dict={x:feature_batch,y:label_batch,keep_prob:keep_probability})

    
    
def print_stats(session, feature_batch, label_batch, cost, mae):
   
    Loss = session.run(cost,feed_dict={x:feature_batch,y:label_batch,keep_prob:1.0})
    #Accuracy = session.run(accuracy,feed_dict={x:valid_features,y:valid_labels,keep_prob:1.0})
    MAE = session.run(mae,feed_dict={x:feature_batch,y:label_batch,keep_prob:1.0})
    print('Loss  {:.6f} - MAE {:.6f}'.format(Loss,MAE))

    
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

epochs = 100
batch_size = 64
keep_probability = 0.5

'''
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)


'''
save_model_path = './prevgg_image_classification'
#save_model_path = './vgg16_weights.h5'
saver = tf.train.Saver()

print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_model_path)
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 17
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, Image Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, mae)
            
    #Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)








    
    
    
    
    
    
