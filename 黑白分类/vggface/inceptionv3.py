from keras.applications.inception_v3 import InceptionV3
#from keras.layers import Dense, GlobalAveragePooling2D
#from keras.layers.pooling import AveragePooling2D
#from keras.layers.convolutional import Conv2D  
#from keras.layers.pooling import MaxPooling2D 
from keras.layers import Dense, GlobalMaxPooling2D
from keras import backend as K
from keras.layers import Dropout
from keras.initializers import RandomNormal
#from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import pickle
from keras import optimizers
#from keras.layers.advanced_activations import LeakyReLU
#from keras.regularizers import l2
import numpy as np
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
save_model_path = './Kerasinv3_image_classification'
batch_size = 32 
num_classes = 85
epochs = 20


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

#
# def get_center_loss(features, labels, alpha, num_classes):
#     """获取center loss及center的更新op
#         features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
#         labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
#         alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
#         num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
#     Return：
#         loss: Tensor,可与softmax loss相加作为总的loss进行优化.
#         centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
#     """
#     # 获取特征的维数，例如256维
#     len_features = features.get_shape()[1]
#     # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
#     # 设置trainable=False是因为样本中心不是由梯度进行更新的
#     centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
#                               initializer=tf.constant_initializer(0), trainable=False)
#     # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
#     labels = tf.reshape(labels, [-1])
#
#     # 根据样本label,获取mini-batch中每一个样本对应的中心值
#     centers_batch = tf.gather(centers, labels)
#     # 计算loss
#     loss = tf.div(tf.nn.l2_loss(features - centers_batch), int(len_features))
#     # 当前mini-batch的特征值与它们对应的中心值之间的差
#     diff = centers_batch - features
#     # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
#     unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
#     appear_times = tf.gather(unique_count, unique_idx)
#     appear_times = tf.reshape(appear_times, [-1, 1])
#
#     diff = diff / tf.cast((1 + appear_times), tf.float32)
#     diff = alpha * diff
#
#     centers_update_op = tf.scatter_sub(centers, labels, diff)
#     return loss, centers_update_op


# def MyActivationLayer(x):
#     return tf.nn.relu(x) * tf.nn.tanh(x)
# this could also be the output a different Keras model or layer
#input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32,[None,85])

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
#x1 = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid', data_format=None)(x)
#x = GlobalAveragePooling2D()(x)
x = GlobalMaxPooling2D()(x)
# let's add a fully-connected layer
#x1 = Dense(4096, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(x)
x1 = Dense(4096, activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.01),bias_initializer='zeros')(x)
x2 = Dropout(0.7)(x1)
# x2 = BatchNormalization(momentum=0.9, epsilon=0.001,beta_initializer='zeros', gamma_initializer='ones' )(x1)
x3 = Dense(2048, activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.01),bias_initializer='zeros')(x2)
# x4 = BatchNormalization(momentum=0.9, epsilon=0.001,beta_initializer='zeros', gamma_initializer='ones')(x3)
# # and a logistic layer -- let's say we have 200 classes
# predictions = Dense(85, activation='softmax')(x4)
#x1 = Dense(4096, activation='relu',kernel_initializer='glorot_normal',bias_initializer='zeros')(x)
#x2 = Dropout(0.8)(x1)
#predictions = Conv2D(85,[1,1],kernel_initializer='glorot_normal',bias_initializer='zeros')(x2)
#x3 = Dense(2048, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(x2)
#x3 = Dense(2048, activation='relu',kernel_initializer='glorot_normal',bias_initializer='zeros')(x2)
x4 = Dropout(0.7)(x3)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(85, activation='softmax')(x4)
#result = tf.nn.softmax(predictions)

# this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# def get_loss(sess,batch_features,batch_labels,i):
#     Loss = 0
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
#     Loss_ = sess.run(cost, feed_dict={input_tensor: batch_features, y: batch_labels, K.learning_phase(): 1})
#     Loss += Loss_
#     Loss /= i
#     # loss_list = []
#     # loss_list.append(Loss)
#     # cost = np.mean(loss_list)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(Loss)
#     sess.run(optimizer, feed_dict={input_tensor: batch_features, y: batch_labels, K.learning_phase(): 1})
#     return Loss

# 损失函数定义
# centerloss, centers_update_op = get_center_loss(predictions, tf.argmax(y, 1), 0.5, 85)
# cost = tf.nn.softmax(predictions) + 0.05 * centerloss
# tf.summary.scalar('loss',self.loss)
# 优化器
#optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(cost)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
#cost = tf.reduce_mean(tf.abs(tf.subtract(tf.argmax(result, 1), tf.argmax(y, 1))))
#cost = tf.reduce_sum(tf.where(tf.greater(tf.arg_max(tf.nn.softmax(predictions),1),tf.arg_max(y,1))),0.9*predictions,1.1*predictions)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01,momentum=0.9,decay=1e-6).minimize(cost)
mae = tf.reduce_mean(tf.abs(tf.subtract(tf.argmax(predictions, 1), tf.argmax(y, 1))))

# train the model on the new data for a few epochs  2
sess = tf.Session() # 创建session
K.set_session(sess)
#K.set_learning_phase(1)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
print('Training...')
for epoch in range(epochs):
    #saver.restore(sess, save_model_path)
    # Loop over all batches
    n_batches = 13
    for batch_i in range(1, n_batches + 1):
        for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
            Loss = sess.run(cost, feed_dict={input_tensor: batch_features, y: batch_labels, K.learning_phase(): 1})
            sess.run(optimizer, feed_dict={input_tensor: batch_features, y: batch_labels, K.learning_phase(): 1})
            # Accuracy = session.run(accuracy,feed_dict={x:valid_features,y:valid_labels,keep_prob:1.0})
            MAE = sess.run(mae, feed_dict={input_tensor: batch_features, y: batch_labels,K.learning_phase(): 1})

        print('Epoch {:>2}, Image Batch {}:  '.format(epoch + 1, batch_i), end='')
        print('Loss  {:.6f} - MAE {:.6f}'.format(Loss, MAE))

save_path = saver.save(sess, save_model_path)
