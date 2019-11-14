from keras.applications.inception_v3 import InceptionV3
#from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense, GlobalMaxPooling2D
from keras import backend as K
from keras.layers import Dropout
import tensorflow as tf
import pickle
#import os
import numpy as np

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
test_features, test_labels = pickle.load(open('preprocess_test_2.p', mode='rb'))
save_model_path = './Kerasinv3_image_classification'

input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 1])

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = GlobalMaxPooling2D()(x)
# let's add a fully-connected layer
x1 = Dense(4096, activation='relu')(x)
x2 = Dropout(0.5)(x1)
#x3 = Dense(2048, activation='relu',kernel_initializer='glorot_normal',bias_initializer='zeros')(x2)
x3 = Dense(2048,activation='relu')(x2)
x4 = Dropout(0.5)(x3)
predictions = Dense(1)(x4)

def tran(input):
    a = []
    for i in input:
        if i>=0:
            b = 1
        else:
            b = 0
        a.append(b)
    return a

#correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
MAEALL = []
for i in range(120,121):
    test_features1 = test_features[i*10:(i+1)*10]
    test_labels1 = test_labels[i*10:(i+1)*10]
    labels = np.argmax(test_labels1,1)
    num = len(labels)
    arr = np.zeros((num, 1))
    for i in range(num):
        arr[i][0] = labels[i]
    sess = tf.Session()  # 创建session
    K.set_session(sess)

    saver = tf.train.Saver()
    saver.restore(sess, save_model_path)

    copr = sess.run(predictions, feed_dict={input_tensor: test_features1, y: arr, K.learning_phase(): 0})
    accuracy1 = np.mean(tran(copr))
    print(accuracy1)
    MAEALL.append(accuracy1)

accuracy = np.mean(MAEALL)
print('accuracy {:.6f}'.format(accuracy))

#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
# test_features1 = test_features[200:400]
# test_labels1 = test_labels[200:400]
#
# sess = tf.Session()  # 创建session
# K.set_session(sess)
#
# saver = tf.train.Saver()
# saver.restore(sess,save_model_path)
#
# copr = sess.run(correct_pred, feed_dict={input_tensor: test_features1, y: test_labels1, K.learning_phase(): 1})
# copr1 = sess.run(tf.argmax(y, 1), feed_dict={input_tensor: test_features1, y: test_labels1, K.learning_phase(): 1})
# #a = tran(copr)
# print(copr)
# print(copr1)
# accuracy = np.mean(tran(copr))
# print('accuracy {:.6f}'.format(accuracy))



