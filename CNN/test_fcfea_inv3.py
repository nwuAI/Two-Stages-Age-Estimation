from keras.applications.inception_v3 import InceptionV3
#from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense,GlobalMaxPooling2D
from keras import backend as K
from keras.layers import Dropout
import numpy as np
import tensorflow as tf
import pickle
import scipy.io as sio
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

save_model_path = './Kerasinv3_image_classification'
batch_size = 32
num_classes = 85
epochs = 10


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
    filename = 'preprocess_test_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)
	

# this could also be the output a different Keras model or layer
#input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32,[None,85])

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = GlobalMaxPooling2D()(x)
# let's add a fully-connected layer
#x1 = Dense(4096, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(x)
x1 = Dense(4096, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x)
x2 = Dropout(0.5)(x1)
#x3 = Dense(2048, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros')(x2)
x3 = Dense(2048, activation='relu',kernel_initializer='glorot_uniform',bias_initializer='zeros')(x2)
x4 = Dropout(0.5)(x3)

predictions = Dense(85, activation='softmax')(x4)

# train the model on the new data for a few epochs
sess = tf.Session() # 创建session
K.set_session(sess)
#K.set_learning_phase(1)
saver = tf.train.Saver()
 #sess.run(tf.global_variables_initializer())
saver.restore(sess, save_model_path)
 
n_batches = 4
batch_i = 1
print('正在运行')
invtestfea = np.zeros((1, 2048))
invtestfea_lab = np.zeros((1, 85))
for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
    batch_fea = sess.run(x3,  feed_dict={input_tensor: batch_features,K.learning_phase(): 0})
    invtestfea = np.row_stack((invtestfea, batch_fea))
    invtestfea_lab = np.row_stack((invtestfea_lab, batch_labels))
invtestfea = np.delete(invtestfea, 0, axis=0)
invtestfea_lab = np.delete(invtestfea_lab, 0, axis=0)
for batch_i in range(2, n_batches + 1):
    for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
        batch_fea = sess.run(x3, feed_dict={input_tensor: batch_features,K.learning_phase(): 0})
        invtestfea = np.row_stack((invtestfea, batch_fea))
        invtestfea_lab = np.row_stack((invtestfea_lab, batch_labels))
print('运行完毕')

print('正在存储')

save_fea = 'invtestfc_layer_fea.mat'
save_fea_lab = 'invtestfc_layer_fea_lab.mat'
sio.savemat(save_fea,{'invtestfc_feature':invtestfea})
sio.savemat(save_fea_lab,{'invtestfc_feature_lab':invtestfea_lab})
print('存储完毕')
