from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace

from keras.applications.inception_v3 import InceptionV3
#from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense,GlobalMaxPooling2D
from keras import backend as K
from keras.layers import Dropout
import numpy as np
import tensorflow as tf
import pickle
import scipy.io as sio

batch_size = 32

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
	
input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])

# Convolution Features
vgg_features = VGGFace(input_tensor=input_tensor,include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max
x = vgg_features.output
# After this point you can use your model to predict.

sess = tf.Session() # 创建session
K.set_session(sess)

n_batches = 17
batch_i = 1
print('正在运行')
invfea = np.zeros((1, 2048))
invfea_lab = np.zeros((1, 85))
for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
    batch_fea = sess.run(x,  feed_dict={input_tensor: batch_features})
    print(np.shape(batch_fea))
 
 