from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.convolutional import Conv2D  
from keras.layers.pooling import MaxPooling2D 
from keras.layers import Dense, GlobalMaxPooling2D
from keras import backend as K
from keras.layers import Dropout
from keras.initializers import RandomNormal
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import pickle
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import numpy as np
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
save_model_path = './Kerasvgg_image_classification'
test_features, test_labels = pickle.load(open('preprocess_test_2.p', mode='rb'))
batch_size = 30
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

# def MyActivationLayer(x):
#     return tf.nn.relu(x) * tf.nn.tanh(x)
# this could also be the output a different Keras model or layer
#input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32,[None,85])

nb_class = 85
#hidden_dim = 512

vgg_model = VGGFace(weights='vggface',input_tensor=input_tensor,include_top=False, input_shape=(224, 224, 3),pooling='max')
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
#x1 = Dense(4096, activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.01),bias_initializer='zeros')(x)
#x2 = Dropout(0.7)(x1)
#x3 = Dense(2048, activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.01),bias_initializer='zeros')(x2)
#x4 = Dropout(0.7)(x3)
predictions = Dense(85, activation='softmax')(x)

abs11 = tf.abs(tf.subtract(tf.argmax(predictions, 1), tf.argmax(y, 1)))
test_features1 = test_features[1000:1200]
test_labels1 = test_labels[1000:1200]

sess = tf.Session()  # 创建sessionnvidia
K.set_session(sess)

saver = tf.train.Saver()
saver.restore(sess,save_model_path)


MAEs = sess.run(abs11, feed_dict={input_tensor: test_features1, y: test_labels1,K.learning_phase(): 0})
MAE = np.mean(MAEs)
print('MAE {:.6f}'.format(MAE))
