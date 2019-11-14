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
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
test_features, test_labels = pickle.load(open('preprocess_test_1.p', mode='rb'))
#test_features, test_labels = pickle.load(open('preprocess_batch_10.p', mode='rb'))
save_model_path = './Kerasinv3_image_classification'


batch_size = 32
num_classes = 85

def batch_features_labels(features, labels, batch_size):

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

input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 85])

base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
#x1 = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid', data_format=None)(x)
#x = GlobalAveragePooling2D()(x)
x = GlobalMaxPooling2D()(x)
x1 = Dense(4096, activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.01),bias_initializer='zeros')(x)
x2 = Dropout(0.7)(x1)
x3 = Dense(2048, activation='relu',kernel_initializer=RandomNormal(mean=0.0,stddev=0.01),bias_initializer='zeros')(x2)
x4 = Dropout(0.7)(x3)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(85, activation='softmax')(x4)


#vgg_model = VGGFace(input_tensor=input_tensor,include_top=False, input_shape=(224, 224, 3))
#last_layer = vgg_model.get_layer('pool5').output
#x = Flatten(name='flatten')(last_layer)
#x = Dense(hidden_dim, activation='relu', name='fc6')(x)
#x = Dense(hidden_dim, activation='relu', name='fc7')(x)
#out = Dense(nb_class, activation='softmax', name='fc8')(x)
abs11 = tf.abs(tf.subtract(tf.argmax(predictions, 1), tf.argmax(y, 1)))
test_features1 = test_features[100:200]
test_labels1 = test_labels[100:200]

sess = tf.Session()  # 创建session
K.set_session(sess)

saver = tf.train.Saver()
saver.restore(sess,save_model_path)


MAEs = sess.run(abs11, feed_dict={input_tensor: test_features1, y: test_labels1,K.learning_phase(): 0})
MAE = np.mean(MAEs)
print('MAE {:.6f}'.format(MAE))



