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
batch_size = 30
num_classes = 85
epochs = 100


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
#x = Dense(hidden_dim, activation='relu', name='fc6')(x)
#x = Dense(hidden_dim, activation='relu', name='fc7')(x)
#out = Dense(nb_class, activation='softmax', name='fc8')(x)
#custom_vgg_model = Model(vgg_model.input, out)


# this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
#for layer in base_model.layers:
 #   layer.trainable = False

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01,momentum=0.9,decay=1e-6).minimize(cost)
mae = tf.reduce_mean(tf.abs(tf.subtract(tf.argmax(predictions, 1), tf.argmax(y, 1))))

# train the model on the new data for a few epochs  2
sess = tf.Session() # 创建session
K.set_session(sess)
#K.set_learning_phase(1)
saver = tf.train.Saver()
print('Training...')
sess.run(tf.global_variables_initializer())
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
