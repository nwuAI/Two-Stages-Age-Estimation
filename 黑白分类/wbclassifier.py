from keras.applications.inception_v3 import InceptionV3
#from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dense, GlobalMaxPooling2D
from keras import backend as K
from keras.layers import Dropout
import tensorflow as tf
import pickle
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import  random
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

save_model_path = './Kerasinv3_image_classification'
batch_size = 24
num_classes = 85
epochs = 2


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
    labels = np.argmax(labels,1)
    num = len(labels)
    arr = np.zeros((num, 1))
    for i in range(num):
        arr[i][0] = labels[i]
    ind = [i for i in range(len(features))]
    random.shuffle(ind)
    features = features[ind]
    labels = arr[ind]

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


# this could also be the output a different Keras model or layer
#input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'
input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32,[None,1])

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

# this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
def tran(input):
    a = []
    for i in input:
        if i>=0.5:
            b = 1
        else:
            b = 0
        a.append(b)
    return a
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=y))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
# train the model on the new data for a few epochs
sess = tf.Session() # 创建session
K.set_session(sess)
#K.set_learning_phase(1)
saver = tf.train.Saver()
print('Training...')
for epoch in range(epochs):
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, save_model_path)
    # Loop over all batches
    n_batches = 8
    for batch_i in range(1, n_batches + 1):
        for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
            Loss = sess.run(cost,feed_dict={input_tensor: batch_features, y: batch_labels, K.learning_phase(): 1})
            sess.run(optimizer, feed_dict={input_tensor: batch_features, y: batch_labels, K.learning_phase(): 1})
            copr = sess.run(tf.nn.sigmoid(predictions), feed_dict={input_tensor: batch_features, y: batch_labels,K.learning_phase(): 1})
            accuracy = np.mean(tran(copr))
            # a = tran(copr)
            # c = np.mean(a)
            # print(a)
            # print(c)
        print('Epoch {:>2}, Image Batch {}:  '.format(epoch + 1, batch_i), end='')
        print('Loss  {:.6f} - accuracy {:.6f}'.format(Loss, accuracy))
            # history = model.fit(batch_features,batch_labels,batch_size=batch_size,epochs=epochs,verbose=1)#显示日志


save_path = saver.save(sess, save_model_path)
