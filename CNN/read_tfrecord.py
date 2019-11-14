import tensorflow as tf
#import creat_tfrecord

#%%
def read_and_decode():
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(["wbc_train_252.tfrecords"])  #left_up_7820.tfrecords   wbc_train_89.tfrecords  wbc_train_252.tfrecords

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [128, 128, 3])
    #img = tf.reshape(img, [227, 227, 3])
    #img = tf.reshape(img, [39, 39, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    print (img,label)
    return img, label
    
def get_batch(image, label, batch_size,crop_size=64,capacity=20000):  
        #数据扩充变换  
    distorted_image = tf.random_crop(image, [crop_size, crop_size, 3])#随机裁剪  
    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转  
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=2)#亮度变化  
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化  
    distorted_image = tf.image.per_image_standardization(distorted_image)
    #生成batch  
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大  
    #保证数据打的足够乱  
    #images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
    #                                            num_threads=16,capacity=50000,min_after_dequeue=10000)  
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
                                                 num_threads=64,capacity=capacity,min_after_dequeue=batch_size-1) 
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size)  
    image_batch = images
    # 调试显示  
    #tf.image_summary('images', images)  
    print ("in get batch")
    print (image_batch,label_batch)
    return image_batch,label_batch

def get_batch1(image, label, batch_size, capacity=20000):  
    distorted_image = image
    #distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转  
    #distorted_image = tf.image.random_brightness(distorted_image,max_delta=2)#亮度变化  
    #distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)#对比度变化
    distorted_image = tf.image.per_image_standardization(distorted_image)
#    '''
#    Args:
#        image: list type   要生成batch的图像和标签list
#        label: list type
#        image_W: image width   图片的宽高
#        image_H: image height
#        batch_size: batch size   每个batch有多少张图片
#        capacity: the maximum elements in queue   队列容量 
#    Returns:
#        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
#        label_batch: 1D tensor [batch_size], dtype=tf.int32
#		图像和标签的batch 
#    '''
    #生成batch  
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大  
    #保证数据打的足够乱  
    #images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
    #                                             num_threads=16,capacity=50000,min_after_dequeue=10000)  
    images, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,  
                                                 num_threads=64,capacity=capacity,min_after_dequeue=batch_size-1) 
    #images, label_batch=tf.train.batch([distorted_image, label],batch_size=batch_size,num_threads=64,capacity=capacity)  
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(images, tf.float32)
    # 调试显示  
    #tf.image_summary('images', images)  
    print ("in get batch")
    print (image_batch,label_batch)
    #return (images, tf.reshape(label_batch, [batch_size]) )
    return image_batch, label_batch


#%%


#%% TEST
# To test the generated batches of images
# When training the model, DO comment the following codes


#import numpy as np
##
#import matplotlib.pyplot as plt
#
#BATCH_SIZE = 5
#CAPACITY = 256
#IMG_W = 128
#IMG_H = 128
#batch_size = 10
##train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
#train_dir = '/home/zzh/image_test_wbc/01_cats_vs_dogs/data/CatVSdogtrain/train/'
##image_list, label_list = get_files(train_dir)
#image_list, label_list = read_and_decode()
##image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, batch_size, CAPACITY)
#image_batch, label_batch = get_batch1(image_list, label_list, batch_size, capacity=2000)
#with tf.Session() as sess:
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            
#            img, label = sess.run([image_batch, label_batch])
#            
#            # just test one batch
#            for j in np.arange(batch_size):
#                print('label: %d' %label[j])
#                plt.imshow(img[j,:,:,:])
#                plt.show()
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads)


#%%





