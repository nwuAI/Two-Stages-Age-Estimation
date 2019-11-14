
import pickle
from PIL import Image
import numpy as np
import os
import matplotlib.image as plimg
import h5py
import scipy.io as sio

class DictSave(object):
    def __init__(self,filenames):
        self.filenames = filenames

    #def image_input(self,filenames):
    def image_input(self,filenames,num):
        i = 0
        self.all_arr = [[0 for m in range(150528)] for n in range(len(filenames))]
        #self.label_all_arr = [[0] for k in range(len(filenames))]
        for filename in filenames:
            j = 0
            #self.arr, self.label_arr = self.read_file(filename)
            self.arr = self.read_file(filename,num)
            self.all_arr[i][j:len(self.arr)] = self.arr
            #self.label_all_arr[i][j:len(self.label_arr)] = self.label_arr
            i = i+1

    def read_file(self,filename,num):
    #def read_file(self,filename):
        #im = Image.open(os.path.join("14",filename),'r')#打开一个图像
        im = Image.open(os.path.join('image'+ str(num),filename),'r')#打开一个图像
        #site=filename.find('.')
        #label_arr = np.array([int(filename[site-2:site])])
        #label_arr = np.array([int(filename[0:2])])
        #label_arr = np.array([int(filename[0])])
        # 将图像的RGB分离
        r, g, b = im.split()
        # 将PILLOW图像转成数组
        r_arr = plimg.pil_to_array(r)
        g_arr = plimg.pil_to_array(g)
        b_arr = plimg.pil_to_array(b)

        # 将224*224二维数组转成50176的一维数组
        r_arr1 = r_arr.reshape(50176)
        g_arr1 = g_arr.reshape(50176)
        b_arr1 = b_arr.reshape(50176)
        # 3个一维数组合并成一个一维数组,大小为150528
        arr = np.concatenate((r_arr1, g_arr1, b_arr1))
        return arr
    def pickle_save(self,arr,label_arr,num):
    #def pickle_save(self,arr,label_arr):
        print ("正在存储")

        # 构造字典,所有的图像数据都在arr数组里,这里只存图像数据,没有存label
        #data_batch_14= {'data': arr ,'labels': label_arr}
        #f = open('data_batch_14', 'wb')

        #pickle.dump(data_batch_14, f)#把字典存到文本中去

        name='data_batch_' + str(num)
        dict= {'data': arr ,'labels': label_arr}
        f = open(name, 'wb')

        pickle.dump(dict, f)#把字典存到文本中去
        f.close()
        print ("存储完毕")

"""
if __name__ == "__main__":
    path = "14"
    filenames = os.listdir(path)

    ds = DictSave(filenames)
    ds.image_input(ds.filenames)
    ds.pickle_save(ds.all_arr,ds.label_all_arr)
    print(ds.label_all_arr)
    print ("最终数组的大小:"+str(np.shape(ds.label_all_arr))+str(np.shape(ds.all_arr)))
"""
if __name__ == "__main__":
    labelmix = 'label' + str(8) + '.mat'
    labelin = h5py.File(labelmix)
    label_all_arr = np.transpose(labelin['bwlabel'])
    for num in np.arange(8,9):
        path = 'image'+ str(num)
        filenames = os.listdir(path)
        ds = DictSave(filenames)
        ds.image_input(ds.filenames,num)
        ds.pickle_save(ds.all_arr,label_all_arr,num)
        print ("最终数组的大小:"+str(np.shape(label_all_arr))+str(np.shape(ds.all_arr)))



