
import pickle
from PIL import Image
import numpy as np
import os
import matplotlib.image as plimg
import h5py

class DictSave(object):
    def __init__(self,filenames):
        self.filenames = filenames

    def image_input(self,filenames):
        i = 0
        self.all_arr = [[0 for m in range(150528)] for n in range(len(filenames))]
        #self.label_all_arr = [[0] for k in range(len(filenames))]
        for filename in filenames:
            j = 0
            self.arr  = self.read_file(filename)
            self.all_arr[i][j:len(self.arr)] = self.arr
            #self.label_all_arr[i][j:len(self.label_arr)] = self.label_arr
            i = i+1

    def read_file(self,filename):
        im = Image.open(os.path.join("image_test1",filename),'r')#打开一个图像
        #site=filename.find('.')
        #label_arr = np.array([int(filename[site-2:site])])
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
        #return arr
    def pickle_save(self,arr,label_arr):
        print ("正在存储")

        # 构造字典,所有的图像数据都在arr数组里,这里只存图像数据,没有存label
        test_batch_1= {'data': arr,'labels': label_arr}
        f = open('test_batch_1', 'wb')

        pickle.dump(test_batch_1, f)#把字典存到文本中去
        f.close()
        print ("存储完毕")
if __name__ == "__main__":
    labelmix = 'testlabel' + str(1) + '.mat'
    labelin = h5py.File(labelmix)
    label_all_arr = np.transpose(labelin['bwlabel'])
    path = "image_test1"
    filenames = os.listdir(path)

    ds = DictSave(filenames)
    ds.image_input(ds.filenames)
    ds.pickle_save(ds.all_arr,label_all_arr)
    print ("最终数组的大小:"+str(np.shape(label_all_arr))+str(np.shape(ds.all_arr)))