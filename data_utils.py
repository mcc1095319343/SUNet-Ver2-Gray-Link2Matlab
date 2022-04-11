from os import listdir
from os.path import join
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import scipy.io as sio
import scipy
import threading
import torch
import os
import re
import math
from torch.autograd import Variable
baqb = [[ float('-inf'), -2.94819004, -2.47266615, -2.13768889, -1.86853297,
        -1.63811907, -1.43318492, -1.24613825, -1.07218127, -0.90806337,
        -0.75146222, -0.60064523, -0.45426961, -0.31125639, -0.1707061 ,
        -0.03183875,  0.10605188,  0.24363598,  0.38158185,  0.52058993,
         0.66143046,  0.8049889 ,  0.9523254 ,  1.10475815,  1.26398743,
         1.4322922 ,  1.61286227,  1.81040497,  2.03236696,  2.29175046,
         2.61509222,  3.07560059,  float('inf')]]
baqy = [[-3.23498704, -2.6610698 , -2.28391173, -1.99110496, -1.74560356,
        -1.5302938 , -1.33576437, -1.15624157, -0.98790291, -0.82806891,
        -0.67477348, -0.52651642, -0.3821111 , -0.24058481, -0.10110978,
         0.03704781,  0.1745683 ,  0.31211352,  0.45036003,  0.59003401,
         0.73195155,  0.87706925,  1.02655255,  1.181874  ,  1.3449632 ,
         1.51844988,  1.70608529,  1.91353418,  2.15002663,  2.43233904,
         2.79677245,  3.35345352]]
baqy = np.array(baqy)
baqb = np.array(baqb)
         
def load_file_list(path=None, regx='\.npz', printable=True):
    if path == False:
        path = os.getcwd()
    file_list = os.listdir(path)
    return_list = []
    for idx, f in enumerate(file_list):
        if re.search(regx, f):
            return_list.append(f)
    # return_list.sort()
    if printable:
        print('Match file list = %s' % return_list)
        print('Number of files = %d' % len(return_list))
    return return_list
def get_imgs_fn(file_name, path):
    im = scipy.misc.imread(path + file_name, mode='L')
    return im

def threading_data(data=None, fn=None, **kwargs):            #线程函数，用于读取数据
    ## plot function info
    # for name, value in kwargs.items():
    #     print('{0} = {1}'.format(name, value))
    # exit()
    # define function for threading
    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    ## start multi-threaded reading.
    results = [None] * len(data) ## preallocate result list
    threads = []
    for i in range(len(data)):
        t = threading.Thread(
                        name='threading_and_return',
                        target=apply_fn,
                        args=(results, i, data[i], kwargs)
                        )
        t.start()
        threads.append(t)

    ## <Milo> wait for all threads to complete
    for t in threads:
        t.join()

    return np.asarray(results)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])
def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1, channel_index=2):
    """Randomly or centrally crop an image.

    Parameters
    ----------
    x : numpy array
        An image with dimension of [row, col, channel] (default).
    wrg : float
        Size of weight.
    hrg : float
        Size of height.
    is_random : boolean, default False
        If True, randomly crop, else central crop.
    row_index, col_index, channel_index : int
        Index of row, col and channel, default (0, 1, 2), for theano (1, 2, 0).
    """
    h, w = x.shape[row_index], x.shape[col_index]
    assert (h >= hrg) and (w >= wrg), "The size of cropping should smaller than the original image"
    if is_random:
        h_offset = int(np.random.uniform(0, h-hrg) -1)
        w_offset = int(np.random.uniform(0, w-wrg) -1)
        # print(h_offset, w_offset, x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset].shape)
        return x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset]
    else:   # central crop
        h_offset = int(np.floor((h - hrg)/2.))
        w_offset = int(np.floor((w - wrg)/2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset: h_end, w_offset: w_end]
        # old implementation
        # h_offset = (h - hrg)/2
        # w_offset = (w - wrg)/2
        # # print(x[h_offset: h-h_offset ,w_offset: w-w_offset].shape)
        # return x[h_offset: h-h_offset ,w_offset: w-w_offset]
        # central crop
def data_aug(img, mode=0):  #数据旋转翻转增强8倍
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)  #矩阵上下翻转
    elif mode == 2:
        return np.rot90(img)   #逆时针旋转90
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def crop_sub_imgs_fn(x,sl=0,sh=5,w=192,h=192,is_random=True):  
    x = crop(x, wrg=w, hrg=h, is_random=is_random)   #截取数据             
    x = data_aug(x, mode=np.random.randint(0, 8)) #使用数据增强
    if sl == sh:
        sigma = sl
    else:
        sigma = np.random.randint(sl,sh-1)+np.random.rand(1)
    x1 = x.astype(np.float64) + np.random.normal(0,sigma,[w,h]).round()  #产生噪声
    x = x.reshape([1,w,h])
    x = x / 255. 
    #x = x - 1.
    x = x.astype(np.float32)
    x1 = x1.reshape([1,w,h])
    x1 = x1 / 255. 
    #x1 = x1 - 1.
    x1 = x1.astype(np.float32)
    x = np.concatenate((x,x1),axis=0)   #原图与噪声图合到一起返回（由于线程函数的限制，只能返回单值，在主程序中应用时将两者再分开即可）
    return x
  
def get_test_img(path,sigma):     
    
    test_img = scipy.misc.imread(path, mode='L') 
    w, h = test_img.shape
    noisy_img = test_img.astype(np.float64) + np.random.normal(0,sigma,[w,h]).round()  #产生噪声
    test_img = test_img.reshape([1,1,w,h])
    test_img = test_img / 255.
    test_img = test_img.astype(np.float32)
    noisy_img = noisy_img.reshape([1,1,w,h])
    noisy_img = noisy_img / 255.
    noisy_img = noisy_img.astype(np.float32)
    test_img = torch.from_numpy(test_img)  #将numpy转为tensor格式
    noisy_img = torch.from_numpy(noisy_img)
    if torch.cuda.is_available():
        test_img = test_img.cuda()   #数据入显卡
        noisy_img = noisy_img.cuda()
    return test_img, noisy_img
    
def PSNR(img1, img2):
    D = np.array(img1 - img2, dtype=np.float64)
    D[:, :] = D[:, :]**2
    RMSE = D.sum()/img1.size
    psnr = 10*math.log10(float(1.**2)/RMSE)
    return psnr

    
def get_imgs_fn1(file_name, path):#pytorch 默认输入格式为B C H W
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    #return scipy.misc.imread(path + file_name, mode='L')
    data1 = sio.loadmat(path + file_name)
    data1 = data1['data'] 
    if len(data1.shape)>2:
        data1 = data1.reshape((data1.shape[2],data1.shape[0],data1.shape[1]))
    else:
        data1 = data1.reshape((1,data1.shape[0],data1.shape[1]))
    data1 = data1.astype(float)
    return data1

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
