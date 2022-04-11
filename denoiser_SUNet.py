#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 19:00:30 2020

@author: 
"""
import sys
sys.path.append(r"D:/machao/python/")  # matlab调用本地文件
import torch
from SUNet.model.SUNet import SUNet_model

import numpy as np
import array

# other setting
import yaml  # setting
import argparse  # parameter
import os  # content1
# import sys
# sys.path.append(r"C:/ProgramData/Anaconda3/envs/mcc/Lib/site-packages/natsort/")
# from natsort import natsorted  # content2
from glob import glob  # content3
from PIL import Image
from collections import OrderedDict  # OrderedDict
import math  # process type
import cv2  # for output
from skimage import img_as_ubyte  # change restored pixel
from skimage import img_as_float
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def overlapped_square(timg, kernel=256, stride=128):
    patch_images = []
    b, c, h, w = timg.size()
    # 321, 481
    X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
    img = torch.zeros(1, 1, X, X).type_as(timg)  # 3, h, w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
    patch = patch.contiguous().view(b, c, -1, kernel, kernel)  # B, C, #patches, K, K
    patch = patch.permute(2, 0, 1, 4, 3)  # patches, B, C, K, K

    for each in range(len(patch)):
        patch_images.append(patch[each])

    return patch_images, mask, X

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint2(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def denoise124(noisy, sigma_hat):

    mwcnn_path = 'D:/machao/python/SUNet/checkpoints/Denoising/models/'
    imsize = int(np.sqrt(len(noisy)))
    # imsize = 512

    if sigma_hat > 500:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 300:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 150:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 125:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 100:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 95:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 90:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 85:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 80:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 75:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 70:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 65:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 60:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 55:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 50:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 45:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 40:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 35:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 30:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 25:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 20:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 15:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 10:
        path = mwcnn_path + 'model_bestPSNR.pth'
    elif sigma_hat > 5:
        path = mwcnn_path + 'model_bestPSNR.pth'
    else:
        path = mwcnn_path + 'model_bestPSNR.pth'

    if sigma_hat > 100:
        path = mwcnn_path + 'MWCNN_500_1000.pth'
    else:
        path = mwcnn_path + 'MWCNN_90_100.pth'
    path = mwcnn_path + 'SUNet_50_60.pth'
    # 仿demo
    with open('D:/machao/python/SUNet/training.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    parser = argparse.ArgumentParser(description='Demo Image Restoration')
    parser.add_argument('--input_dir', default='D:/machao/python/Training_Images/Training_data/', type=str,
                        help='Input images')
    parser.add_argument('--window_size', default=8, type=int, help='window size')
    parser.add_argument('--size', default=256, type=int, help='model image patch size')
    parser.add_argument('--stride', default=128, type=int, help='reconstruction stride')
    parser.add_argument('--result_dir', default='./demo_results/', type=str, help='Directory for results')
    parser.add_argument('--weights', default=path, type=str, help='Path to weights')
    args = parser.parse_args()
    inp_dir = args.input_dir  # 输入地址，需更改成matlab输入
    out_dir = args.result_dir  # 输出地址，需更改成matlab输出
    os.makedirs(out_dir, exist_ok=True)  # 创建输入图像目录，无需这么多，只需要matlab导入单张图片
    # files = natsorted(glob(os.path.join(inp_dir, '*.jpg'))
    #                   + glob(os.path.join(inp_dir, '*.JPG'))
    #                   + glob(os.path.join(inp_dir, '*.png'))
    #                   + glob(os.path.join(inp_dir, '*.PNG')))
    # if len(files) == 0:
    #     raise Exception(f"No files found at {inp_dir}")

    # 导入模型
    net = SUNet_model(opt)
    net.cuda(device=1)
    load_checkpoint2(net, args.weights)
    net.eval()
    print('restoring images......')
    stride = args.stride
    model_img = args.size

    noisy = np.array(noisy)
    noisy = Image.fromarray(noisy).convert('L')

    # input_ = torch.from_numpy(np.array(noisy)).float() / 255
    # input_ = input_.permute(2, 0, 1).unsqueeze(0).cuda(device=1)

    noisy = np.array(noisy)
    noisy = torch.from_numpy(noisy)
    noisy = noisy.cuda(device=1)
    noisy = noisy.float() /255
    noisy = noisy.unsqueeze(2)
    noisy = noisy.unsqueeze(0).permute(0, 3, 1, 2)
    noisy = torch.reshape(noisy, (1, 1, imsize, imsize))
    input_ = noisy

    # # PIL_noisy = noisy.ToPILImage()
    # input_ = TF.to_tensor(noisy).unsqueeze(0).cuda(device=1)

    with torch.no_grad():
        # pad to multiple of 256
        square_input_, mask, max_wh = overlapped_square(input_.cuda(device=1), kernel=model_img, stride=stride)  #
        output_patch = torch.zeros(square_input_[0].shape).type_as(square_input_[0])

        for i, data in enumerate(square_input_):
            restored = net(square_input_[i])
            if i == 0:
                output_patch += restored
            else:
                output_patch = torch.cat([output_patch, restored], dim=0)

        B, C, PH, PW = output_patch.shape
        weight = torch.ones(B, C, PH, PH).type_as(output_patch)  # weight_mask

        patch = output_patch.contiguous().view(B, C, -1, model_img*model_img)
        patch = patch.permute(2, 1, 3, 0)  # B, C, K*K, #patches
        patch = patch.contiguous().view(1, C*model_img*model_img, -1)

        weight_mask = weight.contiguous().view(B, C, -1, model_img * model_img)
        weight_mask = weight_mask.permute(2, 1, 3, 0)  # B, C, K*K, #patches
        weight_mask = weight_mask.contiguous().view(1, C * model_img * model_img, -1)

        restored = F.fold(patch, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
        we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
        restored /= we_mk

        restored = torch.masked_select(restored, mask.bool()).reshape(input_.shape)  # 根据掩码张量mask中的二元值，将取值返回到一个新的1D张量
        restored = torch.clamp(restored, 0, 1)  # 限定范围至0-1

    restored = restored.permute(0, 2, 3, 1)

    # output
    restored_out = restored.cpu().detach().numpy()
    restored_out = img_as_ubyte(restored_out[0])  # 像素值灰度转到0-255整型
    f = os.path.splitext(os.path.split("_")[-1])[0]
    save_img((os.path.join(out_dir, f + '.png')), restored_out)

    # return
    restored_hat = restored.double() * 255
    restored_hat = restored_hat.cpu().detach().numpy()
    x_hat = (restored_hat[0]).reshape(imsize * imsize, 1)

    # x_hat = np.dot(x_hat[..., :3], [0.299, 0.587, 0.114])
    # restored = img_as_float(restored[0])
    # x_hat = restored[0].double()
    # x_hat = x_hat * 255
    # x_hat = torch.reshape(x_hat, (imsize * imsize, 3))
    # x_hat = x_hat.cpu().numpy()
    #
    x_hat = array.array('d', x_hat)
    return x_hat


#
# noisy = Image.open("D:/machao/python/Training_Images/Training_data_ori/1.png")
# denoise124(noisy, 100)