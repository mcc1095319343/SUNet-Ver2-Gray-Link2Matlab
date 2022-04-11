import os

import torch
import yaml

from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import utils
import numpy as np
import random
from data_RGB import get_training_data, get_validation_data

from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model

from SUNet.data_utils import *  # for noisy

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']
train_hr_img_list = sorted(load_file_list(path=train_dir, regx='.*.png', printable=False))  # add
test_img_list = sorted(load_file_list(path=val_dir, regx='.*.tif', printable=False))
n_test = len(test_img_list)

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr)

## Scheduler (Strategy)
# warmup_epochs = 3
# scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
#                                                         eta_min=float(OPT['LR_MIN']))
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
# scheduler.step()

## Resume (Continue training by a pretrained model)
# if Train['RESUME']:
#     path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
#     utils.load_checkpoint(model_restored, path_chk_rest)
#     start_epoch = utils.load_start_epoch(path_chk_rest) + 1
#     utils.load_optim(optimizer, path_chk_rest)
#
#     for i in range(1, start_epoch):
#         scheduler.step()
#     new_lr = scheduler.get_lr()[0]
#     print('------------------------------------------------------------------')
#     print("==> Resuming Training with learning rate:", new_lr)
#     print('------------------------------------------------------------------')

## Loss
# L1_loss = nn.L1Loss()
mse_criterion = nn.MSELoss()
mse_criterion.cuda()

## DataLoaders
print('==> Loading datasets')
# train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
# train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
#                           shuffle=True, num_workers=0, drop_last=False)
# val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
# val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
#                         drop_last=False)

# Show the training configuration
# print(f'''==> Training details:
# ------------------------------------------------------------------
#     Restoration mode:   {mode}
#     Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
#     Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
#     Model parameters:   {p_number}
#     Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
#     Batch sizes:        {OPT['BATCH']}
#     Learning rate:      {OPT['LR_INITIAL']}
#     GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
# best_ssim = 0
best_epoch_psnr = 0
# best_epoch_ssim = 0
total_start_time = time.time()
img_size = 256  # add 裁剪大小



for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    # train_id = 1

    model_restored.train()
    # random.shuffle(train_hr_img_list)
    epoch_time = time.time()  # 计时
    total_mse_loss1, total_mse_loss2, n_iter = 0, 0, 0
    if epoch > 10 and (epoch % 10 == 0):
        lr = new_lr / (2 ** ((epoch - 10) // 10))
        log = " ** new learning rate: %f " % (new_lr / (2 ** ((epoch - 10) // 10)))
        print(log)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr                            # 每过10个epoch更新一次学习率（epoch>20），学习率缩减三倍

    for idx in range(0, len(train_hr_img_list)//OPT['BATCH']*OPT['BATCH'], OPT['BATCH']):      # epoch中batchsize小训练循环 add
        step_time = time.time()
        train_hr_imgs = threading_data(train_hr_img_list[idx:idx + OPT['BATCH']], fn=get_imgs_fn,path=train_dir)  # 读取图片数据
        flag = 0
        for num in range(0, OPT['BATCH']):
            b_width, b_height = train_hr_imgs[num].shape
            if (b_width < 256) | (b_height < 256):
                flag = 1
        if flag == 1:
            continue
        train_batch_data = threading_data(train_hr_imgs,fn=crop_sub_imgs_fn,sl=50,sh=60,w=img_size,h=img_size,is_random=True)  # 图片数据截取与加噪
        [b_imgs_384,n_imgs_384] = np.split(train_batch_data, 2, axis=1)  # 噪声图与原图分离
        real_img = Variable(torch.from_numpy(b_imgs_384))  # 将numpy转为tensor格式
        z = Variable(torch.from_numpy(n_imgs_384))
        if torch.cuda.is_available():
            real_img = real_img.cuda()   # 数据入显卡
            z = z.cuda()

        model_restored.zero_grad()    # 网络梯度清零，每次更新前需要清零操作
        fake_img = model_restored(z)  # 反馈网络结果

        # g_loss = L1_loss(fake_img, real_img)
        g_loss = mse_criterion(fake_img, real_img)
        g_loss.backward()  #反馈loss计算梯度
        optimizer.step()   #更新参数
        print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f" % (epoch, OPT['EPOCHS'], n_iter, time.time() - step_time, g_loss.item()))#打印loss
        total_mse_loss1 += g_loss.item()
        n_iter += 1
    log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f\r\n" % (epoch, OPT['EPOCHS'], time.time() - epoch_time, total_mse_loss1 / n_iter)  # 打印loss
    print(log)
    model_restored.eval()

    # test
    psnr = np.zeros(n_test)
    for idx in range(n_test):
        t_img, n_img = get_test_img(val_dir + test_img_list[idx], 10)
        with torch.no_grad():
            denoise = model_restored(n_img)
            psnr[idx] = PSNR(t_img.cpu().numpy(), denoise.cpu().numpy())
            log = test_img_list[idx] + ": Epoch[%d] psnr = %.4f" % (epoch, psnr[idx])
            print(log)
    mean_psnr = np.mean(psnr)

    if mean_psnr > best_psnr:
        best_psnr = mean_psnr
        best_epoch_psnr = epoch
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "SUNet_50_60.pth"))
    print("(epoch %d PSNR: [%.4f] --- best_epoch %d Best_PSNR [%.4f])" % (epoch, mean_psnr, best_epoch_psnr, best_psnr))
    writer.add_scalar('val/PSNR', mean_psnr, epoch)
    # print("mean_psnr = [%.4f]\r\n" % mean_psnr)
    # torch.save(model_restored.state_dict(), 'MWCNN_500_1000.pth')  # 保存模型
writer.close()


