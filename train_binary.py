from __future__ import division

from data.binary_dataset import Binary_dataset, Time_dataset
from model.resnet import ResNet_112_32, ResNet_Time
from utils.utils import binary_accuracy
from utils.logger import Logger
from config import cfg

import copy
import numpy as np
import os
import cv2
import sys
import time
import datetime
import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from IPython import embed

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')    
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
    parser.add_argument('--lr_decay_in_epoch', type=int, default=20, help='multiply by a gamma every lr_decay_in_epoch iterations')    
    parser.add_argument('--type', type=str, default='white', help='training type')
    parser.add_argument('--Data_type', type=str, default=None,
                        help='whether train the dataset with position information')
    parser.add_argument('--with_Time', type=str, default=None,
                        help='whether train the dataset with Time sequential')                        
    opt = parser.parse_args()
    return opt

opt=parser()
print(opt)
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
log_dir=os.path.join('logs',TIMESTAMP)
os.makedirs(log_dir, exist_ok=True)
logger = Logger(log_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get data configuration
train_path = cfg.w_train_binary_txt_path
val_path = cfg.w_val_binary_txt_path
# class_names = cfg.class_names

k = cfg.Consecutive_frames
# Get dataloader
if opt.with_Time is None:
    train_set = Binary_dataset(train_path, img_size=cfg.binary_input_size,
                                Data_type=opt.Data_type)
    val_set = Binary_dataset(val_path, img_size=cfg.binary_input_size,
                                Data_type=opt.Data_type, phase='val')
else:
    train_set = Time_dataset(train_path, img_size=cfg.binary_input_size,
                                Data_type=opt.Data_type, k=k)
    val_set = Time_dataset(val_path, img_size=cfg.binary_input_size,
                                Data_type=opt.Data_type, phase='val', k=k)

train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True
)

data_loader={'train':train_dataloader,'val':val_dataloader}
dataset_sizes = {'train':len(train_set),'val':len(val_set)}

print('the dataset_sizes is {}'.format(dataset_sizes))

# Initiate model
if opt.with_Time is None:
    model = ResNet_112_32(Data_type=opt.Data_type).to(device)
else:
    model = ResNet_Time(Data_type=opt.Data_type, k=cfg.Consecutive_frames).to(device)
    
alpha = cfg.ALPHA[opt.type]
#----正负样本的权重样本--
weights = torch.Tensor([1.0,alpha]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
#---位置区域的权重，中间按键是最多的，两边少
pos_weights = torch.Tensor(cfg.pos_wieghts).to(device)
criterion_pos = nn.CrossEntropyLoss(weight=pos_weights)


print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
# label_weight=torch.from_numpy(cfg.loss_weight)
# criterion = nn.CrossEntropyLoss(weight=label_weight)  


def train(model):
    optimizer = optim.Adam(model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)


    # metrics=cfg.metrics
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    txt_log=os.path.join(log_dir,'log.txt')
    f_out = open(txt_log, 'w')
    f_out.write('Data_type:{}\twith_Time:{}\tlr_decay_in_epoch:{}\nlr:{}\n'.format(
                opt.Data_type, opt.with_Time, opt.lr_decay_in_epoch,opt.lr))
    # f_out.write(param)
    for epoch in range(opt.epochs):
        print('Epoch {}/{}'.format(epoch, opt.epochs - 1))
        print('-' * 10)        
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   
            start_time = time.time()
            running_loss = 0.0
            running_correct = 0
            running_pos_correct = 0
            # embed()
            for batch_i, (imgs, targets) in enumerate(data_loader[phase]):
                batches_done = len(data_loader[phase]) * epoch + batch_i
                # test_img=np.array(imgs[0]*255,dtype=np.uint8).transpose((1,2,0))
                # opencv_img=cv2.cvtColor(test_img,cv2.COLOR_RGB2BGR)
                # cv2.imwrite('./test_img.jpg',opencv_img)
                # index=torch.where(targets[0]==1)[0]
                # print(targets[0])
                # for i in range(len(imgs)):
                #     a = imgs[i][0]
                #     test_img=np.array(a*255,dtype=np.uint8).transpose((1,2,0))
                #     opencv_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite('./{}.jpg'.format(i), opencv_img)
                
                if isinstance(imgs, list):
                    img_lists = []
                    for img in imgs:
                        img = img.to(device)
                        img_lists.append(img)
                    imgs=img_lists[:]
                else:
                    imgs = imgs.to(device)

                # imgs = imgs.to(device)
                #---由于数据集大小限制的原因，不是每个batch刚好都是取batch_size大小的,一般在最后一个iteration，不能够整除
                if isinstance(targets, list):
                    press_label = targets[0].to(device)
                    pos_label = targets[1].to(device)
                    batch_shape = targets[0].shape[0]
                else:
                    press_label = targets.to(device)
                    batch_shape = targets.shape[0]
                    
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    if opt.Data_type:
                        #----加上一些权重把
                        press_loss = criterion(outputs[0], press_label)
                        pos_loss = criterion_pos(outputs[1], pos_label)
                        # print('press_loss:{}\tpos_loss:{}\t'.format(press_loss, pos_loss))
                        loss = press_loss + pos_loss
                        correct = binary_accuracy(outputs[0], press_label)
                        pos_correct = binary_accuracy(outputs[1], pos_label)
                    else:
                        loss = criterion(outputs, press_label)
                        correct = binary_accuracy(outputs, press_label)
                        pos_correct = 0

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.item() * batch_shape
                    running_correct += correct
                    running_pos_correct += pos_correct
                    correct_metrics = float((correct.cpu().numpy()) / batch_shape)

                    if phase == 'train':
                        train_metrics = [
                            ("train_loss", loss.item()),
                            ("train_acc", correct_metrics)
                        ]
                        logger.list_of_scalars_summary(train_metrics, batches_done)
                        # logger.list_of_scalars_summary(("val_loss", loss.item()), batches_done)

                    else:
                        val_metrics = [
                            ("val_loss", loss.item()),
                            ("val_acc", correct_metrics)
                        ]
                        logger.list_of_scalars_summary(val_metrics, batches_done)                        
                        # logger.list_of_scalars_summary(("val_loss", loss.item()), batches_done)                    
                    

            if phase == 'train':
                scheduler.step()       

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            epoch_pos_correct = running_pos_correct / dataset_sizes[phase]

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch=epoch

            data = '{}\tLoss: {:.4f}\tAcc: {:.4f}\tPos_Acc: {:.4f}'.format(phase,
                    epoch_loss, epoch_acc, epoch_pos_correct)
            print(data)
            f_out.write('epoch: {}:\n'.format(epoch))
            f_out.write(data)
            f_out.write('\n')
            f_out.write('\n')
            
            end_time = time.time()
            print('current epoch time cost {:.2f} minutes'.format((end_time-start_time)/60))
            print('\n')

    print('Epoch {} has the best Acc is {} '.format(best_epoch, best_acc))

    f_out.close()
    if opt.with_Time:
        if opt.Data_type:
            #--包含位置和时间信息
            torch.save(best_model_wts, "checkpoints/time_with_pos_keys_epoch_{}_Acc_{:.3f}.pth".format(best_epoch, best_acc))
        else:
            #--只包含时间信息
            torch.save(best_model_wts, "checkpoints/time_keys_epoch_{}_Acc_{:.3f}.pth".format(best_epoch, best_acc))
    elif opt.Data_type:
        #---包含位置信息不包含时间信息
        torch.save(best_model_wts, "checkpoints/with_pos_keys_epoch_{}_Acc_{:.3f}.pth".format(best_epoch,best_acc))   
    else:
        #---什么都不包含
        torch.save(best_model_wts, "checkpoints/binary_keys_epoch_{}_Acc_{:.3f}.pth".format(best_epoch,best_acc))   

if __name__ == "__main__":    
    train(model)
        


