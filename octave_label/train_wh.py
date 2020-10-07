from __future__ import division

# from data.dataset import KeyDataset_select, TimeDataset_select
from data.wb_dataset import WBDataset
from model.resnet import resnet18
from utils.utils import cal_accuracy,VisAmtHelper
from utils.keyboard_helper.seghand import SegHand
from utils.logger import Logger
from config import cfg
from test_wh import test_videos

from PIL import Image
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
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')    
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
    parser.add_argument('--lr_decay_in_epoch', type=int, default=20, help='multiply by a gamma every lr_decay_in_epoch iterations')    
    parser.add_argument('--type', type=str, default='white', help='training type')
    parser.add_argument('--Data_type', type=str, default=None,
                        help='whether train the dataset with position information')
    parser.add_argument('--with_Time', type=str, default='a',
                        help='whether train the dataset with Time sequential')
    parser.add_argument('--mode', type=str, default='white',
                        help='train white or black')
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
train_path = cfg.octave_train_list
val_path = cfg.octave_val_list
# class_names = cfg.class_names

k = cfg.Consecutive_frames
# Get dataloader
if opt.with_Time is None:
    train_set = KeyDataset_select(train_path, img_size=cfg.octave_final_size
                                )
    val_set = KeyDataset_select(val_path, img_size=cfg.octave_final_size
                                , phase='val')
else:
    train_set = WBDataset(train_path, img_size=cfg.octave_final_size
                                , k=k)
    val_set = WBDataset(val_path, img_size=cfg.octave_final_size
                                , phase='val', k=k)

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
num_classes = 7 if opt.mode == 'white' else 5
# Initiate model
if opt.with_Time is None:
    model = resnet18(pretrained=False,num_classes=num_classes).to(device)
else:
    model = resnet18(pretrained=False, num_classes=num_classes,
                     k=k, phase='time').to(device)
    
#---位置区域的权重，中间按键是最多的，两边少
pos_weights = torch.Tensor(cfg.pos_wieghts).to(device)
criterion_pos = nn.CrossEntropyLoss(weight=pos_weights)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
label_weight = torch.Tensor([1, 1, 1, 2, 1, 1, 1]).double().to(device)
criterion = nn.BCELoss(weight=label_weight).to(device)

VisAmt = VisAmtHelper()
VisAmt.init_model_load()
hand_seg = SegHand()

def eval_video(model):  
    
    img_path1 = [os.path.join(cfg.Tencent_path, x) for x in os.listdir(cfg.Tencent_path)]
    img_path2 = [os.path.join(cfg.SightToSound_paper_path, x) for x in os.listdir(cfg.SightToSound_paper_path)]
    test_img_lists=[]
    test_img_lists.extend(img_path1)
    test_img_lists.extend(img_path2)
    test_img_lists.sort()
    average_Wscore = []
    for video_path in test_img_lists:
        file_seq = os.path.basename(video_path)
        # if not file_seq in ['level_1_no_02', 'level_2_no_02']: continue
        if not file_seq in ['level_2_no_02']: continue
        print('-' * 50)
        print('eval video: {}'.format(video_path))
        begin = time.time()
        w_F, note_wF = test_videos(model, hand_seg, VisAmt, video_path, opt)
        average_Wscore.append((w_F + note_wF) / 2)
        end = time.time()
        print('eval cost : {:.2f} minutes'.format((end - begin) / 60))
    return np.mean(average_Wscore)



def train(model):
    optimizer = optim.SGD(model.parameters(), 
                          lr=opt.lr, 
                          momentum=opt.momentum, 
                          weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)


    # metrics=cfg.metrics
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    txt_log=os.path.join(log_dir,'log.txt')
    f_out = open(txt_log, 'w')

    param = 'lr: {}\tprob_thresh: {}\ttop_k: {}\tlr_decay_in_epoch: {}\toctave_test_thresh: {}\n'.format(
            opt.lr, cfg.octave_prob_thresh, cfg.octave_top_k, opt.lr_decay_in_epoch, cfg.octave_test_thresh)
    
    best_Wscore = 0
    print(param)
    f_out.write(param)
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

            running_accuracy = 0
            running_precision = 0
            running_recall=0
            running_pos_keys=0
            running_total_keys=0
            running_true_keys=0
            pressed_keys=[]
            for batch_i, (img_paths,imgs, targets) in enumerate(data_loader[phase]):
                batches_done = len(data_loader[phase]) * epoch + batch_i
                # for i in range(len(imgs)):
                #     a = imgs[i][0]
                #     img_path = img_paths[i][0]
                #     print(img_path)
                #     test_img=np.array(a*255,dtype=np.uint8).transpose((1,2,0))
                #     opencv_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite('./{}.jpg'.format(i), opencv_img)
                if isinstance(imgs, list):
                    img_lists = []
                    for img in imgs:
                        img = img.to(device)
                        img_lists.append(img)
                    imgs = img_lists[:]
                else:
                    imgs = imgs.to(device)

                #---由于数据集大小限制的原因，不是每个batch刚好都是取batch_size大小的,一般在最后一个iteration，不能够整除
                if isinstance(targets, list):
                    press_label = targets[0].to(device)
                    pos_label = targets[1].to(device)
                    batch_shape = targets[0].shape[0]
                else:
                    press_label = targets.to(device)
                    batch_shape = targets.shape[0]

                # targets = targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs).type(torch.double)  #--batch,12

                    correct,pos_keys,recall,true_keys,TN,total_keys,pressed_keys=cal_accuracy(outputs,
                                        press_label,cfg.octave_top_k,cfg.octave_prob_thresh,pressed_keys,epoch)
                    
                    func = nn.Sigmoid()
                    loss=criterion(func(outputs),press_label)
                    # print(loss)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()           

                running_loss += loss.item() * press_label.size(0)
                running_accuracy+=(correct+TN)
                running_total_keys+=total_keys
                running_precision += correct
                running_pos_keys+=pos_keys
                running_recall += recall
                running_true_keys+=true_keys                

            if phase == 'train':
                scheduler.step()       

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc= running_accuracy/running_total_keys
            epoch_prec = running_precision / running_pos_keys
            epoch_recall = running_recall / running_true_keys   
            F = 2.0*epoch_recall*epoch_prec/(epoch_recall+epoch_prec+1e-6)
            # torch.save(model.state_dict(),'./checkpoints/epoch_{}.pth'.format(epoch))

            data='{}\tLoss: {:.4f}\tAcc: {:.4f}\tPrec: {:.4f}\tRecall: {:.4f}\t Fscore: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc,epoch_prec,epoch_recall,F)
            print(data)

            if phase == 'val':
                average_Wscore = eval_video(model)
                print("average_Wscore= : {}".format(average_Wscore))
            if phase == 'val' and average_Wscore > best_Wscore:
                best_acc = F
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_Wscore = average_Wscore
                

            f_out.write('epoch: {}:\n'.format(epoch))
            f_out.write(data)
            f_out.write('\n')
            f_out.write('\n')
            
            time_elapsed = time.time() - start_time
            print('current epoch time cost {:.2f} minutes'.format((time_elapsed)/60))
            print('\n')
    
    print('Epoch {} has the best Acc is {} '.format(best_epoch, best_acc))

    f_out.close()
    if opt.with_Time:
        if opt.Data_type:
            #--包含位置和时间信息
            torch.save(best_model_wts, "checkpoints/wb_{}_octave_time_with_pos_keys_epoch_{}_Acc_{:.3f}.pth".format(
                       opt.mode,best_epoch, best_acc))
        else:
            #--只包含时间信息
            torch.save(best_model_wts, "checkpoints/wb_{}_octave_time_keys_epoch_{}_Acc_{:.3f}.pth".format(
                       opt.mode, best_epoch, best_acc))
    elif opt.Data_type:
        #---包含位置信息不包含时间信息
        torch.save(best_model_wts, "checkpoints/wb_{}_octave_with_pos_keys_epoch_{}_Acc_{:.3f}.pth".format(
                   opt.mode,best_epoch,best_acc))   
    else:
        #---什么都不包含
        torch.save(best_model_wts, "checkpoints/wb_{}_octave_keys_epoch_{}_Acc_{:.3f}.pth".format(
                   opt.mode,best_epoch,best_acc))   

if __name__ == "__main__":
    train(model)
        


