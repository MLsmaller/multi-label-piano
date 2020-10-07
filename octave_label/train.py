from __future__ import division

from data.dataset import KeyDataset,KeyDataset_select
from model.resnet import resnet18
from utils.utils import modify_last_layer_lr,cal_accuracy
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
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from IPython import embed

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')    
    parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
    parser.add_argument('--lr_decay_in_epoch', type=int, default=20, help='multiply by a gamma every lr_decay_in_epoch iterations')    
    parser.add_argument('--type', type=str, default='key', help='training type')     
    opt = parser.parse_args()
    return opt

opt=parser()
print(opt)
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
log_dir=os.path.join('logs',TIMESTAMP)
os.makedirs(log_dir, exist_ok=True)
# logger = Logger(log_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs("output", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Get data configuration
train_path = cfg.train_txt_path
val_path=cfg.val_txt_path
# class_names = cfg.class_names

# Get dataloader

train_set = KeyDataset_select(train_path, img_size=cfg.input_size)
train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    pin_memory=True
)

val_set = KeyDataset_select(val_path, img_size=cfg.input_size,phase='val')
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

model = resnet18(pretrained=True,num_classes=cfg.num_classes).to(device)
criterion = nn.BCELoss().to(device)

# criterion = nn.MultiLabelSoftMarginLoss().to(device)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
label_weight=torch.from_numpy(cfg.loss_weight)

# criterion = nn.CrossEntropyLoss(weight=label_weight)  


def train(model):
    # modify learning rate of last layer
    finetune_params = modify_last_layer_lr(model.named_parameters(), 
                                            opt.lr, cfg.lr_mult_w, cfg.lr_mult_b)

    optimizer = optim.SGD(finetune_params, 
                          opt.lr, 
                          momentum=opt.momentum, 
                          weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)


    # metrics=cfg.metrics
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    txt_log=os.path.join(log_dir,'log.txt')
    f_out=open(txt_log,'w')    
    param='prob_thresh: {}\ttop_k: {}\n'.format(cfg.prob_thresh,cfg.top_k)
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

                # test_img=np.array(imgs[0]*255,dtype=np.uint8).transpose((1,2,0))
                # opencv_img=cv2.cvtColor(test_img,cv2.COLOR_RGB2BGR)
                # cv2.imwrite('./test_img.jpg',opencv_img)
                # index=torch.where(targets[0]==1)[0]
                # print(targets[0])
                # embed()

                imgs = imgs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs).type(torch.double)  #--batch,88
                    # print(outputs.shape)

                    correct,pos_keys,recall,true_keys,TN,total_keys,pressed_keys=cal_accuracy(outputs,
                                                                    targets,cfg.top_k,cfg.prob_thresh,pressed_keys)
                    
                    func=nn.Sigmoid()
                    loss=criterion(func(outputs),targets)
                    # print(loss)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * imgs.size(0)
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

            if phase == 'val' and F > best_acc:
                best_acc = F
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch=epoch

            data='{}\tLoss: {:.4f}\tAcc: {:.4f}\tPrec: {:.4f}\tRecall: {:.4f}\t Fscore: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc,epoch_prec,epoch_recall,F)
            print(data)
            f_out.write('epoch: {}:\n'.format(epoch))
            f_out.write(data)
            f_out.write('\n')
            f_out.write('\n')
            
            end_time = time.time()
            print('current epoch time cost {:.2f} minutes'.format((end_time-start_time)/60))
            print('\n')

    print('Epoch {} has the best Fscore is {} '.format(best_epoch,F))
    torch.save(best_model_wts, "checkpoints/keys_epoch_{}_Fscore_{:.3f}.pth".format(best_epoch,best_acc))   

if __name__ == "__main__":    
    train(model)
    # criterion1 = nn.CrossEntropyLoss() 
    # criterion2 = nn.MultiLabelSoftMarginLoss()
    # output=torch.Tensor([[0,0.2,0.6,0.5]])
    # label=torch.Tensor([1]).type(torch.LongTensor)
    # label1=torch.Tensor([0,0,1,1])

    # softmax_output=nn.Softmax(dim=1)
    # output1=softmax_output(output)
    # y=criterion1(output,label)
    # y1=criterion2(output,label1)

    # test_output=criterion2(torch.Tensor([[0,0.2,0.6,0.5]]),label1)


    # loss = nn.MultiLabelMarginLoss()
    # x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
    # # for target y, only consider labels 3 and 0, not after label -1
    # y = torch.LongTensor([[3, 0, 0, 0]])
    # final=loss(x, y)



    #         # ----------------
    #         #   Log progress
    #         # ----------------

    #         log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

    #         metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

    #         # Log metrics at each YOLO layer
    #         for i, metric in enumerate(metrics):
    #             formats = {m: "%.6f" for m in metrics}
    #             formats["grid_size"] = "%2d"
    #             formats["cls_acc"] = "%.2f%%"
    #             #--model.yolo_layer即3个fm对应的层
    #             row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
    #             metric_table += [[metric, *row_metrics]]
    #             #---metric_table：：
    #             #--[['Metrics', 'YOLO Layer 0', 'YOLO Layer 1', 'YOLO Layer 2'],
    #             #---['grid_size', '12', '24', '48']]

    #             # Tensorboard logging
    #             tensorboard_log = []
    #             #---分别计算出yolo中三个fm层对应的loss和准确率
    #             for j, yolo in enumerate(model.yolo_layers):
    #                 for name, metric in yolo.metrics.items():
    #                     if name != "grid_size":
    #                         tensorboard_log += [(f"{name}_{j+1}", metric)]
    #             tensorboard_log += [("loss", loss.item())]
    #             logger.list_of_scalars_summary(tensorboard_log, batches_done)

    #         log_str += AsciiTable(metric_table).table
    #         log_str += f"\nTotal loss {loss.item()}"

    #         # Determine approximate time left for epoch
    #         epoch_batches_left = len(dataloader) - (batch_i + 1)
    #         time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
    #         log_str += f"\n---- ETA {time_left}"

    #         if batches_done % len(dataloader)==0:
    #             f_out.write(log_str)
    #             f_out.write('\n')

    #         print(log_str)

    #         model.seen += imgs.size(0)

    #     print('epoch {} has take {:.3f} minute'.format(epoch,(time.time()-start_time)/60))

    #     if epoch % cfg.evaluation_interval == 0:
    #         print("\n---- Evaluating Model ----")
    #         # Evaluate the model on the validation set
    #         precision, recall, AP, f1, ap_class = evaluate(
    #             model,
    #             path=val_path,
    #             iou_thres=0.5,
    #             conf_thres=0.5,
    #             nms_thres=0.5,
    #             img_size=cfg.img_size,
    #             batch_size=8,
    #         )
    #         evaluation_metrics = [
    #             ("val_precision", precision.mean()),
    #             ("val_recall", recall.mean()),
    #             ("val_mAP", AP.mean()),
    #             ("val_f1", f1.mean()),
    #         ]
    #         logger.list_of_scalars_summary(evaluation_metrics, epoch)

    #         # Print class APs and mAP
    #         ap_table = [["Index", "Class name", "AP"]]
    #         for i, c in enumerate(ap_class):
    #             ap_table += [[c, class_names[c], "%.5f" % AP[i]]]

    
    #         print(AsciiTable(ap_table).table)
    #         mAP=AP.mean()
            
    #         print(f"---- mAP {mAP}")
    #         f_out.write('\n')
    #         f_out.write(f"---- mAP {mAP}")
    #         f_out.write(AsciiTable(ap_table).table)
    #         f_out.write('\n')

    #         if mAP>best_mAP:
    #             best_mAP=mAP
    #             best_model_wts=copy.deepcopy(model.state_dict())
    #             best_epoch=epoch

    # f_out.close()
    # print('Epoch {} has the best_map is {} '.format(best_epoch,best_mAP))
    # torch.save(best_model_wts, "checkpoints/yolov3_epoch_{}_mAP_{:.3f}.pth".format(best_epoch,best_mAP))            

