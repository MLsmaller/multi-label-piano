import torch
import torchvision.transforms as transforms
import torch.nn as nn

import os 
import numpy as np
import cv2
import random
from PIL import Image
import shutil
import argparse

from model.resnet import resnet18
from config import cfg
from data.dataset import KeyDataset_select,random_crop
from utils.utils import get_key_nums1,cal_accuracy
from utils.helper import get_cam_img
from IPython import embed

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_lists", type=str, default=None, 
                        help="None for test img lists,true for test dataloader") 
    opt = parser.parse_args()
    return opt
opt=parser()

def select_img(save_path,select_num=10):
    with open(cfg.val_txt_path,'r') as f:
        lines=f.readlines()
    test_lines=random.sample(lines,select_num)
    for line in test_lines:
        line=line.strip().split()
        path=line[0]
        file_seq=os.path.basename(os.path.split(path)[0])
        if file_seq in cfg.crop_file_seq:continue
        print(path)
        shutil.copy(path,os.path.join(save_path,os.path.basename(os.path.split(path)[0])+
                                        '_'+os.path.basename(path)))    

def select_img1(select_num=10):
    with open(cfg.val_txt_path,'r') as f:
        lines=f.readlines()
    test_lines=random.sample(lines,select_num)
    return test_lines

fmap_block = list()
grad_block = list()  


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

def farward_hook(module, input, output):
    fmap_block.append(output)


def main(img_lists,val_dataloader):
    model = resnet18(pretrained=False, num_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(cfg.ckpt_path))
    model.eval()

    model.layer4[1].conv2.register_forward_hook(farward_hook)
    model.layer4[1].conv2.register_backward_hook(backward_hook)

    print('the ckpt path is {}'.format(cfg.ckpt_path))
    # test_lines=select_img1

    out_dir='./output'

    if opt.img_lists:
        for img_path in img_lists:
            # if not 'test_0013' in os.path.basename(img_path):continue
            print(img_path)
            # ori_img=cv2.imread(img_path,1)
            thresh=0.4
            img=Image.open(img_path)
            # file_seq=os.path.basename(os.path.split(img_path)[0])        
            # if file_seq in cfg.crop_file_seq:
            #     rect=cfg.EVALUATE_MAP[file_seq]['rect']
            #     img=img.crop((rect))  #--img.size-> w,h        
            
            img=img.resize((cfg.input_size))
            
            img=transforms.ToTensor()(img)
            if len(img.shape) !=3:
                img=img.unsqueeze(0)
                img=img.expand((3,img.shape[1:]))
            img=img.unsqueeze(0).to(device)

            output=model(img)
            func=nn.Sigmoid()
            prob=func(output)
            value,index=prob.topk(cfg.top_k)
            final_index=(index[value>thresh]).cpu().numpy()

            if len(final_index)>0:
                img_draw=get_key_nums1(img_path,out_dir,final_index)  
                cam_img = (get_cam_img(cfg.input_size,img_draw, output, final_index, grad_block,
                           fmap_block,cfg.num_classes)).astype(np.uint8)
                path_cam_img = os.path.join(out_dir, os.path.basename(img_path))
                cv2.imwrite(path_cam_img, cam_img)

            for index in final_index:
                print('the predict press key is {}'.format(cfg.labels[index]+1))
            print('\n')
    else:
        running_accuracy = 0
        running_precision = 0
        running_recall=0
        running_pos_keys=0
        running_total_keys=0
        running_true_keys=0
        fout=open(cfg.res_txt_path,'w')
        for batch_i,(paths,imgs,targets) in enumerate(val_dataloader):
            imgs=imgs.to(device)
            targets=targets.to(device)
            outputs=model(imgs)
            pressed_keys=[]
            correct,pos_keys,recall,true_keys,TN,total_keys,pressed_keys=cal_accuracy(
                                 outputs,targets,cfg.top_k,cfg.prob_thresh,pressed_keys)
            
            running_accuracy+=(correct+TN)
            running_total_keys+=total_keys
            running_precision += correct
            running_pos_keys+=pos_keys
            running_recall += recall
            running_true_keys+=true_keys 

            epoch_acc= running_accuracy/running_total_keys
            epoch_prec = running_precision / running_pos_keys
            epoch_recall = running_recall / running_true_keys   
            F = 2.0*epoch_recall*epoch_prec/(epoch_recall+epoch_prec+1e-6)
            data='Acc: {:.4f}\tPrec: {:.4f}\tRecall: {:.4f}\t Fscore: {:.4f}'.format(
                  epoch_acc,epoch_prec,epoch_recall,F)
            print(data)

            for i,path in enumerate(paths):
                fout.write('{} '.format(path))
                for key in pressed_keys[i]:
                    fout.write('{} '.format(cfg.labels[key]+1))
                fout.write('\n')
            print('one batch is down')
            # embed()
            fout.close()
            break

if __name__=='__main__':
    save_path='./test_imgs'
    img_lists=[os.path.join(save_path,x) for x in os.listdir(save_path)
               if x.endswith('.jpg')]
    img_lists.sort()    

    val_set = KeyDataset_select(cfg.val_txt_path, img_size=cfg.input_size,phase='val')
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )    
    # select_img(save_path)
    main(img_lists,val_dataloader)

