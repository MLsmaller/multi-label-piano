
import os 
import torch.nn as nn
import torch
import cv2
import torchvision.transforms as transforms

import sys
from config import cfg
from PIL import Image
from IPython import embed

from .bwlabel import BwLabel
from .helper import *

def input_img(img_path):
    img=Image.open(img_path)      
    img=img.resize((cfg.input_size))
    img=transforms.ToTensor()(img)
    if len(img.shape) !=3:
        img=img.unsqueeze(0)
        img=img.expand((3,img.shape[1:]))
    img=img.unsqueeze(0)
    return img

def binary_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #----(batch,1)
    _, pred = output.topk(maxk, 1, True, True)
    #----(1,batch)
    pred = pred.t()
    #----(1,batch)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    total_correct=0
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        total_correct+=correct_k
        # res.append(correct_k.mul_(100.0 / batch_size))
    return total_correct

def cal_accuracy(outputs,targets,top_k,prob_thresh,pressed_keys):
    func=nn.Sigmoid()
    probs=func(outputs).cpu()
    batch_size=outputs.shape[0]
    correct=0
    pos_keys=0.0001  
    total_keys=0.0001   
    recall=0
    true_keys=0   #---实际按下的键
    TN=0  #---预测为负样本实际也为负样本

    for i in range(batch_size):
        prob=probs[i]
        target=targets[i].cpu()
        top_value, indexs = prob.topk(top_k)
        true_index=torch.where(target==1)[0]
        #---预测出来的结果中认为的正样本
        key_indexs=indexs[top_value>prob_thresh]
        # print('the pos index is {}'.format(key_indexs)) 
        pressed_keys.append(key_indexs)

        #--预测出来的结果中认为的负样本
        key_neg=indexs[~(top_value>prob_thresh)]
        # print('the target index is {}'.format(true_index))
        # print('\n')
        pos_keys+=len(key_indexs)
        true_keys+=len(true_index)
        total_keys+=len(indexs)
        # print(key_indexs)
        for index in key_neg:
            #--负样本不在标签中则预测正确
            if index not in true_index:
                TN+=1
        for index in key_indexs:
            if index in true_index:
                correct+=1
        for index in true_index:
            if index in key_indexs:
                recall+=1
        # print(key_indexs)
        # print(true_index)
        # print('correct: {}\trecall: {}\n'.format(correct,recall))
    
    # print('the label is {}'.format(true_index))
    # print('correct is {} recall is {}'.format(correct,recall))
    # print('total correct is {} total recall is {}'.format(pos_keys,true_keys))
    # print('the acc is {:.2f} rec_acc is {:.2f}'.format(correct/pos_keys,recall/true_keys))
    # print('\n')
    # embed()
    return correct,pos_keys,recall,true_keys,TN,total_keys,pressed_keys
    
def modify_last_layer_lr(named_params, base_lr, lr_mult_w, lr_mult_b):
    params = list()
    for name, param in named_params: 
        if 'bias' in name:
            if 'fc' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_b, 'weight_decay': 0}]
            else:
                params += [{'params':param, 'lr': base_lr * 2, 'weight_decay': 0}]
        else:
            if 'fc' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_w}]
            else:
                params += [{'params':param, 'lr': base_lr * 1}]
    return params

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def new_save_dir(root,file_mark):
    base_img_dir = os.path.join(root,file_mark,'base_img')

    white_key_img_dir=os.path.join(root,file_mark,'white_key')
    black_key_img_dir=os.path.join(root,file_mark,'black_key')

    #--存储该键被按下的帧的前后几帧(特定选取的负样本)
    neg_white_key_img_dir=os.path.join(root,file_mark,'neg_white_key')
    neg_black_key_img_dir=os.path.join(root,file_mark,'neg_black_key')

    #--存储该键被按下的帧中的其他按键(随机选取的负样本)
    neg_white_key_img_dir1=os.path.join(root,file_mark,'other_neg_white_key')
    neg_black_key_img_dir1=os.path.join(root,file_mark,'other_neg_black_key')
    
    #--存储测试视频中每一帧的按键,正负样本都有(用以存储每一帧中的所有按键)
    test_white_key_img_dir=os.path.join(root,file_mark,'test_white_key')
    test_black_key_img_dir=os.path.join(root,file_mark,'test_black_key')    

    #---带有键盘编号的图像存储位置
    draw_img_path = os.path.join(root, file_mark, 'draw_img')
    
    ensure_dir(base_img_dir)
    ensure_dir(white_key_img_dir)
    ensure_dir(black_key_img_dir)

    ensure_dir(neg_white_key_img_dir)
    ensure_dir(neg_black_key_img_dir)

    ensure_dir(neg_white_key_img_dir1)
    ensure_dir(neg_black_key_img_dir1)
    
    ensure_dir(test_white_key_img_dir)
    ensure_dir(test_black_key_img_dir)
    ensure_dir(draw_img_path)

    test_dirs = [test_white_key_img_dir, test_black_key_img_dir]
    img_dirs=[white_key_img_dir,black_key_img_dir,neg_white_key_img_dir,
              neg_black_key_img_dir, neg_white_key_img_dir1, neg_black_key_img_dir1, draw_img_path]
              
    return base_img_dir, img_dirs, test_dirs

def init_save_file_dir(save_path,file_mark):
    base_img_dir, img_dirs, test_dirs = new_save_dir(save_path, file_mark)
    return base_img_dir, img_dirs, test_dirs

#---将钢琴键盘中的按键进行编号
def get_key_nums(img_path):
    bwlabel = BwLabel()
    img_lists=[os.path.join(img_path,x) for x in os.listdir(img_path)
               if x.endswith('.jpg')]
    img_lists.sort()
    file_seq=os.path.basename(img_path)
    base_img_dir, img_dirs, test_dirs = init_save_file_dir(cfg.One_key_SAVE_IMG_DIR, file_seq)
    
    base_frame=int(cfg.EVALUATE_MAP[file_seq]['base_frame'])
    base_img=cv2.imread(img_lists[base_frame])
    #---因为现在没用检测键盘的那一步，先手动将键盘的位置分割出来
    rect=cfg.EVALUATE_MAP[file_seq]['keyboard_rect']
    base_img=base_img[rect[1]:rect[3],rect[0]:rect[2]]    
    #---对于背景图中有手的情况，需要输入两次进行调整,(论文中数据集)
    if file_seq in cfg.file_loc:
        keyboard_crop=cfg.EVALUATE_MAP[file_seq]['keyboard_crop']
        test_img=base_img[0:keyboard_crop,:]
        boxes_height = find_key_loc(bwlabel,base_img,phase='Notest')
        white_loc,black_boxes,total_top,total_bottom = find_key_loc(bwlabel,test_img)
        new_black_boxes=[]
        for box in black_boxes:
            if 'level' in file_seq:
                #----对于level_no数据集不用加上
                data = (box[0], box[1], box[2], boxes_height)
            else:
                #---对于SightToSound_paper_path数据集需要加上一点，因为第一个检测到的键h较小
                data = (box[0], box[1], box[2], boxes_height + 6)
            new_black_boxes.append(data)
        black_boxes=new_black_boxes.copy()
    else:
        white_loc,black_boxes,total_top,total_bottom = find_key_loc(bwlabel,base_img)

    draw_img=vis_white_loc_boxes(base_img,white_loc,black_boxes)
    cv2.imwrite(os.path.join(base_img_dir, 'base_draw.jpg'), draw_img)
    cv2.imwrite(os.path.join(base_img_dir, 'base_img.jpg'), base_img)
    
    #---将视频帧图像进行按键编号的绘制
    # vis_white_black_loc_youtube(white_loc, black_boxes, img_dirs[-1], img_lists, rect)
    return white_loc,black_boxes
    


#---直接用视频21的键盘数据绘制对应编号
def get_key_nums1(img_file,save_path,final_index):

    # bwlabel = BwLabel()
    # img_lists=[os.path.join(img_path,x) for x in os.listdir(img_path)
    #            if x.endswith('.jpg')]
    # img_lists.sort()
    # file_seq=os.path.basename(img_path)
    # base_frame=int(cfg.EVALUATE_MAP[file_seq]['base_frame'])
    # base_img=cv2.imread(img_lists[base_frame])

    # save_path=init_save_file_dir(file_seq)
    # white_loc,black_boxes,total_top,total_bottom = find_key_loc(bwlabel,base_img)
    # draw_img=vis_white_loc_boxes(base_img,white_loc,black_boxes)

    #---for test---
    white_loc=cfg.black_white_loc['21']['white_loc']
    black_boxes=cfg.black_white_loc['21']['black_boxes']

    black_num=cfg.black_num
    white_num=cfg.white_num

    img = Image.open(img_file)
    w,h = img.size 
    file_seq = os.path.basename(img_file).split('.')[0]
    opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    
    img_copy=opencv_img.copy()
    height,width,_ = img_copy.shape 
    for i,loc in enumerate(white_loc):
        if i==len(white_loc)-1:break
        key_num=white_num[i]
        cv2.putText(img_copy,str(key_num),(loc+2,height-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    for i,box in enumerate(black_boxes):
        x1,y1= box[0],box[1]
        key_num=black_num[i]
        cv2.putText(img_copy,str(key_num),(x1+3,y1+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
    
    #---将按下的键绘制出来
    press_key=[cfg.labels[x]+1 for x in final_index]
    for index in press_key:
        key_num=int(index)
        if key_num in black_num:
            idx=black_num.index(key_num)
            box=black_boxes[idx]
            x1,y1= box[0],box[1]
            cv2.putText(img_copy,str(key_num),(x1+3,y1+30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)            
        else:
            idx=white_num.index(key_num)
            loc=white_loc[idx]
            cv2.putText(img_copy,str(key_num),(loc+2,height-40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    # cv2.imwrite(os.path.join(save_path,'pres'+os.path.basename(img_file)),img_copy)    
    return img_copy

#---得到视频帧中的各个按键
def get_test_imgs(rect, white_loc, black_boxes, path, test_dirs):
    test_lists = []

    # for video_path in img_lists:
    #     file_seq = os.path.basename(video_path)
    #     if not file_seq == 'level_2_no_02': continue
    #     rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect']
    #     img_list = [os.path.join(video_path, x) for x in os.listdir(video_path)
    #                 if x.endswith('.jpg')]
    w_offset = 4
    y_begin = 5
    b_offset = 3
    y2_offset = 2
    
    # white_loc, black_boxes = get_key_nums(video_path)
    # base_img_dir, img_dirs, test_dirs = init_save_file_dir(file_seq)
    # for path in img_list:
    file_mark = os.path.basename(path).split('.')[0]
    img = cv2.imread(path)
    crop_img = img[rect[1]:rect[3], rect[0]:rect[2]]
    h, w, _ = crop_img.shape
    # w_idx = white_num.index(index)
    for w_idx in range(len(white_loc) - 1):
        start = max(int(white_loc[w_idx]-w_offset),0)
        end = min(int(white_loc[w_idx+1]+w_offset),w)                
        save_img = crop_img[y_begin:h, start:end]
        index = cfg.white_num[w_idx]
        press_path = os.path.join(test_dirs[0], '{}_{}.jpg'.format(file_mark, index))
        cv2.imwrite(press_path, save_img)
        PIL_img = Image.fromarray(save_img).convert('RGB')
        # input_h, input_w = cfg.binary_input_size
        # img = img.resize((input_w, input_h))
        # print(press_path)
        test_lists.append((press_path, PIL_img))
    return test_lists


if __name__=='__main__':
    # img_path=cfg.SightToSound_paper_path
    img_path=cfg.Tencent_path
    img_paths=[os.path.join(img_path,x) for x in os.listdir(img_path)]
    img_paths.sort()
    for path in img_paths:
        # if not os.path.basename(path) in cfg.EVALUATE_MAP.keys():continue
        if not os.path.basename(path) == 'level_4_no_02': continue
        print(path)
        # get_key_nums(path)


    