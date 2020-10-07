
import os 
import torch.nn as nn
import torch
import cv2
import torchvision.transforms as transforms

import sys
sys.path.append('../')
sys.path.append('./')
from config import cfg, PROJECT_ROOT
sys.path.append(os.path.join(PROJECT_ROOT, 'utils'))

from PIL import Image
from IPython import embed

from keyboard_helper.keyboard import KeyBoard
from keyboard_helper.seghand import SegHand
from keyboard_helper.helper import *


from bwlabel import BwLabel
from myhelper import *

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

#----可以写一个关于每个类别的准确率
def cal_accuracy(outputs,targets,top_k,prob_thresh,pressed_keys,epoch):
    func=nn.Sigmoid()
    probs=func(outputs).cpu()
    batch_size=outputs.shape[0]
    correct=0
    pos_keys=0.0001  
    total_keys=0.0001   
    recall=0
    true_keys=0   #---实际按下的键
    TN=0  #---预测为负样本实际也为负样本
    # if epoch > 3: embed()

    for i in range(batch_size):
        prob=probs[i]
        target=targets[i].cpu()
        top_value, indexs = prob.topk(top_k)
        true_index = torch.from_numpy(np.where(np.array(target) == 1)[0])
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

    #----用以存储手区域范围内的按键
    save_white_dir = os.path.join(root, file_mark, 'hand_white_key')
    # save_black_dir = os.path.join(root, file_mark, 'hand_black_key')
    #---用以存储每个八度区域内的键盘
    save_octave_dir = os.path.join(root, file_mark, 'octave_path')
    octave_pos_path = os.path.join(save_octave_dir, 'pos_path')
    octave_neg_path = os.path.join(save_octave_dir, 'neg_path')
    octave_total_path = os.path.join(save_octave_dir, 'total_path')

    ensure_dir(octave_pos_path)
    ensure_dir(octave_neg_path)
    ensure_dir(octave_total_path)

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
    
    ensure_dir(save_white_dir)
    ensure_dir(save_octave_dir)
    ensure_dir(draw_img_path)


    test_dirs = [test_white_key_img_dir, test_black_key_img_dir]
    img_dirs=[white_key_img_dir,black_key_img_dir,neg_white_key_img_dir,
              neg_black_key_img_dir, neg_white_key_img_dir1, neg_black_key_img_dir1, draw_img_path]
    save_dirs = [save_white_dir,save_octave_dir]
    return base_img_dir, img_dirs, test_dirs, save_dirs

def init_save_file_dir(save_path,file_mark):
    base_img_dir, img_dirs, test_dirs,save_dirs = new_save_dir(save_path, file_mark)
    return base_img_dir, img_dirs, test_dirs, save_dirs

def get_octave_imgs(rect, white_loc, black_boxes,
                    path, save_dirs, hand_seg, base_info):
    test_lists = []
    file_mark = os.path.basename(path).split('.')[0]
    img = cv2.imread(path)
    ori_h, ori_w, _ = img.shape
    #---将有旋转矩阵和透视变化的图像要进行处理,正视角
    if base_info is not None:
        if base_info['rote_M'] is None:
            img = img.copy()
        elif base_info['warp_M'] is None:
            rotated_img = cv2.warpAffine(img, base_info['rote_M'], (ori_w, ori_h))
            img = rotated_img.copy()
        else:
            rotated_img = cv2.warpAffine(img, base_info['rote_M'], (ori_w, ori_h))
            warp_img = cv2.warpPerspective(rotated_img, base_info['warp_M'], (ori_w, ori_h))
            img = warp_img.copy()
    
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hand_boxes, mask = hand_seg.segment_detect_hand(pil_img, rect)
    index_list = near_octave_white(white_loc, hand_boxes)
    octave_list = []
    for idxs in index_list:
        for idx in idxs:
            octave_list.append(idx)
    octave_list = list(set(octave_list))
    crop_img = img[rect[1]:rect[3], rect[0]:rect[2]]
    h, w, _ = crop_img.shape
    
    if len(octave_list) > 0:
        l_offset, r_offset = cfg.l_offset, cfg.r_offset
        for idx in octave_list:
            if 'level' in path:
                new_r_offset = r_offset + 1 if idx > 3 else r_offset
            else:
                new_r_offset = r_offset + 3 if idx > 3 else r_offset
            new_l_offset = l_offset - 1 if idx > 3 else l_offset            
            begin = 2 + idx * 7
            end = 9 + idx * 7
            octave_img = crop_img[:, np.maximum(white_loc[begin] - new_l_offset,0):
                                np.minimum(white_loc[end] + new_r_offset, w)]
            test_lists.append(((path, idx), octave_img))
    return test_lists

#---得到视频帧中的各个按键,用以测试视频的准确率
def get_test_imgs(rect, white_loc, black_boxes, path, save_dirs, hand_seg):
    test_lists = []
    file_seq = os.path.dirname(path).split('/')[-1]
    offset = cfg.offset['level'] if len(file_seq) > 3 else cfg.offset['paper']
    w_offset = offset['w_offset']
    b_offset = offset['b_offset']
    y2_offset = offset['y2_offset']
    y_begin = offset['y_begin']    
    w_offset = 4
    y_begin = 5
    b_offset = 3
    y2_offset = 2

    file_mark = os.path.basename(path).split('.')[0]
    img = cv2.imread(path)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hand_boxes, mask = hand_seg.segment_detect_hand(pil_img, rect)
    index_list = near_white(white_loc, hand_boxes)
    whole_list = []
    for hand_list in  index_list:
        for index in range(hand_list[0],hand_list[1]+1):
            # whole_list.append(cfg.white_num[index])
            whole_list.append(index)
    whole_list = np.maximum(whole_list,0)
    whole_list = np.minimum(whole_list,len(white_loc)-2) 
    whole_list = list(set(whole_list))
    whole_list.sort()
    for box in hand_boxes:
        img = cv2.rectangle(img, box[0], box[1], (0, 0, 255))
    cv2.imwrite('./test.jpg', img)

    crop_img = img[rect[1]:rect[3], rect[0]:rect[2]]
    h, w, _ = crop_img.shape
    # print(whole_list)
    if len(whole_list) > 0:
        for w_idx in whole_list:
            start = max(int(white_loc[w_idx]-w_offset),0)
            end = min(int(white_loc[w_idx+1]+w_offset),w)                
            save_img = crop_img[y_begin:h, start:end]
            index = cfg.white_num[w_idx]
            press_path = os.path.join(save_dirs[0], '{}_{}.jpg'.format(file_mark, index))
            # print(press_path)
            cv2.imwrite(press_path, save_img)
            PIL_img = Image.fromarray(save_img).convert('RGB')
            # input_h, input_w = cfg.binary_input_size
            # img = img.resize((input_w, input_h))
            # print(press_path)
            press_path = press_path.replace(os.path.basename(save_dirs[0]), 'test_white_key')
            test_lists.append((press_path, PIL_img))
    return test_lists

class VisAmtHelper(object):
    def __init__(self):
        super(VisAmtHelper, self).__init__()

    def init_model_load(self):
        self.keyboard = KeyBoard()
        self.hand_seg = SegHand()
        self.bwlabel = BwLabel()

    def process_img_dir(self, img_path):
        img_lists=[os.path.join(img_path,x) for x in os.listdir(img_path)
                   if x.endswith('.jpg')]
        img_lists.sort()
        file_seq = os.path.basename(img_path)
        #---whether it is the Tencent video 

        if len(file_seq) > 3:
            base_info = find_base_img(self.keyboard, self.hand_seg, img_lists)
            base_img = base_info['base_img']
            base_all_img = base_info['img']  #---原始背景图
            rect = base_info['rect']
            warp_M = base_info['warp_M']
            rote_M = base_info['rote_M']
            #---第一张检测到键盘的帧/无论是否有手
            begin_frame, end_frame = find_begin_frame(self.keyboard, img_lists)
            base_info['count_frame'] = begin_frame
            base_info['end_frame'] = end_frame

        else:            
            base_frame=int(cfg.EVALUATE_MAP[file_seq]['base_frame'])
            base_img = cv2.imread(img_lists[base_frame])
            #---因为现在没用检测键盘的那一步，先手动将键盘的位置分割出来
            rect=cfg.EVALUATE_MAP[file_seq]['keyboard_rect']
            base_img = base_img[rect[1]:rect[3], rect[0]:rect[2]]
            base_info = None
            first_img_num = 0

        h, w, _ = base_img.shape
        #---对于背景图像有手的视频
        if file_seq in cfg.file_loc:
            keyboard_crop=cfg.EVALUATE_MAP[file_seq]['keyboard_crop']
            h_edge = int(h * 1.0 / 2)
            test_img = base_img[0:h_edge,:]
            # test_img = base_img[0:keyboard_crop,:]
            boxes_height = find_img_key_loc(self.bwlabel,base_img,phase='Notest')
            white_loc,black_boxes,total_top,total_bottom = find_img_key_loc(self.bwlabel,test_img)
            new_black_boxes=[]
            for box in black_boxes:
                if len(file_seq) > 3:
                    #----对于level_no数据集不用加上
                    data = (box[0], box[1], box[2], boxes_height)
                else:
                    #---对于SightToSound_paper_path数据集需要加上一点，因为第一个检测到的键h较小
                    data = (box[0], box[1], box[2], boxes_height + 6)
                new_black_boxes.append(data)
            black_boxes = new_black_boxes.copy()
        else:
            # pass
            white_loc, black_boxes, total_top, total_bottom = find_img_key_loc(self.bwlabel, base_img)
        img_save_path = cfg.Test_Key_Dir if file_seq in cfg.Test_video else cfg.One_key_SAVE_IMG_DIR
        base_img_dir, img_dirs, test_dirs, save_dirs = init_save_file_dir(img_save_path, file_seq)
        imgs = [os.path.join(img_dirs[-1], x) for x in os.listdir(img_dirs[-1])
                if x.endswith('.jpg')]
        # --是否绘制键编号并保存,有的话就不用绘制
        save_flag = False if len(imgs) > 0 else True
        # save_flag = True
        draw_img = vis_white_loc_boxes(base_img, white_loc, black_boxes)
        cv2.imwrite(os.path.join(base_img_dir, 'base_draw.jpg'), draw_img)
        cv2.imwrite(os.path.join(base_img_dir, 'base_img.jpg'), base_img)
        #---将视频帧图像进行按键编号的绘制
        if save_flag:
            vis_white_black_loc_youtube(white_loc, black_boxes, img_dirs[-1], img_lists, rect, base_info)
        return white_loc, black_boxes,base_info
        

if __name__ == '__main__':
    # img_path=cfg.SightToSound_paper_path
    img_path = cfg.Tencent_path
    # img_path = cfg.Record_path
    img_paths=[os.path.join(img_path,x) for x in os.listdir(img_path)]
    img_paths.sort()
    VisAmt = VisAmtHelper()
    VisAmt.init_model_load()
    for path in img_paths:
        # if not os.path.basename(path) in cfg.file_loc:continue
        if not os.path.basename(path) == 'level_4_no_02': continue
        print(path)
        # get_key_nums(path)
        VisAmt.process_img_dir(path)



    