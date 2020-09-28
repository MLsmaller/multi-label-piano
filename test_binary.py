import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import os 
import numpy as np
import cv2
import random
from PIL import Image
import shutil
import argparse
from tqdm import tqdm

from model.resnet import ResNet_112_32, ResNet_Time
from config import cfg
from utils.keyboard_helper.seghand import SegHand
from data.binary_dataset import Binary_dataset, Time_dataset, get_continue_path
from utils.utils import cal_accuracy, get_test_imgs, init_save_file_dir, VisAmtHelper
from utils.evaluate import Accuracy
from IPython import embed

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_lists", type=str, default=None, 
                        help="None for test img lists,true for test dataloader")
    parser.add_argument('--Data_type', type=str, default=None,
                        help='whether train the dataset with position information')
    parser.add_argument('--with_Time', type=str, default=True,
                        help='whether train the dataset with Time sequential')
    parser.add_argument('--thresh', type=float, default=0.9,
                        help='the prob thresh to judge whether pressed or not')
    opt = parser.parse_args()
    return opt
opt=parser()

def select_img(save_path,select_num=10):
    with open(cfg.w_val_binary_txt_path,'r') as f:
        lines=f.readlines()
    test_lines=random.sample(lines,select_num)
    for line in test_lines:
        line=line.strip().split()
        path=line[0]
        file_seq=os.path.basename(os.path.split(path)[0])
        print(path)
        shutil.copy(path,os.path.join(save_path,os.path.basename(os.path.split(path)[0])+
                                        '_'+os.path.basename(path)))    

def select_img1(select_num=10):
    with open(cfg.w_val_binary_txt_path,'r') as f:
        lines=f.readlines()
    test_lines=random.sample(lines,select_num)
    return test_lines

fmap_block = list()
grad_block = list()  


# def backward_hook(module, grad_in, grad_out):
#     grad_block.append(grad_out[0].detach())

# def farward_hook(module, input, output):
#     fmap_block.append(output)

#---预处理
def process_img(img, img_paths, mode='nolist'):
    input_h, input_w = cfg.binary_input_size
    if mode == 'list':
        img_lists = []
        for img_path in img_paths:
            #---即构造出来的图像(主要针对最开始和最后面几帧)
            if '9999' in img_path:
                tmp_img = cv2.imread(img_paths[int((cfg.Consecutive_frames - 1) / 2)])
                h, w, c = tmp_img.shape
                img = np.random.randint(128, 129, (h, w, c)).astype(np.uint8)
                img = Image.fromarray(img).convert('RGB')
            else:
                img = Image.open(img_path).convert('RGB')            
            img = img.resize((input_w, input_h))
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(0).to(device)
            img_lists.append(img)
        return img_lists
    else:
        img = img.resize((input_w, input_h))
        img = transforms.ToTensor()(img)
        # embed()
        if len(img.shape) !=3:
            img=img.unsqueeze(0)
            img=img.expand((3,img.shape[1:]))
        img = img.unsqueeze(0).to(device)
        return img

def test_videos(video_path):
    white_loc, black_boxes,base_info = VisAmt.process_img_dir(video_path)
    rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect'] if base_info is None else base_info['rect']
    # rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect']
    img_list = [os.path.join(video_path, x) for x in os.listdir(video_path)
                if x.endswith('.jpg')]
    img_list.sort()
    base_img_dir, img_dirs, test_dirs,save_dirs = init_save_file_dir(cfg.Test_Key_Dir, file_seq)
    fps = int(cfg.EVALUATE_MAP[file_seq]['fps'])
    pframe_time = 1.0 / fps

    if opt.with_Time:
        if opt.Data_type:
            w_txt_path, b_txt_path = 'time_pos_pitch_white.txt', 'time_pos_pitch_black.txt'
        else:
            w_txt_path, b_txt_path = 'time_pitch_white.txt', 'time_pitch_black.txt'
    else:
        if opt.Data_type:
            w_txt_path, b_txt_path = 'pos_pitch_white.txt', 'pos_pitch_black.txt'
        else:
            w_txt_path, b_txt_path = 'pitch_white.txt', 'pitch_black.txt'
                            
    w_detectPath = os.path.join(cfg.Test_Key_Dir, file_seq, w_txt_path)
    b_detectPath = os.path.join(cfg.Test_Key_Dir, file_seq, b_txt_path)
        
    b_out = open(b_detectPath, 'w')
    b_out.close()
    w_out = open(w_detectPath, 'w')
    
    begin_frame = 0 if base_info is None else base_info['count_frame']
    end_frame = len(img_list)-1 if base_info is None else base_info['end_frame']
    for path in tqdm(img_list):
        #--只有这些范围的帧中才包含键盘图像
        num = int(os.path.basename(path).split('.')[0])
        if not num in range(begin_frame, end_frame + 1): continue
        press_list = []
        #---只包含手范围附近的白键
        test_lists = get_test_imgs(rect, white_loc, black_boxes, path, save_dirs, hand_seg)
        for (img_path, img) in test_lists:
            #---对于带有时间信息的模型
            if opt.with_Time:
                cur_frame = int(os.path.basename(img_path).split('.')[0].split('_')[0])
                k = cfg.Consecutive_frames
                continuous_frame = int((k - 1) / 2)
                sequence_imgs = get_continue_path(img_path, begin_frame, end_frame, cur_frame, continuous_frame)
                # print(img_path)
                # embed()
                img = process_img(img, sequence_imgs, mode='list')
                output = model(img)
            else:
                img = process_img(img,img_path)
                # if not os.path.basename(img_path) == '0000_18.jpg': continue
                output = model(img)

            if opt.Data_type:
                prob = F.softmax(output[0], dim=1).squeeze()
            else:
                prob = F.softmax(output, dim=1).squeeze()
            prob = prob.cpu().detach().numpy()
            result = np.where(prob > thresh)[0]
            key_idx = os.path.basename(img_path).split('.')[0].split('_')[-1]
            if len(result) > 0 and result[0] == 1:
                press_list.append(int(key_idx))
                # print('the press path is {}'.format(img_path))
                # embed()

        cur_frame = int(os.path.basename(path).split('.')[0])
        cur_time = pframe_time * cur_frame
        w_out.write('{} '.format(path))
        w_out.write('{} '.format(cur_time))
        if len(press_list) > 0:
            for key in press_list:
                w_out.write('{} '.format(key))
            w_out.write('\n')
        else:
            w_out.write('{}\n'.format(0))
        
    w_out.close()
    midiPath = cfg.EVALUATE_MAP[file_seq]['label_path']
    midiPath = midiPath.replace('_label', '_note')
    evaluate = Accuracy(midiPath=midiPath,
                        w_detectPath=w_detectPath,
                        b_detectPath=b_detectPath,
                        pframe_time = pframe_time,
                        midi_offset=0)    

def main(img_lists, test_img_lists):
    if opt.with_Time:
        model = ResNet_Time(Data_type=opt.Data_type, k=cfg.Consecutive_frames).to(device)
        ckpt_path = cfg.time_ckpt_path if opt.Data_type is None else cfg.time_pos_ckpt_path
    else:
        model = ResNet_112_32(opt.Data_type).to(device)
        ckpt_path = cfg.binary_ckpt_path if opt.Data_type is None else cfg.pos_binary_ckpt_path
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # model.make_layers[-1].right[-2].register_forward_hook(farward_hook)
    # model.make_layers[-1].right[-2].register_backward_hook(backward_hook)
    print('the ckpt path is {}'.format(ckpt_path))
    test_lines=select_img1()
    out_dir = './output'
    thresh = opt.thresh
    
    #---找到开始帧和结束帧,用以训练数据的标注
    # VisAmt = VisAmtHelper()
    # VisAmt.init_model_load()
    # for video_path in test_img_lists:
    #     img_list = [os.path.join(video_path, x) for x in os.listdir(video_path)
    #                 if x.endswith('.jpg')]        
    #     white_loc, black_boxes, base_info = VisAmt.process_img_dir(video_path)
    #     begin_frame = 0 if base_info is None else base_info['count_frame']
    #     end_frame = len(img_list) - 1 if base_info is None else base_info['end_frame']
    #     print(video_path)
    #     print('the begin_frame is {} and end_frame is {}'.format(begin_frame, end_frame))

    if opt.img_lists:
        for line in test_lines:
            # if not 'test_0013' in os.path.basename(img_path):continue
            line=line.strip().split()
            img_path = line[0]
            print(img_path)
            # ori_img=cv2.imread(img_path,1)
            
            img_draw = cv2.imread(img_path)  #---for draw
            img = Image.open(img_path)
            img = process_img(img, img_path)
            
            output = model(img)
            prob = torch.nn.functional.softmax(output, dim=1).squeeze()
            prob = prob.cpu().detach().numpy()
            result = np.where(prob > thresh)[0]
            print('the label is {} and predict is {}'.format(line[1], result))
            print(prob)
    else:
        fout = open(cfg.res_txt_path, 'w')
        VisAmt = VisAmtHelper()
        VisAmt.init_model_load()
        hand_seg = SegHand()
        for video_path in test_img_lists:
            file_seq = os.path.basename(video_path)
            if not file_seq == 'level_1_no_02': continue    #----for test
            
            if not file_seq in cfg.Test_video: continue
            if file_seq == 'level_4_no_02': continue
            test_videos(video_path)
                                

if __name__=='__main__':
    save_path='./test_imgs'
    img_lists=[os.path.join(save_path,x) for x in os.listdir(save_path)
               if x.endswith('.jpg')]
    img_lists.sort()    
    
    img_path1 = [os.path.join(cfg.Tencent_path, x) for x in os.listdir(cfg.Tencent_path)]
    img_path2 = [os.path.join(cfg.SightToSound_paper_path, x) for x in os.listdir(cfg.SightToSound_paper_path)]
    test_img_lists=[]
    test_img_lists.extend(img_path1)
    test_img_lists.extend(img_path2)
    test_img_lists.sort()

    # select_img(save_path)
    
    # test_lists = get_test_imgs(test_img_lists)
    main(img_lists, test_img_lists)
    


