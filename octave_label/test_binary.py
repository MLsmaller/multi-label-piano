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

from model.resnet import resnet18
from config import cfg
from utils.keyboard_helper.seghand import SegHand
from data.binary_dataset import Binary_dataset, Time_dataset, get_continue_path
from utils.utils import cal_accuracy, get_octave_imgs, init_save_file_dir, VisAmtHelper
from utils.total_evaluate import Accuracy as Total_Accuracy
from utils.evaluate import Accuracy
from IPython import embed

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_lists", type=str, default=None, 
                        help="None for test img lists,true for test dataloader")
    parser.add_argument('--Data_type', type=str, default=None,
                        help='whether train the dataset with position information')
    parser.add_argument('--with_Time', type=str, default=False,
                        help='whether train the dataset with Time sequential')
    parser.add_argument('--thresh', type=float, default=0.9,
                        help='the prob thresh to judge whether pressed or not')
    opt = parser.parse_args()
    return opt




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

#---预处理
def process_img(img, img_paths, mode='nolist'):
    input_w, input_h = cfg.octave_final_size
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
                img = Image.open(img_path)         
            img = img.resize((input_w, input_h))
            img = transforms.ToTensor()(img)
            img = img.unsqueeze(0).to(device)
            img_lists.append(img)
        return img_lists
    else:
        #---要不直接读取图像然后crop
        # ori_img = img.copy()
        #---要先转换为RGB再调用函数fromarray，而不是先fromarray再调用convert
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # img = Image.fromarray(img).convert('RGB')
        img = img.resize((input_w, input_h))
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)
        return img

#---将八度中的键转换为1-88的按键
def convert2note(idx, key_indexs):
    new_indexs = []
    for index in key_indexs:
        press_key = 4 + int(np.array(index)) + idx * 12
        new_indexs.append(press_key)
    return new_indexs



def test_videos(model, hand_seg, VisAmt, video_path,opt):
    file_seq = os.path.basename(video_path)
    white_loc, black_boxes,base_info = VisAmt.process_img_dir(video_path)
    rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect'] if base_info is None else base_info['rect']
    # rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect']
    img_list = [os.path.join(video_path, x) for x in os.listdir(video_path)
                if x.endswith('.jpg')]
    img_list.sort()
    base_img_dir, img_dirs, test_dirs,save_dirs = init_save_file_dir(cfg.Test_Key_Dir, file_seq)
    fps = float(cfg.EVALUATE_MAP[file_seq]['fps'])
    pframe_time = 1.0 / fps

    if opt.with_Time:
        if opt.Data_type:
            txt_path = 'octave_time_pos_pitch.txt'
            octave_w_path = 'octave_w_time_pos_pitch.txt'
            octave_b_path = 'octave_b_time_pos_pitch.txt'
        else:
            txt_path = 'octave_time_pitch.txt'
            octave_w_path = 'octave_w_time_pitch.txt'
            octave_b_path = 'octave_b_time_pitch.txt'
    else:
        if opt.Data_type:
            txt_path = 'octave_pos_pitch.txt'
            octave_w_path = 'octave_w_pos_pitch.txt'
            octave_b_path = 'octave_b_pos_pitch.txt'
        else:
            txt_path = 'octave_pitch.txt'
            octave_w_path = 'octave_w_pitch.txt'
            octave_b_path = 'octave_b_pitch.txt'
                            
    detectPath = os.path.join(cfg.Test_Key_Dir, file_seq, txt_path)
    w_detectPath = os.path.join(cfg.Test_Key_Dir, file_seq, octave_w_path)
    b_detectPath = os.path.join(cfg.Test_Key_Dir, file_seq, octave_b_path)
    f_out = open(detectPath, 'w')
    w_out = open(w_detectPath, 'w')
    b_out = open(b_detectPath, 'w')
    
    begin_frame = 0 if base_info is None else base_info['count_frame']
    end_frame = len(img_list)-1 if base_info is None else base_info['end_frame']
    for path in tqdm(img_list):
        #--只有这些范围的帧中才包含键盘图像
        num = int(os.path.basename(path).split('.')[0])
        if not num in range(begin_frame, end_frame + 1): continue
        press_list = []
        #---只包含手范围附近的八度
        test_lists = get_octave_imgs(rect, white_loc, black_boxes,
                     path, save_dirs, hand_seg, base_info)
        for ((img_path, idx), img) in test_lists:
            # print(img_path)
            # print('the octave idx is {}'.format(idx))
            #---对于带有时间信息的模型
            if opt.with_Time:
                cur_frame = int(os.path.basename(img_path).split('.')[0].split('_')[0])
                k = cfg.Consecutive_frames
                continuous_frame = int((k - 1) / 2)
                octave_path = os.path.join(cfg.Test_Key_Dir, file_seq, 'octave_path/total_path')
                file_mark = os.path.basename(img_path).split('.')[0] + '_{}.jpg'.format(idx)
                octave_imgpath = os.path.join(octave_path, file_mark)
                sequence_imgs = get_continue_path(octave_imgpath, begin_frame, end_frame, cur_frame, continuous_frame)
                img = process_img(img, sequence_imgs, mode='list')
                # #---查看数据是否正确
                # for idx,a_img in enumerate(img):
                #     numpy_img = np.squeeze((a_img.cpu().numpy()) * 255, 0).astype(np.uint8).transpose((1, 2, 0))
                #     numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite('./{}.jpg'.format(idx),numpy_img)
                output = model(img)
            else:
                img = process_img(img, img_path)
                output = model(img)
            
            func=nn.Sigmoid()
            if opt.Data_type:
                probs = func(output[0]).cpu()
            else:
                probs = func(output).cpu()
            top_value, indexs = probs.topk(cfg.octave_top_k)
            key_indexs = indexs[top_value > cfg.octave_test_thresh]
            new_indexs = convert2note(idx, key_indexs)

            if len(new_indexs) > 0:
                press_list.append(new_indexs)
                # print('the press key is {}'.format(new_indexs))
                # print('the press path is {}'.format(img_path))
                # print('\n')

        cur_frame = int(os.path.basename(path).split('.')[0])
        cur_time = pframe_time * cur_frame
        #--Total eval
        f_out.write('{} '.format(path))
        f_out.write('{:.2f} '.format(cur_time))
        #--White eval
        w_out.write('{} '.format(path))
        w_out.write('{:.2f} '.format(cur_time))
        #--Black eval
        b_out.write('{} '.format(path))
        b_out.write('{:.2f} '.format(cur_time))
        if len(press_list) > 0:
            w_count, b_count = 0, 0
            for key in press_list:
                for key_idx in key:
                    f_out.write('{} '.format(key_idx))
                    if key_idx in cfg.white_num:
                        w_count+= 1
                        w_out.write('{} '.format(key_idx))
                    else:
                        b_count+= 1
                        b_out.write('{} '.format(key_idx))
            if w_count == 0: w_out.write('{}'.format(0))
            if b_count == 0: b_out.write('{}'.format(0))
            f_out.write('\n')
            w_out.write('\n')
            b_out.write('\n')
        else:
            f_out.write('{}\n'.format(0))
            w_out.write('{}\n'.format(0))
            b_out.write('{}\n'.format(0))

        if int(cur_frame) > 120:
            break

    f_out.close()
    w_out.close()
    b_out.close()
    midiPath = cfg.EVALUATE_MAP[file_seq]['label_path']
    midiPath = midiPath.replace('_label', '_note')
    # midi_offset = cfg.EVALUATE_MAP[file_seq]['midi_offset']
    midi_offset = 0
    Total_evaluate = Total_Accuracy(midiPath=midiPath,
                        detectPath=detectPath,
                        pframe_time=pframe_time,
                        midi_offset = midi_offset,
                        )
    evaluate = Accuracy(midiPath=midiPath,
                        w_detectPath=w_detectPath,
                        b_detectPath=b_detectPath,
                        pframe_time=pframe_time,
                        midi_offset = midi_offset,
                        )
    embed()

def main(img_lists, test_img_lists):
    if opt.with_Time:
        model = resnet18(pretrained=False, num_classes=cfg.octave_num_classes,
                         k=cfg.Consecutive_frames, phase='time').to(device)        
        ckpt_path = cfg.octave_time_path if opt.Data_type is None else cfg.time_pos_ckpt_path
    else:
        model = resnet18(pretrained=False,num_classes=cfg.octave_num_classes).to(device)
        ckpt_path = cfg.octave_ckpt_path if opt.Data_type is None else cfg.binary_ckpt_path
    model.load_state_dict(torch.load(ckpt_path,map_location='cuda:0'))
    model.eval()

    print('the ckpt path is {}'.format(ckpt_path))
    test_lines=select_img1()
    out_dir = './output'
    thresh = opt.thresh
    
    # #---找到开始帧和结束帧,用以训练数据的标注
    # VisAmt = VisAmtHelper()
    # VisAmt.init_model_load()
    # for video_path in test_img_lists:
    #     if not 'level_2_no_02' in video_path: continue
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
        VisAmt = VisAmtHelper()
        VisAmt.init_model_load()
        hand_seg = SegHand()
        for video_path in test_img_lists:
            file_seq = os.path.basename(video_path)
            if not file_seq in ['level_1_no_02', 'level_2_no_02']: continue  #----for test
            
            if not file_seq in cfg.Test_video: continue
            if file_seq == 'level_4_no_02': continue
            print(video_path)
            print('the octave test thresh is {}'.format(cfg.octave_test_thresh))
            test_videos(model, hand_seg, VisAmt, video_path, opt)
            

if __name__ == '__main__':
    opt = parser()
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
    
    main(img_lists, test_img_lists)



