import os
import cv2
from PIL import Image
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import cfg
from IPython import embed

class Binary_dataset(Dataset):
    def __init__(self, list_path,img_size,Data_type=None,phase='train'):
        self.phase=phase
        self.img_size = img_size
        self.split = cfg.split
        self.img_files = self.get_imgs(list_path)
        self.count_nums()
        self.type = Data_type
        self.num_classes=2

        self.transforms=True if phase=='train' else False

    #---统计不同区域按键的数量
    def count_nums(self):
        left_nums, middle_nums, right_nums = 0, 0, 0
        begin, end = self.split[0], self.split[1]
        for line in self.img_files:
            key = int(os.path.basename(line[0]).split('.')[0].split('_')[1])
            # print(line[0])
            # print(key)
            if key <= begin:
                left_nums += 1
            elif key >= end:
                right_nums += 1
            else:
                middle_nums += 1
        print('phase:{}\tleft:{}\tmiddle:{}\tright:{} '.format(
              self.phase, left_nums, middle_nums, right_nums))

    #---得到位置信息的标签,给按键添加上位置信息，分为3个区域
    def get_pos_label(self, img_path):
        begin, end = self.split[0], self.split[1]
        key = int(os.path.basename(img_path).split('.')[0].split('_')[1])
        #---左中右: 0.1.2
        if key <= begin:
            return 0
        elif key >= end:
            return 2
        else:
            return 1

    def get_imgs(self,list_path):
        with open(list_path, "r") as file:
            img_files = file.readlines()
        key_line=[]

        for line in img_files:
            line = line.strip().split()
            key = line[1]
            path=line[0]
            key_line.append((path,key))

        return key_line

    def __getitem__(self, index):
        lines = self.img_files[index % len(self.img_files)]
        img_path=lines[0]
        press_label = int(lines[1])
        pos_label = self.get_pos_label(img_path)
        target=(press_label,pos_label)

        # Extract image as PyTorch tensor
        img = Image.open(img_path).convert('RGB')
        (input_h, input_w) = self.img_size
        if self.phase == 'train':
            transform=transforms.Compose([
                transforms.Resize((input_h,input_w)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform=transforms.Compose([
                transforms.Resize((input_h,input_w)),
                transforms.ToTensor(),
            ])
        img = transform(img)
        # label = torch.LongTensor([label])

        if self.type == None:
            return img, press_label
        else:
            return img,target

    def __len__(self):
        return len(self.img_files)

#---得到当前帧前后的连续几帧---
def get_continue_path(path, begin_frame, end_frame, cur_frame, continuous_frame):
    sequence_imgs = []
    #---对于一开始的几帧----
    if cur_frame - begin_frame < continuous_frame:
        temp_frame = cur_frame
        #---将前面的几帧补上
        while True:
            if temp_frame - 1 >= 0:
                temp_frame -= 1
                before_path = path.replace('{:0>4d}'.format(cur_frame),
                                        '{:0>4d}'.format(temp_frame))
                # img = cv2.imread(before_path)
                sequence_imgs.insert(0, before_path)
            else:
                tmp_path = path.replace('{:0>4d}'.format(cur_frame),
                                        '{:9>4d}'.format(9))
                sequence_imgs.insert(0, tmp_path)
            if len(sequence_imgs) == continuous_frame:
                break
        sequence_imgs.append(path)
        #---将后面的几帧补上
        temp_frame = cur_frame
        while True:
            temp_frame += 1
            after_path = path.replace('{:0>4d}'.format(cur_frame),
                                        '{:0>4d}'.format(temp_frame))
            # img = cv2.imread(after_path)
            sequence_imgs.append(after_path)
            if len(sequence_imgs) == cfg.Consecutive_frames:
                break
    #---对于最后面的几帧，其之后没有对应帧
    elif cur_frame + continuous_frame > end_frame:
        temp_frame = cur_frame
        #---将前面的几帧补上
        while True:   
            temp_frame -= 1
            before_path = path.replace('{:0>4d}'.format(cur_frame),
                                        '{:0>4d}'.format(temp_frame))
            # img = cv2.imread(before_path)
            sequence_imgs.insert(0, before_path)
            if len(sequence_imgs) == continuous_frame:
                break

        sequence_imgs.append(path)
        #---将后面的几帧补上
        temp_frame = cur_frame
        while True:
            if temp_frame + 1 > end_frame:
                after_path = path.replace('{:0>4d}'.format(cur_frame),
                                     '{:9>4d}'.format(9))
                # img = cv2.imread(before_path)
                sequence_imgs.append(after_path)
            else:
                temp_frame += 1
                tmp_path = path.replace('{:0>4d}'.format(cur_frame),
                                        '{:0>4d}'.format(temp_frame))
                sequence_imgs.append(tmp_path)
            if len(sequence_imgs) == cfg.Consecutive_frames:
                break
    #---对于一般的帧，加上其前后几帧
    else:
        for i in range(1,continuous_frame+1):
            before_path = path.replace('{:0>4d}'.format(cur_frame),
                                        '{:0>4d}'.format(cur_frame - i))
            after_path = path.replace('{:0>4d}'.format(cur_frame),
                                        '{:0>4d}'.format(cur_frame + i))
            sequence_imgs.insert(0, before_path)
            sequence_imgs.append(after_path)
        sequence_imgs.insert(continuous_frame, path)
    return sequence_imgs

#---生成用以训练连续几帧的数据
class Time_dataset(Dataset):
    def __init__(self, list_path, img_size,Data_type=None, k=5, phase='train'):
        self.phase = phase
        self.img_size = img_size
        self.split = cfg.split
        self.k = k  #---输入连续k帧
        self.continuous_frame = int((self.k - 1) / 2)
        self.img_files = self.get_imgs(list_path)
        self.count_nums()
        self.type = Data_type
        self.num_classes=2

        self.transforms = True if phase == 'train' else False

    #---统计不同区域按键的数量
    def count_nums(self):
        left_nums, middle_nums, right_nums = 0, 0, 0
        begin, end = self.split[0], self.split[1]
        for line in self.img_files:
            line = line[0]
            key = int(os.path.basename(line[0]).split('.')[0].split('_')[1])
            # print(line[0])
            # print(key)
            if key <= begin:
                left_nums += 1
            elif key >= end:
                right_nums += 1
            else:
                middle_nums += 1
        print('phase:{}\tleft:{}\tmiddle:{}\tright:{} '.format(
              self.phase, left_nums, middle_nums, right_nums))

    #---得到位置信息的标签,给按键添加上位置信息，分为3个区域
    def get_pos_label(self, img_path):
        begin, end = self.split[0], self.split[1]
        key = int(os.path.basename(img_path).split('.')[0].split('_')[1])
        #---左中右: 0.1.2
        if key <= begin:
            return 0
        elif key >= end:
            return 2
        else:
            return 1

    def get_imgs(self,list_path):
        with open(list_path, "r") as file:
            img_files = file.readlines()
        img_lists = []

        def get_tmp_path(cur_frame, path):
            img = cv2.imread(path)
            h, w, c = img.shape
            tmp_img = np.random.randint(128, 129, (h, w, c)).astype(np.uint8)
            return (tmp_path, tmp_img)
        
        for line in tqdm(img_files):
            line = line.strip().split()
            key = line[1]
            path = line[0]
            dir_name = os.path.dirname(path).split('/')[-1]
            file_seq = os.path.dirname(path).split('/')[-2]
            begin_frame = cfg.EVALUATE_MAP[file_seq]['begin_frame']
            end_frame = cfg.EVALUATE_MAP[file_seq]['end_frame']
            path = path.replace(dir_name, 'test_white_key')
            cur_img = cv2.imread(path)
            self.continuous_frame = int((self.k - 1) / 2)
            cur_frame = int(os.path.basename(path).split('.')[0].split('_')[0])
            #---对于刚开始的几帧，其之前没有对应帧
            sequence_imgs = get_continue_path(path, begin_frame,end_frame, cur_frame, self.continuous_frame)
            seq_imgs = (sequence_imgs, key)
            img_lists.append(seq_imgs)
        return img_lists

    def __getitem__(self, index):
        lines = self.img_files[index % len(self.img_files)]
        img_paths = lines[0]
        press_label = int(lines[1])
        pos_label = self.get_pos_label(img_paths[0])
        target = (press_label, pos_label)

        # Extract image as PyTorch tensor
        img_lists = []
        flag = False
        #---因为是连续几帧图像,要镜像的话就需要一起镜像---
        if np.random.rand() > 0.5:
            flag = True
        
        (input_h, input_w) = self.img_size
        if self.phase == 'train' and flag == True:
            transform=transforms.Compose([
                transforms.Resize((input_h,input_w)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform=transforms.Compose([
                transforms.Resize((input_h,input_w)),
                transforms.ToTensor(),
            ])
                          
        for i, img_path in enumerate(img_paths):
            if '9999' in img_path:
                tmp_img = cv2.imread(img_paths[self.continuous_frame])
                h, w, c = tmp_img.shape
                img = np.random.randint(128, 129, (h, w, c)).astype(np.uint8)
                img = Image.fromarray(img).convert('RGB')
            else:
                img = Image.open(img_path).convert('RGB')

            img = transform(img)
            img_lists.append(img)

        if self.type == None:
            return img_lists,img_paths, press_label
        else:
            return img_lists, target

    def __len__(self):
        return len(self.img_files)

if __name__ == '__main__':
    # train_set = Binary_dataset(cfg.w_train_binary_txt_path,
    #                            cfg.binary_input_size, Data_type='pos')
    # img, target = train_set.__getitem__(30)

    Time_set=Time_dataset(cfg.w_train_binary_txt_path,
                          cfg.binary_input_size, k=5, Data_type=None)
    img_lists,img_paths, label = Time_set.__getitem__(5)
