import glob
import random
import os
import cv2
import sys
sys.path.append('../')
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import cfg
from IPython import embed

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def random_crop(img,img_size):
    width,height=img.size
    # left = np.random.uniform(0, 5)
    #---上下随机crop,左右不crop
    top= np.random.uniform(0, 5)

    # convert to integer rect x1,y1,x2,y2
    rect =[0, int(top), int(img_size[0]), int(img_size[1]+top)]
    img=img.crop(rect)  
    return img


def random_brightness(img):
    prob = np.random.uniform(0, 1)
    if np.random.rand()>0.5:
        delta = np.random.uniform(-cfg.brightness_delta,
                                  cfg.brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class KeyDataset(Dataset):
    def __init__(self, list_path, img_size=(224, 165), phase='train'):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.img_size = img_size
        self.transforms=True if phase=='train' else False

    def __getitem__(self, index):
        lines = self.img_files[index % len(self.img_files)].strip().split()
        img_path=lines[0]
        keys=lines[1:]
        targets = np.zeros(12)
        for index in keys:
            #---标签中标的是从0-11,已经是从1开始了        
            targets[int(index)]=1

        w, h = self.img_size
        # Extract image as PyTorch tensor
        img = Image.open(img_path).convert('RGB')
        file_seq=os.path.basename(os.path.split(img_path)[0])        
        
        img = img.resize((w, h + 5))
        img = random_crop(img, (w, h))

        if self.transforms:
            img = random_brightness(img)

        img = transforms.ToTensor()(img)
        return img, targets

    def __len__(self):
        return len(self.img_files)

#---还可以加上pos信息
class KeyDataset_select(Dataset):
    def __init__(self, list_path, img_size=(224, 165),phase='train'):
        self.phase=phase

        self.img_files, self.img_dict, self.key_num = self.get_imgs(list_path)
        self.num_classes = len(self.key_num) #12
        print('the key num is {}'.format(self.key_num))
        # print(len(self.key_num))

        self.img_size = img_size
        self.transforms=True if phase=='train' else False

    def get_imgs(self,list_path):
        with open(list_path, "r") as file:
            img_files = file.readlines()
        img_dict = self.get_classes_nums(img_files)
        key_threshold = cfg.train_key_threshold if self.phase == 'train' else cfg.val_key_threshold
        #---筛选出数量大于500的键
        key_num = [key for key, value in img_dict.items() if value > key_threshold]
        # for key,value in img_dict.items():
        #     if key in key_num:
        #         print(key,value)

        new_lines=[]
        for line in tqdm(img_files):
            line=line.strip().split()
            keys=line[1:]
            for key in keys:
                #---如果当前帧有键属于数量较多的键
                if int(key) in key_num:
                    new_lines.append(line)
                    break

        return new_lines, img_dict, key_num

    def __getitem__(self, index):
        lines = self.img_files[index % len(self.img_files)]
        img_path = lines[0]
        keys = lines[1:]
        targets = np.zeros(self.num_classes)
        for index in keys:
            pressed=int(index)
            if pressed in self.key_num:
                #---对应到key_num中的位置
                idx = self.key_num.index(pressed)
                targets[idx]=1

        # Extract image as PyTorch tensor
        img=Image.open(img_path).convert('RGB')
        file_seq = os.path.basename(os.path.split(img_path)[0])
        w, h = self.img_size
        if self.transforms:
            img = img.resize((w, h + 5))
            #---random crop还是有用的，可以增加模型的鲁棒性0.0
            img = random_crop(img, (w, h))
            img = random_brightness(img)
        else:            
            img = img.resize((w, h))
        img = transforms.ToTensor()(img)
        return img_path,img, targets

    def __len__(self):
        return len(self.img_files)

    def get_classes_nums(self,img_files):
        key_dic={}
        for i in range(12):
            key_dic[i] = 0

        key_num=0
        for line in img_files:
            line = line.strip().split()
            keys = line[1:]
            for key in keys:
                key_dic[int(key)]+=1
                
        key_lists=[]
        for i in range(12):
            if key_dic[i] == 0: continue
            key_lists.append((i,key_dic[i]))
            key_num+=key_dic[i]

        # key_lists=sorted(key_lists,key=lambda x:(x[1]))
        print('the training phase is {}'.format(self.phase))
        # for key in key_lists:
            # print(key)
        print('the total num is {}'.format(key_num))
        return key_dic

#---得到当前帧前后的连续几帧---
def get_continue_path(path, begin_frame, end_frame, cur_frame, continuous_frame):
    sequence_imgs = []
    #---对于一开始的几帧----
    if cur_frame - begin_frame < continuous_frame:
        temp_frame = cur_frame
        #---将前面的几帧补上
        while True:
            if temp_frame - 1 >= begin_frame:
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
    #---对于一般的帧,加上其前后几帧
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

class WBDataset(Dataset):
    def __init__(self, list_path, img_size=(224, 165), k=5, mode='white', phase='train'):
        self.phase = phase
        self.mode = mode
        self.nums = cfg.octave_w if mode == 'white' else cfg.octave_b
        self.k = k  #---输入连续k帧
        self.continuous_frame = int((self.k - 1) / 2)

        self.img_files, self.img_dict, self.key_num = self.get_imgs(list_path)
        self.num_classes = len(self.key_num) #12
        print('the key num is {}'.format(self.key_num))
        # print(len(self.key_num))

        self.img_size = img_size
        self.transforms=True if phase=='train' else False

    def get_imgs(self,list_path):
        with open(list_path, "r") as file:
            img_files = file.readlines()
        img_dict = self.get_classes_nums(img_files)
        
        #---筛选出数量大于500的键
        key_num = [key for key, value in img_dict.items() if key in self.nums]

        for key,value in img_dict.items():
            if key in key_num:
                print(key, value)
        img_lists = []

        for line in tqdm(img_files):
            line=line.strip().split()
            keys = line[1:]
            path = line[0]
            dir_name = os.path.dirname(path).split('/')[-1]
            file_seq = os.path.dirname(path).split('/')[-3]
            begin_frame = cfg.EVALUATE_MAP[file_seq]['begin_frame']
            end_frame = cfg.EVALUATE_MAP[file_seq]['end_frame']
            path = path.replace(dir_name, 'total_path')
            cur_img = cv2.imread(path)
            self.continuous_frame = int((self.k - 1) / 2)
            cur_frame = int(os.path.basename(path).split('.')[0].split('_')[0])
            #---对于刚开始的几帧，其之前没有对应帧
            sequence_imgs = get_continue_path(path, begin_frame,end_frame, cur_frame, self.continuous_frame)
            seq_imgs = (sequence_imgs, keys)
            for key in keys:
                #---如果当前帧的按键为白/黑键
                if int(key) in key_num:
                    img_lists.append(seq_imgs)
                    break

        return img_lists, img_dict, key_num

    def __getitem__(self, index):
        lines = self.img_files[index % len(self.img_files)]
        img_paths = lines[0]
        keys = lines[1]
        targets = np.zeros(self.num_classes)
        for index in keys:
            pressed = int(index)
            if pressed in self.key_num:
                #---对应到key_num中的位置
                idx = self.key_num.index(pressed)
                targets[idx]=1

        img_lists = []
        w, h = self.img_size
        for i, img_path in enumerate(img_paths):
            if '9999' in img_path:
                tmp_img = cv2.imread(img_paths[self.continuous_frame])
                #---这里的tmp_img的h/w不能和上面self.img_size的相同
                tmp_h, tmp_w, tmp_c = tmp_img.shape
                img = np.random.randint(128, 129, (tmp_h, tmp_w, tmp_c)).astype(np.uint8)
                img = Image.fromarray(img).convert('RGB')
            else:
                img = Image.open(img_path).convert('RGB')

            if self.phase == 'train':
                img = img.resize((w, h + 5))
                img = random_crop(img, (w, h))
                img = random_brightness(img)
            else:
                img = img.resize((w, h))
            img = transforms.ToTensor()(img)
            img_lists.append(img)
        return img_paths, img_lists, targets

    def __len__(self):
        return len(self.img_files)

    def get_classes_nums(self,img_files):
        key_dic={}
        for i in range(12):
            key_dic[i] = 0

        key_num=0
        for line in img_files:
            line = line.strip().split()
            keys = line[1:]
            for key in keys:
                key_dic[int(key)]+=1
                
        key_lists=[]
        for i in range(12):
            if key_dic[i] == 0: continue
            key_lists.append((i,key_dic[i]))
            key_num+=key_dic[i]

        # key_lists=sorted(key_lists,key=lambda x:(x[1]))
        print('the training phase is {}'.format(self.phase))
        # for key in key_lists:
            # print(key)
        print('the total num is {}'.format(key_num))
        return key_dic

if __name__ == '__main__':
    val_set = WBDataset(cfg.octave_val_list, img_size=cfg.octave_final_size,
                                 k=5, mode='black', phase='val')
    train_set = WBDataset(cfg.octave_train_list, img_size=cfg.octave_final_size,
                                 k=5, mode='black', phase='train')
    img_paths, img_lists, targets = val_set.__getitem__(2)

