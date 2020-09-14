import glob
import random
import os
import sys
sys.path.append('../')
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from config import cfg
from IPython import embed


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_crop(img):
    width,height=img.size
    left = np.random.uniform(0, 5)
    top= np.random.uniform(0, 5)

    # convert to integer rect x1,y1,x2,y2
    rect =[int(left), int(top), int(cfg.input_size[0]+left), int(cfg.input_size[1]+top)]
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
    def __init__(self, list_path, img_size=(800,145),phase='train'):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()
        self.img_size = img_size
        self.transforms=True if phase=='train' else False

    def __getitem__(self, index):
        lines = self.img_files[index % len(self.img_files)].strip().split()
        img_path=lines[0]
        keys=lines[1:]
        targets=np.zeros(88)
        for index in keys:
            #---标签中标的是从1-88，对应到数组中是0-87            
            targets[int(index)-1]=1

        # Extract image as PyTorch tensor
        img=Image.open(img_path).convert('RGB')
        file_seq=os.path.basename(os.path.split(img_path)[0])        
        if file_seq in cfg.crop_file_seq:
            rect=cfg.EVALUATE_MAP[file_seq]['rect']
            img=img.crop((rect))  #--img.size-> w,h
        
        img=img.resize((cfg.img_size))
        img=random_crop(img)
        if self.transforms:
            img=random_brightness(img)

        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        return img, targets

    def __len__(self):
        return len(self.img_files)

    def get_classes_nums(self):
        key_dic={}
        for i in range(88):
            key_dic[i]=0

        key_num=0
        for line in self.img_files:
            line = line.strip().split()
            keys=line[1:]
            for key in keys:
                key_dic[int(key)-1]+=1
        key_lists=[]
        for i in range(88):
            if key_dic[i]==0:continue
            # print('the key {} number is {}'.format(i+1,key_dic[i]))
            key_lists.append((i,key_dic[i]))
            key_num+=key_dic[i]
        key_lists=sorted(key_lists,key=lambda x:(x[1]))
        for key in key_lists:
            print(key)
        print('the total num is {}'.format(key_num))
        return key_dic

class KeyDataset_select(Dataset):
    def __init__(self, list_path, img_size=(800,145),phase='train'):
        self.phase=phase

        self.img_files,self.img_dict ,self.key_num=self.get_imgs(list_path)
        self.num_classes=len(self.key_num)  #--24
        #--[22, 24, 26, 27, 29, 31, 32, 33, 34, 36, 38, 39, 41, 43, 44,
        #-- 45, 46, 48, 50, 51, 53, 55, 57, 58]          
        print(self.key_num)
        # print(len(self.key_num))

        self.img_size = img_size
        self.transforms=True if phase=='train' else False

    def get_imgs(self,list_path):
        with open(list_path, "r") as file:
            img_files = file.readlines()
        img_dict=self.get_classes_nums(img_files)
        key_threshold=500 if self.phase=='train' else 100
        #---筛选出数量大于500的键
        key_num=[key for key,value in img_dict.items() if value>key_threshold]
        # for key,value in img_dict.items():
        #     if key in key_num:
        #         print(key,value)

        new_lines=[]
        for line in img_files:
            line=line.strip().split()
            keys=line[1:]
            for key in keys:
                #---如果当前帧有键属于数量较多的键
                if int(key)-1 in key_num:
                    new_lines.append(line)
                    break

        return new_lines,img_dict,key_num

    def __getitem__(self, index):
        lines = self.img_files[index % len(self.img_files)]
        img_path=lines[0]
        keys=lines[1:]
        targets=np.zeros(self.num_classes)
        for index in keys:
            pressed=int(index)-1
            if pressed in self.key_num:
                #---对应到key_num中的位置
                idx = self.key_num.index(pressed)
                targets[idx]=1

        # Extract image as PyTorch tensor
        img=Image.open(img_path).convert('RGB')
        file_seq=os.path.basename(os.path.split(img_path)[0])        
        if file_seq in cfg.crop_file_seq:
            rect=cfg.EVALUATE_MAP[file_seq]['rect']
            img=img.crop((rect))  #--img.size-> w,h
        

        #---训练集才有random crop/brightness
        if self.transforms:
            img=img.resize((cfg.img_size))
            #---random crop还是有用的，可以增加模型的鲁棒性0.0
            img=random_crop(img)            
            img=random_brightness(img)
        else:            
            img=img.resize(cfg.input_size)

        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))
        return img_path,img, targets

    def __len__(self):
        return len(self.img_files)

    def get_classes_nums(self,img_files):
        key_dic={}
        for i in range(88):
            key_dic[i]=0

        key_num=0
        for line in img_files:
            line = line.strip().split()
            keys=line[1:]
            for key in keys:
                key_dic[int(key)-1]+=1
                
        key_lists=[]
        for i in range(88):
            if key_dic[i]==0:continue
            key_lists.append((i,key_dic[i]))
            key_num+=key_dic[i]

        key_lists=sorted(key_lists,key=lambda x:(x[1]))
        for key in key_lists:
            print(key)
        print('the total num is {}'.format(key_num))
        return key_dic

if __name__=='__main__':
    # key_dataset=KeyDataset_select(cfg.train_txt_path)
    # key_dic=key_dataset.__getitem__(4)

    train_set = KeyDataset_select(cfg.train_txt_path, img_size=cfg.input_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # val_set = KeyDataset_select(cfg.val_txt_path, img_size=cfg.input_size,phase='val')
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_set,
    #     batch_size=64,
    #     shuffle=True,
    #     num_workers=8,
    #     pin_memory=True
    # )
