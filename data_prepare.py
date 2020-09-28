#-*- coding:utf-8 -*-
import os
import numpy as np
import random
from tqdm import tqdm

from utils.utils import *
from config import cfg
from IPython import embed

def write_txt(path,dic):
    fout=open(path,'w')
    for key,value in dic.items():
        fout.write('{} '.format(key))
        for item in value:
            fout.write('{} '.format(item))
        fout.write('\n')
    fout.close()

#---将数据分为训练集和验证集(用以训练88个类的多标签分类)
def main():
    img_paths=[]
    keys=[]
    dic_train={}
    dic_val={}
    total_lines=[]
    for key,value in cfg.EVALUATE_MAP.items():
        label_path=value['label_path']
        print(label_path)
        with open(label_path,'r') as f:
            lines=f.readlines()
        for line in lines:
            line=line.strip().split()
            total_lines.append(line)
    
    #---shuffle
    random.shuffle(total_lines)
    for line in total_lines:
        if int(line[1])==0:continue
        if np.random.rand()>0.2:
            dic_train[line[0]]=line[1:]
        else:
            dic_val[line[0]]=line[1:]
        img_paths.append(line[0])
        keys.append(line[1:])
    train_path=cfg.train_txt_path
    val_path=train_path.replace('train','val')
    write_txt(train_path,dic_train)
    write_txt(val_path,dic_val)


def save_neg_data(path,x1,y1,x2,y2,index,h_idx,rect,save_path,label_path):
    file_mark = int(os.path.basename(path).split('.')[0])
    note_path=label_path.replace('_label','_note')
    new_lines=[]
    with open(label_path,'r') as f:
        ori_lines=f.readlines()
        for line in ori_lines:
            new_lines.append(line.strip().split())

    To_stop=len(new_lines)
    with open(note_path,'r') as f:
        lines=f.readlines()
    negframe_offset=5

    #---相当于就是说有针对性的选取负样本，将某个键按下时其前后几帧没按下的状态当做负样本
    for line in lines:
        line=line.strip().split()
        begin_mark=int(line[0].split('frame')[1])
        end_mark=int(line[2].split('frame')[1])
        cur_key=int(line[-1])
        if begin_mark==file_mark and cur_key==index:
            #---将按下帧前面的几帧对应按键当做负样本
            if begin_mark-negframe_offset>0:
                for i in range(begin_mark-negframe_offset,begin_mark):
                    before_flag=True
                    file_seq='{:0>4d}'.format(i)
                    img_path=path.replace('{:0>4d}'.format(file_mark),file_seq)
                    #---确保其前后几帧对应的键是没有被按下的
                    for cur_line in new_lines:
                        line_keys=cur_line[1:]
                        if cur_line[0]==img_path and str(cur_key) in line_keys: 
                            before_flag=False
                            break
                    if before_flag:
                        img=cv2.imread(img_path)
                        crop_img=img[rect[1]:rect[3],rect[0]:rect[2]]
                        h,w,_=crop_img.shape    
                        save_img = crop_img[y1:y2,x1:x2]    
                        press_path=os.path.join(save_path,'{}_{}.jpg'.format(file_seq,index))                
                        # print(press_path)
                        cv2.imwrite(press_path,save_img)  
            
            #---将按下帧前面的几帧对应按键当做负样本
            if end_mark+negframe_offset<To_stop:
                for i in range(end_mark+1,end_mark+negframe_offset):
                    after_flag=True
                    file_seq='{:0>4d}'.format(i)
                    img_path=path.replace('{:0>4d}'.format(file_mark),file_seq)
                    #---确保其前后几帧对应的键是没有被按下的
                    for cur_line in new_lines:
                        line_keys=cur_line[1:]
                        if cur_line[0]==img_path and str(cur_key) in line_keys: 
                            after_flag=False
                            break
                    if after_flag:
                        img=cv2.imread(img_path)
                        crop_img=img[rect[1]:rect[3],rect[0]:rect[2]]
                        h,w,_=crop_img.shape    
                        save_img = crop_img[y1:y2,x1:x2]    
                        press_path=os.path.join(save_path,'{}_{}.jpg'.format(file_seq,index))                
                        # print(press_path)
                        cv2.imwrite(press_path,save_img)  
            # embed()

def b_save_other_neg_data(crop_img,black_boxes,b_index,
                        black_num,file_mark,save_path,h,w,b_offset):
    neg_index=[x for x in black_num if x not in b_index]
    num=0
    # random_index=random.sample(neg_index,10)
    for press_index in b_index:
        for index in neg_index:
            if not abs(index-press_index)<8:continue
            h_idx=black_num.index(index)
            bbox = black_boxes[h_idx]
            x1,y1,x2,y2 = bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]
            x1,y1,x2,y2 = max(0,x1-b_offset),max(0,y1-b_offset+1),min(w,x2+b_offset),min(h,y2+b_offset)
            save_img = crop_img[y1:y2,x1:x2]         
            press_path=os.path.join(save_path,'{}_{}.jpg'.format(file_mark,index))                
            # print(press_path)
            cv2.imwrite(press_path,save_img)
            num+=1
            if num>10:break
            
def w_save_other_neg_data(crop_img,white_loc,w_index,white_num,file_mark,
                          save_path,y_begin,h,w,w_offset):
    neg_index=[x for x in white_num if x not in w_index]
    num=0
    # random_index=random.sample(neg_index,20)
    for press_index in w_index:
        for index in neg_index:
            if not abs(index-press_index)<6:continue
            w_idx=white_num.index(index)
            start = max(int(white_loc[w_idx]-w_offset),0)
            end = min(int(white_loc[w_idx+1]+w_offset),w)                
            save_img = crop_img[y_begin:h,start:end]
            press_path=os.path.join(save_path,'{}_{}.jpg'.format(file_mark,index))                
            # print(press_path)
            cv2.imwrite(press_path,save_img)
            num+=1
            if num>10:break

def get_train_img(img_save_path, img_path, VisAmt):
    # white_loc,black_boxes=get_key_nums(img_path)
    white_loc, black_boxes, base_info = VisAmt.process_img_dir(img_path)
    file_seq = os.path.basename(img_path)
    base_img_dir, img_dirs, test_dirs = init_save_file_dir(img_save_path, file_seq)

    label_path = cfg.EVALUATE_MAP[file_seq]['label_path']
    rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect'] if base_info is None else base_info['rect']
    # rect=cfg.EVALUATE_MAP[file_seq]['keyboard_rect']
    with open(label_path,'r') as f:
        lines=f.readlines()
    
    black_num=cfg.black_num
    white_num = cfg.white_num

    offset = cfg.offset['level'] if len(file_seq) > 3 else cfg.offset['paper']
    w_offset = offset['w_offset']
    b_offset = offset['b_offset']
    y2_offset = offset['y2_offset']
    y_begin = offset['y_begin']

    for line in tqdm(lines):
        line=line.strip().split()   
        if int(line[1]) == 0: continue
        path=line[0]
        img = cv2.imread(path)
        crop_img=img[rect[1]:rect[3],rect[0]:rect[2]]
        h,w,_=crop_img.shape
        file_mark = os.path.basename(path).split('.')[0]
        # print(line)
        w_index=[]
        b_index=[]
        for index in line[1:]:
            index=int(index)
            if index in black_num:
                b_index.append(index)
            else:
                w_index.append(index)

        for index in b_index:
            h_idx=black_num.index(index)
            bbox = black_boxes[h_idx]
            x1,y1,x2,y2 = bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]
            x1,y1,x2,y2 = max(0,x1-b_offset),max(0,y1-b_offset+1),min(w,x2+b_offset),min(h,y2+b_offset+y2_offset)
            save_img = crop_img[y1:y2,x1:x2]    
            press_path=os.path.join(img_dirs[1],'{}_{}.jpg'.format(file_mark,index))                
            cv2.imwrite(press_path,save_img)
            save_neg_data(path,x1,y1,x2,y2,index,h_idx,rect,img_dirs[3],label_path)
        if len(b_index)>0:
            b_save_other_neg_data(crop_img,black_boxes,b_index,
                                black_num,file_mark,img_dirs[5],h,w,b_offset)

        for index in w_index:
            # y_begin=5
            w_idx=white_num.index(index)
            start = max(int(white_loc[w_idx]-w_offset),0)
            end = min(int(white_loc[w_idx+1]+w_offset),w)                
            save_img = crop_img[y_begin:h,start:end]
            press_path=os.path.join(img_dirs[0],'{}_{}.jpg'.format(file_mark,index))
            # print(press_path)
            cv2.imwrite(press_path,save_img)
            save_neg_data(path,start,y_begin,end,h,index,w_idx,rect,img_dirs[2],label_path)
        if len(w_index)>0:
            w_save_other_neg_data(crop_img,white_loc,w_index,white_num,
                                  file_mark,img_dirs[4],y_begin,h,w,w_offset)    

#---将路径中的图像存入lists中
def get_img_nums(img_dirs):
    #--白键和黑键的正样本
    pos_white_path=img_dirs[0]
    pos_black_path=img_dirs[1]
    pos_white_imgs=[os.path.join(pos_white_path,x) for x in os.listdir(pos_white_path)
                    if x.endswith('.jpg')]
    pos_black_imgs=[os.path.join(pos_black_path,x) for x in os.listdir(pos_black_path)
                    if x.endswith('.jpg')]   

    #---自己选择的该键被按下的帧的前后几帧(特定选取的负样本)
    neg_white_key_img=img_dirs[2]
    neg_black_key_img=img_dirs[3]
    neg_white_imgs=[os.path.join(neg_white_key_img,x) for x in os.listdir(neg_white_key_img)
                    if x.endswith('.jpg')]
    neg_black_imgs=[os.path.join(neg_black_key_img,x) for x in os.listdir(neg_black_key_img)
                    if x.endswith('.jpg')]  
             
    #--该键被按下的帧中的其他按键             
    neg_white_key_img1=img_dirs[4]
    neg_black_key_img1=img_dirs[5]
    neg_white_imgs1=[os.path.join(neg_white_key_img1,x) for x in os.listdir(neg_white_key_img1)
                    if x.endswith('.jpg')]
    neg_black_imgs1=[os.path.join(neg_black_key_img1,x) for x in os.listdir(neg_black_key_img1)
                    if x.endswith('.jpg')]  

    img_dirs=[pos_white_imgs,pos_black_imgs,neg_white_imgs,
              neg_black_imgs,neg_white_imgs1,neg_black_imgs1]    
    return img_dirs

#---去掉这两个list中重复的图片
def del_repeat_imgs(select_neg_white,other_neg_white):
    file_seq=[os.path.basename(x).split('.')[0] for x in other_neg_white]
    pass_lists=[]
    for img_path in select_neg_white:
        cur_seq=os.path.basename(img_path).split('.')[0]
        if cur_seq in file_seq:
            pass_lists.append(cur_seq) 
    new_select_neg_white=[x for x in select_neg_white if 
                          os.path.basename(x).split('.')[0] not in pass_lists]
    return new_select_neg_white,other_neg_white

#----得到负样本
def get_neg_imgs(total_neg_white_imgs,total_neg_white_imgs1,
                 total_neg_black_imgs,total_neg_black_imgs1):
    #---将neg_white_imgs中的70%取出来当负样本,从neg_white_imgs1中选取和neg_white_imgs一样数量的负样本
    select_neg_white=random.sample(total_neg_white_imgs,int(len(total_neg_white_imgs)*cfg.neg_img_selece_ratio))
    other_neg_white=random.sample(total_neg_white_imgs1,int(len(total_neg_white_imgs)*cfg.neg_img_selece_ratio))

    select_neg_black=random.sample(total_neg_black_imgs,int(len(total_neg_black_imgs)*cfg.neg_img_selece_ratio))
    other_neg_black=random.sample(total_neg_black_imgs1,int(len(total_neg_black_imgs)*cfg.neg_img_selece_ratio))
    
    #---去掉重复的元素
    select_neg_white,other_neg_white=del_repeat_imgs(select_neg_white,other_neg_white)
    select_neg_black,other_neg_black=del_repeat_imgs(select_neg_black,other_neg_black)

    neg_white_imgs=[]
    neg_black_imgs=[]
    neg_white_imgs.extend(select_neg_white)
    neg_white_imgs.extend(other_neg_white)
    neg_black_imgs.extend(select_neg_black)
    neg_black_imgs.extend(other_neg_black)

    return neg_white_imgs,neg_black_imgs

#--将训练数据写入txt
def path_To_txt(pos_white_imgs,pos_black_imgs,neg_white_imgs,neg_black_imgs):
    #---分成训练集和验证机吧兄弟0.0
    train_ratio=0.8
    def helper(pos_white_imgs,neg_white_imgs,w_train_out,w_val_out):
        train_pos_w_list=random.sample(pos_white_imgs,int(len(pos_white_imgs)*train_ratio))
        val_pos_w_list=[x for x in pos_white_imgs if x not in train_pos_w_list]

        train_neg_w_list=random.sample(neg_white_imgs,int(len(neg_white_imgs)*train_ratio))
        val_neg_w_list=[x for x in neg_white_imgs if x not in train_neg_w_list]    

        def alloc_label(pos_lists,neg_lists):
            total_lists=[]
            for path in pos_lists:
                path=(path,1)
                total_lists.append(path)
            for path in neg_lists:
                path=(path,0)
                total_lists.append(path)
            random.shuffle(total_lists)
            return total_lists

        train_lists = alloc_label(train_pos_w_list,train_neg_w_list)
        val_lists=alloc_label(val_pos_w_list,val_neg_w_list)
        for info in train_lists:
            data='{} {}\n'.format(info[0],info[1])
            w_train_out.write(data)
        for info in val_lists:
            data='{} {}\n'.format(info[0],info[1])
            w_val_out.write(data)

    w_train_out=open(os.path.join(cfg.binary_path,'white_train.txt'),'w')
    w_val_out=open(os.path.join(cfg.binary_path,'white_val.txt'),'w')
    
    b_train_out = open(os.path.join(cfg.binary_path,'black_train.txt'),'w')
    b_val_out=open(os.path.join(cfg.binary_path,'black_val.txt'),'w')

    helper(pos_white_imgs,neg_white_imgs,w_train_out,w_val_out)
    helper(pos_black_imgs,neg_black_imgs,b_train_out,b_val_out)


#---生成用以训练单个按键是否被按下的数据(二分类)
def from_train_data(img_paths):
    pos_white_imgs=[]
    pos_black_imgs=[]
    total_neg_white_imgs=[]
    total_neg_black_imgs=[]
    total_neg_white_imgs1=[]
    total_neg_black_imgs1=[]
    VisAmt = VisAmtHelper()
    VisAmt.init_model_load()
    for img_path in img_paths:
        # if not os.path.basename(img_path)=='level_4_no_01':continue
        
        if not os.path.basename(img_path) in cfg.EVALUATE_MAP.keys(): continue
        file_seq = os.path.basename(img_path)
        img_save_path = cfg.Test_Key_Dir if file_seq in cfg.Test_video else cfg.One_key_SAVE_IMG_DIR
        exist_path = os.path.join(img_save_path, file_seq, 'white_key')
        #---有图像说明已经存在了数据
        imglists = [os.path.join(exist_path, x) for x in os.listdir(exist_path)
                    if x.endswith('.jpg')]
        # if file_seq in cfg.Test_video: continue  #---对于测试集不包含训练集当中
        if not len(imglists) > 0:    #---如果训练数据已经生成了则跳过
            #---生成训练数据
            get_train_img(img_save_path, img_path, VisAmt)
        print(img_path)
        img_lists=[os.path.join(img_path,x) for x in os.listdir(img_path)
                if x.endswith('.jpg')]
        img_lists.sort()

        base_img_dir, img_dirs, test_dirs = init_save_file_dir(img_save_path, file_seq)
        imgs_info =get_img_nums (img_dirs)
        pos_white_imgs.extend(imgs_info[0])
        pos_black_imgs.extend(imgs_info[1])
        total_neg_white_imgs.extend(imgs_info[2])
        total_neg_black_imgs.extend(imgs_info[3])
        total_neg_white_imgs1.extend(imgs_info[4])
        total_neg_black_imgs1.extend(imgs_info[5])

    w_pos_nums=len(pos_white_imgs)
    b_pos_nums=len(pos_black_imgs)
    # w_neg_nums=w_pos_nums*cfg.neg_pos_ratio
    # b_neg_nums=b_pos_nums*cfg.neg_pos_ratio

    neg_white_imgs,neg_black_imgs=get_neg_imgs(total_neg_white_imgs,total_neg_white_imgs1,
                                               total_neg_black_imgs,total_neg_black_imgs1)
    path_To_txt(pos_white_imgs, pos_black_imgs, neg_white_imgs, neg_black_imgs)
    # embed()
        
#---用以生成带时间信息的训练数据(连续3/5帧)
def from_train_time_data(total_img_path):
    VisAmt = VisAmtHelper()
    VisAmt.init_model_load()
    for video_path in total_img_path:
        print(video_path)
        file_seq = os.path.basename(video_path)
        # if file_seq in cfg.Test_video: continue
        img_lists = [os.path.join(video_path, x) for x in os.listdir(video_path)
                    if x.endswith('.jpg')]
        img_lists.sort()

        save_path = cfg.Test_Key_Dir if file_seq in cfg.Test_video else cfg.One_key_SAVE_IMG_DIR
        #---用以训练时间信息的数据
        white_path = os.path.join(save_path, file_seq, 'test_white_key')
        img_nums = [os.path.join(white_path, x) for x in os.listdir(white_path)
                    if x.endswith('.jpg')]
        #---如果之前已经生成过数据就continue
        print(len(img_nums))
        if len(img_nums) > 0: continue

        white_loc, black_boxes,base_info = VisAmt.process_img_dir(video_path)
        base_img_dir, img_dirs, test_dirs = init_save_file_dir(save_path, file_seq)
        file_seq = os.path.basename(video_path)
        # if not file_seq == 'level_2_no_02': continue
        # rect = base_info['rect']
        rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect'] if base_info is None else base_info['rect']
        # print(video_path)
        
        offset = cfg.offset['level'] if len(file_seq) > 3 else cfg.offset['paper']
        w_offset = offset['w_offset']
        b_offset = offset['b_offset']
        y2_offset = offset['y2_offset']
        y_begin = offset['y_begin']


        for path in tqdm(img_lists):
            file_mark = os.path.basename(path).split('.')[0]
            # print(path)
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

if __name__=='__main__':
    # main()
    img_path=cfg.Tencent_path
    img_paths=[os.path.join(img_path,x) for x in os.listdir(img_path)]

    img_path1=cfg.SightToSound_paper_path
    img_path1=[os.path.join(img_path1,x) for x in os.listdir(img_path1)]

    total_img_path=[]
    total_img_path.extend(img_paths)
    total_img_path.extend(img_path1)
    total_img_path.sort()
                      
    # from_train_data(total_img_path)
    from_train_time_data(total_img_path)