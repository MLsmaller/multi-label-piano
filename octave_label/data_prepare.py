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

#---将网上数据集也进行处理一下
# def process_youtube_data(rect, img_path):
#     img_list = [os.path.join(img_path, x) for x in os.listdir(img_path)
#                 if x.endswith('.jpg')]
#     img = cv2.imread(img_list[0])
#     def post_process(img):
#         h, w, _ = img.shape
#         lt, rt, rb, lb = rect
#         embed()
#         if abs(lt[1]-rt[1])>5 or abs(rb[1]-lb[1])>5:
#             xb1,yb1,xb2,yb2 = lb[0],lb[1],rb[0],rb[1]
#             xt1,yt1,xt2,yt2 = lt[0],lt[1],rt[0],rt[1]
#             center = (w//2,h//2)
#             if abs(yb1-yb2)>abs(yt1-yt2):
#                 angle = calAngle(xb1,yb1,xb2,yb2)
#                 M = cv2.getRotationMatrix2D(center,angle,1)
#                 rotated_img = cv2.warpAffine(img,M,(w,h))
#             else:
#                 angle = calAngle(xt1,yt1,xt2,yt2)
#                 M = cv2.getRotationMatrix2D(center,angle,1)
#                 rotated_img = cv2.warpAffine(img,M,(w,h))
#             result = {'flag':1,'rote_M':M,'warp_M':None,'keyboard_rect':None,
#                     'rotated_img':rotated_img 
#                 }
#         else:
#             result = {'flag':False,'rote_M':None,'warp_M':None,'keyboard_rect':rect,
#                     'rotated_img':None 
#                 }
#         return result

#     def post_process1(img):
#         h, w, _ = img.shape
#         lt, rt, rb, lb = rect
#         if abs(lt[1]-rt[1])>5 or abs(rb[1]-lb[1])>5:
#             xb1,yb1,xb2,yb2 = lb[0],lb[1],rb[0],rb[1]
#             xt1,yt1,xt2,yt2 = lt[0],lt[1],rt[0],rt[1]
#             if abs(yb1-yb2)>abs(yt1-yt2):
#                 pts1 = np.float32([lt,lb,rt,rb])
#                 if yb1>yb2:
#                     pts2 = np.float32([lt,lb,rt,[rb[0],lb[1]]])
#                 else:
#                     pts2 = np.float32([lt,[lb[0],rb[1]],rt,rb])
#                 M=cv2.getPerspectiveTransform(pts1,pts2)
#                 warp_img=cv2.warpPerspective(img,M,(w,h))
#             else:
#                 pts1 = np.float32([lt,lb,rt,rb])
#                 if yt1<yt2:
#                     pts2 = np.float32([lt,lb,[rt[0],lt[1]],rb])
#                 else:
#                     pts2 = np.float32([[lt[0],rt[1]],lb,rt,rb])
#                 M=cv2.getPerspectiveTransform(pts1,pts2)
#                 warp_img=cv2.warpPerspective(img,M,(w,h))
#             result = {
#                     'flag':1,
#                     'warp_M':M,
#                     'keyboard_rect':None,
#                     'warp_img':warp_img
#             }
#         else:
#             result = {
#                     'flag':False,
#                     'warp_M':None,
#                     'keyboard_rect': rect,
#                     'warp_img':None 
#             }   
#     result = post_process(img)
#     if not result['flag']: return result
#     rotated_img = result['rotated_img']
#     warp_result = post_process1(rotated_img)
#     warp_result['rotated_img'] = rotated_img 
#     warp_result['rote_M'] = result['rote_M']
#     if not warp_result['flag']:return warp_result
    

def get_octave_img(img_save_path, img_path, VisAmt, mode='notime'):
    # white_loc,black_boxes=get_key_nums(img_path)
    white_loc, black_boxes, base_info = VisAmt.process_img_dir(img_path)
    file_seq = os.path.basename(img_path)
    base_img_dir, img_dirs, test_dirs, save_dirs = init_save_file_dir(img_save_path, file_seq)

    label_path = cfg.EVALUATE_MAP[file_seq]['label_path']
    rect = cfg.EVALUATE_MAP[file_seq]['keyboard_rect'] if base_info is None else base_info['rect']

    with open(label_path,'r') as f:
        lines=f.readlines()
    
    black_num=cfg.black_num
    white_num = cfg.white_num
    if 'level' in img_path:
        if base_info['rote_M'] is not None:
            print('{} the rotated img  {}'.format('-' * 10, img_path))

    pos_img_dict = {}
    neg_img_dict = {}
    Tostop = 0
    for line in tqdm(lines):
        Tostop += 1
        # if Tostop > 200:  #--for test
            # break
        line = line.strip().split()
        if not mode=='time':
            if int(line[1]) == 0: continue
        path = line[0]
        img = cv2.imread(path)
        h, w, c = img.shape
        #---将有旋转矩阵和透视变化的图像要进行处理,正视角
        if 'level' in img_path:
            if base_info['rote_M'] is None:
                img = img.copy()
            elif base_info['warp_M'] is None:
                rotated_img = cv2.warpAffine(img, base_info['rote_M'], (w, h))
                img = rotated_img.copy()
            else:
                rotated_img = cv2.warpAffine(img, base_info['rote_M'], (w, h))
                warp_img = cv2.warpPerspective(rotated_img, base_info['warp_M'], (w, h))
                img = warp_img.copy()
            
        crop_img = img[rect[1]:rect[3], rect[0]:rect[2]]
        h, w, _ = crop_img.shape
        file_mark = os.path.basename(path).split('.')[0]

        l_offset, r_offset = cfg.l_offset, cfg.r_offset
        for idx in range(0, 7):
            if 'level' in img_path:
                new_r_offset = r_offset + 1 if idx > 3 else r_offset
            else:
                new_r_offset = r_offset + 3 if idx > 3 else r_offset
            new_l_offset = l_offset - 1 if idx > 3 else l_offset
            
            key_indexs = []
            begin = 2 + idx * 7
            end = 9 + idx * 7
            octave_img = crop_img[:, np.maximum(white_loc[begin] - new_l_offset,0):
                                np.minimum(white_loc[end] + new_r_offset, w)]
            if not mode=='time':
                key_range = cfg.octave[str(idx)]
                save_path = os.path.join(save_dirs[-1], 'neg_path',
                                        '{}_{}.jpg'.format(file_mark, idx))            
                for press_key in line[1:]:
                    key = int(press_key)
                    if key in key_range:
                        save_path = save_path.replace('neg_path', 'pos_path')
                        key_indexs.append(key)

                cv2.imwrite(save_path, octave_img)
                if len(key_indexs)>0:
                    pos_img_dict[save_path] = []
                    for key in key_indexs:
                        pos_img_dict[save_path].append(key)
            #---当要生成连续几帧图像时，需要将图像所有帧都存进来
            else:
                save_path = os.path.join(save_dirs[-1], 'total_path',
                        '{}_{}.jpg'.format(file_mark, idx))
                cv2.imwrite(save_path, octave_img)

    return pos_img_dict


#--将训练数据写入txt
def path_To_txt(pos_img_lists):
    train_ratio = 0.8
    def helper(pos_img_lists, train_out, val_out):
        pos_num = 0
        total_pos_img_lists = []
        for img_lists in pos_img_lists:
            pos_num += len(img_lists)
            for key, value in img_lists.items():
                labels = []
                for val in value:
                    #----将按键转换为0-11的12分类
                    label = (val - 4) % 12
                    labels.append(label)
                total_pos_img_lists.append((key, labels))
                 
        train_list = random.sample(total_pos_img_lists, int(pos_num * train_ratio))
        val_list = [x for x in total_pos_img_lists if x not in train_list]
        random.shuffle(val_list)

        for info in train_list:
            train_out.write('{}'.format(info[0]))
            for key in info[1]:
                train_out.write(' {}'.format(key))
            train_out.write('\n')

        for info in val_list:
            val_out.write('{}'.format(info[0]))
            for key in info[1]:
                val_out.write(' {}'.format(key))
            val_out.write('\n')            

        print('the number of training img is {}'.format(pos_num))

    cfg.binary_path = cfg.binary_path.replace('multi_label', 'octave_label')
    train_out = open(os.path.join(cfg.binary_path, 'octave_train.txt'), 'w')
    val_out = open(os.path.join(cfg.binary_path, 'octave_val.txt'), 'w')
    
    helper(pos_img_lists, train_out, val_out)
    train_out.close()
    val_out.close()

#---生成用以训练单个按键是否被按下的数据(二分类)
def from_train_data(img_paths):
    pos_img_lists = []
    VisAmt = VisAmtHelper()
    VisAmt.init_model_load()
    for img_path in img_paths:
        # if not os.path.basename(img_path) in ['level_1_no_05']:continue
        
        if not os.path.basename(img_path) in cfg.EVALUATE_MAP.keys(): continue
        file_seq = os.path.basename(img_path)
        img_save_path = cfg.Test_Key_Dir if file_seq in cfg.Test_video else cfg.One_key_SAVE_IMG_DIR
        exist_path = os.path.join(img_save_path, file_seq, 'octave_path', 'pos_path')
        if file_seq in cfg.Test_video: continue  #---对于测试集不包含训练集当中
        #---有图像说明已经存在了数据
        # imglists = [os.path.join(exist_path, x) for x in os.listdir(exist_path)
        #             if x.endswith('.jpg')]
        # if not len(imglists) > 0:    #---如果训练数据已经生成了则跳过
            #---生成训练数据
            # get_train_img(img_save_path, img_path, VisAmt)
        print(img_path)
        pos_img_dict = get_octave_img(img_save_path, img_path, VisAmt)
        
        pos_img_lists.append(pos_img_dict)
        
    path_To_txt(pos_img_lists)
        
#---用以生成带时间信息的训练数据(连续3/5帧)
def from_train_time_data(total_img_path):
    pos_img_lists = []
    VisAmt = VisAmtHelper()
    VisAmt.init_model_load()
    for img_path in total_img_path:
        # if not 'level_4_no_02' in img_path: continue
        if not os.path.basename(img_path) in cfg.EVALUATE_MAP.keys(): continue
        file_seq = os.path.basename(img_path)
        img_save_path = cfg.Test_Key_Dir if file_seq in cfg.Test_video else cfg.One_key_SAVE_IMG_DIR
        # if file_seq in cfg.Test_video: continue  #---对于测试集不包含训练集当中

        print(img_path)
        pos_img_dict = get_octave_img(img_save_path, img_path, VisAmt, mode='time')

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
    # from_train_time_data(total_img_path)

    