#-*- coding:utf-8 -*-
import os
import numpy as np
from easydict import EasyDict

_C=EasyDict()
cfg=_C

_C.black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
_C.white_num = [x for x in range(1, 89) if x not in cfg.black_num]

_C.SightToSound_paper_path='/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/images'
_C.Tencent_path='/home/lj/cy/data/piano/new/videos/Tencent/images'
_C.SAVE_IMG_DIR='/home/lj/cy/data/piano/new/saved/YouTube'

_C.label_save_path='/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper'

_C.base_img='/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/images/5/3732.jpg'

#----path
# _C.ckpt_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/keys_epoch_99_Fscore_0.965.pth'
# _C.ckpt_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/epoch_19.pth'
_C.ckpt_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/keys_epoch_44_Fscore_0.970.pth'

_C.keyboard_split_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/utils/output'

#---train config,编号是0-87
_C.labels= [22, 24, 26, 27, 29, 31, 32, 33, 34, 36, 38, 39, 
            41, 43, 44, 45, 46, 48, 50, 51, 53, 55, 57, 58]  
_C.res_txt_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/outputval_res.txt'

_C.top_k=5
_C.prob_thresh=0.4
_C.num_classes=24
_C.img_size=(805,150)
_C.input_size=(800,145)  #----wxh
_C.brightness_delta = 0.125
#--Loss weight for cross entropy loss.For example set $loss_weight to [1, 0.8, 0.8] for a 3 labels classification')
_C.loss_weight=np.ones(88)

_C.lr_mult_w=20
_C.lr_mult_b=20

_C.pretrained_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/resnet18.pth'

#---这些视频的背景帧是有手的
_C.file_loc=['10','25','level_4_no_02']
_C.black_white_loc={
        '10':{'black_boxes':[[43, 24, 30, 56],[106, 24, 30, 56],[150, 24, 29, 56],[214, 24, 28, 56],[255, 24, 28, 56],[297, 24, 28, 56],
                             [361, 24, 27, 56],[405, 24, 27, 56],[469, 24, 25, 56],[510, 24, 24, 56],[551, 24, 25, 56],[612, 24, 26, 56],
                             [655, 24, 25, 56],[718, 24, 23, 56]
                             ]},
        '21':{'black_boxes':[[32, 29, 34, 125],[96, 29, 33, 125],[139, 29, 34, 125],[203, 29, 32, 123],[245, 29, 33, 123],[286, 29, 31, 123],
                             [351, 29, 30, 122],[395, 29, 29, 121],[459, 29, 27, 120],[501, 29, 26, 119],[542, 29, 25, 119],[604, 29, 25, 118],
                             [647, 29, 25, 117],[710, 29, 22, 117],[750, 29, 23, 116],[790, 29, 22, 116],[850, 29, 22, 116],[893, 29, 24, 115],
                             [955, 29, 23, 115],[995, 29, 23, 115],[1035, 29, 23, 114],[1096, 29, 25, 114],[1138, 29, 26, 114],[1201, 29, 24, 114],
                             [1241, 29, 25, 114],[1282, 29, 25, 114],[1344, 29, 27, 114],[1387, 29, 28, 114],[1451, 29, 28, 114],[1492, 29, 30, 114],
                             [1534, 29, 30, 114],[1596, 29, 32, 114],[1640, 29, 32, 114],[1704, 29, 33, 114],[1745, 29, 34, 114],[1786, 29, 34, 114]
                            ],
                'white_loc':[  10,   46,   81,  117,  153,  189,  225,  261,  297,  333,  370,
                               406,  442,  478,  514,  549,  585,  620,  655,  690,  726,  761,
                               796,  832,  867,  902,  936,  971, 1006, 1041, 1076, 1112, 1147,
                               1182, 1218, 1253, 1288, 1324, 1363, 1399, 1435, 1471, 1507, 1542,
                               1578, 1618, 1654, 1690, 1726, 1762, 1797, 1833, 1864]}                        
}

_C.train_txt_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/data/train.txt'
_C.val_txt_path='/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/data/val.txt'


#---只将钢琴键盘区域送进网络中训练
_C.crop_file_seq=['level_2_no_01','level_4_no_01']
_C.EVALUATE_MAP = {
        ### Tencent video  第一个rect是包含键盘以及键盘下方的区域(用以训练数据)，keyboard_rect是只包含键盘的(用以连通算法得到黑键和白键的位置)
        #---end_frame即表示该帧以后的图像不再包括钢琴图像
        'level_2_no_01':{'base_frame':613,'end_frame':617,'label_path':'/home/lj/cy/data/piano/new/videos/Tencent/labels/level_2_no_01_label.txt','fps':25,'midi_offset':0,'rect':[28,497,1267,713],'keyboard_rect':[33,498,1258,639]},  #--x1,y1,x2,y2，
        'level_4_no_01': {'base_frame': 1674,'end_frame':1679, 'label_path': '/home/lj/cy/data/piano/new/videos/Tencent/labels/level_4_no_01_label.txt', 'fps': 25, 'midi_offset': 0, 'rect': [29, 514, 1254, 720], 'keyboard_rect': [31, 517, 1247, 663]},
        'level_2_no_02': {'base_frame': 857, 'end_frame':867,'label_path': '/home/lj/cy/data/piano/new/videos/Tencent/labels/level_2_no_02_label.txt', 'fps': 25, 'keyboard_rect': [36, 499, 1248, 635]},
        'level_4_no_02': {'base_frame': 900, 'end_frame':2070,'label_path': '/home/lj/cy/data/piano/new/videos/Tencent/level_4_no_02_label.txt', 'fps': 25,'keyboard_crop':120,'keyboard_rect':[37,521,1249,660]},
        
        '10':{'base_frame':564,'end_frame':3651,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/10_label.txt','fps':29.69,'midi_offset':0.457,'keyboard_crop':120,'keyboard_rect':[14,19,1862,240]}, #--x1,y1,x2,y2
        '21':{'base_frame':2623,'end_frame':2623,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/21_label.txt','fps':29.69,'midi_offset':5.153,'keyboard_crop':120,'keyboard_rect':[10,2,1865,218]}, 
        '23':{'base_frame':3620,'end_frame':3622,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/23_label.txt','fps':29.69,'midi_offset':0.2854,'keyboard_crop':120,'keyboard_rect':[14,25,1870,241]}, 
        '24':{'base_frame':3321,'end_frame':3325,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/24_label.txt','fps':29.69,'midi_offset':0.3192,'keyboard_crop':120,'keyboard_rect':[0,0,1865,221]},
        '25':{'base_frame':686,'end_frame':1959,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/25_label.txt','fps':29.69,'midi_offset':0.35297,'keyboard_crop':120,'keyboard_rect':[2,8,1870,227]}, 
        '26':{'base_frame':2103,'end_frame':2107,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/26_label.txt','fps':29.69,'midi_offset':0.4881,'keyboard_crop':120,'keyboard_rect':[0,7,1860,232]}, 
        '27':{'base_frame':2460,'end_frame':2464,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/27_label.txt','fps':29.69,'midi_offset':0.3868,'keyboard_crop':120,'keyboard_rect':[2,15,1839,234]}, 
        '5':{'base_frame':3732,'end_frame':3740,'label_path':'/home/lj/cy/data/piano/new/videos/Paper/SightToSound_paper/5_label.txt','fps':29.69,'midi_offset':0.2854,'keyboard_crop':120,'keyboard_rect':[12,16,1367,181]}        
}


#----------训练单个按键的二分类模型配置------------
_C.b_train_binary_txt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/data/binary_label/black_train.txt'
_C.b_val_binary_txt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/data/binary_label/black_val.txt'
_C.w_train_binary_txt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/data/binary_label/white_train.txt'
_C.w_val_binary_txt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/data/binary_label/white_val.txt'

#--ckpt path
_C.binary_ckpt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/binary_keys_epoch_25_Acc_0.976.pth'
_C.pos_binary_ckpt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/with_pos_keys_epoch_28_Acc_0.978.pth'
#----训练的时候要在tensorboard上面观察loss和Acc的变化,可能过拟合或者学习率过大,需要调整学习率
#---较好的模型应该是lr=0.001,然后lr_decay_in_epoch=20训练得到的
# _C.time_ckpt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/time_keys_epoch_25_Acc_0.985.pth'
#--这个是lr=0.01,然后lr_decay_in_epoch=5训练得到的,每隔5个epoch学习率下降
_C.time_ckpt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/time_keys_epoch_18_Acc_0.982.pth'
_C.time_pos_ckpt_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/checkpoints/time_with_pos_keys_epoch_25_Acc_0.981.pth'


#---二分类时的正负样本权重(正样本较少)
_C.ALPHA = {
        'white':3.0,
        'black':2.0
}
_C.pos_wieghts = [3, 1, 6]


_C.binary_input_size = [112, 32]   #---(h,w)

_C.neg_pos_ratio=2  #---负样本/正样本
_C.neg_img_selece_ratio=0.7  #---自己选取的负样本/当前帧其他的负样本

#---用以训练单个按键是否被按下
_C.One_key_SAVE_IMG_DIR = '/home/lj/cy/data/piano/new/saved/YouTube/train'
#---用以测试模型的视频图像存储路径
_C.Test_Key_Dir = '/home/lj/cy/data/piano/new/saved/YouTube/'
_C.Test_video = ['level_4_no_02', 'level_2_no_02']
#---label path(训练二分类)
_C.binary_path = '/home/lj/cy/project/piano/vision-piano-amt-master/backup/multi_label/data/binary_label'

#---按键小于27则是左边区域,27-57则是中间区域,>57则是右边区域
_C.split = [27, 57]


#----------训练时间信息的二分类模型配置------------
cfg.Consecutive_frames = 5
