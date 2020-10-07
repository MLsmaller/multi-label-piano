#-*- coding:utf-8 -*-
import math
import os
import mido
import cv2
import argparse
import numpy as np
import sys
sys.path.append('../')

from config import cfg
from IPython import embed

def parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label_path',type=str,
                        help='the label.txt convert to Note.txt')
    parser.add_argument('--midi_path',type=str,
                        help='the midi convert to Note.txt')                        
    args=parser.parse_args()
    return args

def get_fps(video_path):
    capture=cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError('read video wrong')
    fps=capture.get(cv2.CAP_PROP_FPS)
    return int(fps)

def Rounding(frame):
    Decimal, Integer = math.modf(frame)
    res = math.ceil(frame) if Decimal - 0.5 > 0 else math.floor(frame)
    return int(res)
        

#---将自己标的数据转换为note数据,level_4_no_01
def ToNote(txt_path):
    with open(txt_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    
    file_seq=os.path.basename(txt_path).split('_l')[0]
    label_path=txt_path.replace('_label','_note')
    fout=open(label_path,'w')

    if file_seq in cfg.EVALUATE_MAP:
        fps=cfg.EVALUATE_MAP[file_seq]['fps']
    else:
        video_path=os.path.join(os.path.split(txt_path)[0].replace('labels','videos'),
                                file_seq+'.mp4')
        fps=get_fps(video_path)
    pframe_time=1.0/fps

    filenames = []
    times = []
    keys = []        
    for i, line in enumerate(lines):
        key = []
        line = line.strip().split()
        filenames.append(line[0])
        frame = int(os.path.basename(line[0]).split('.')[0])
        times.append(float(frame*pframe_time))
        for j in range(1, len(line)):
            key.append(int(line[j]))
        keys.append(key)

    pro_onset = []

    #---如果第一帧就有键被按下，记录下来，后面是从第二帧开始的
    if not keys[0][0]==0:
        end_frame=0
        for j in range(1,len(times)):
            if keys[0] in keys[j]:
                end_frame = j 
            else:break 
        data = [int(times[0]/pframe_time),times[0], int(times[end_frame]/pframe_time),
                    times[end_frame],keys[0]] 
        pro_onset.append(data)
    

    for i in range(1,len(times)):
        current_keys = set(keys[i])
        last_keys = set(keys[i - 1])
        #---当前帧比上一帧多检出来的键(不包括0)
        difSet = [x for x in list(current_keys.symmetric_difference(last_keys)) if not (x == 0 or x in last_keys)]
        if (len(difSet) > 0):
            difSet = sorted(difSet)
            
            for pressed_key in difSet:
                count = 0
                end_frame = i+1
                #---看该按键持续了多少帧被按下
                for j in range(i+1,len(times)):
                    if pressed_key in keys[j]:
                        count+=1
                        end_frame = j 
                    else:break 
                if count==0:  #--只在当前帧被按下  起始帧 、起始时间、结束帧、结束时间、按键
                    data = [Rounding(times[i]/pframe_time),times[i], Rounding(times[end_frame]/pframe_time),
                            times[end_frame],pressed_key]
                else :
                    data = [Rounding(times[i]/pframe_time),times[i], Rounding(times[end_frame]/pframe_time),
                            times[end_frame], pressed_key]
                pro_onset.append(data)

    if len(pro_onset) > 0:
        pro_onset = sorted(pro_onset, key=lambda x: (x[1], x[-1]))
        for onset in pro_onset:
            data='frame{:0>4d}\t{:.2f}\tframe{:0>4d}\t{:.2f}\t{}\n'.format(onset[0],onset[1],onset[2],
                                                       onset[3],onset[4])
            fout.write(data)
    fout.close()
    return pro_onset        

#---给没有被按下的帧补0，方便处理
def Pad_zero(txt_path):
    with open(txt_path,'r',encoding='utf-8') as f:
        lines=f.readlines()
        new_line=[]
        for line in lines:
            line=line.strip().split()
            if len(line)==1:
                line.append(0)
            new_line.append(line)

    fout=open(txt_path,'w')
    for line in new_line:
        for data in line:
            fout.write('{} '.format(data))
        fout.write('\n')
    fout.close()
    

def processMidi(midiPath, fps):
    mid = mido.MidiFile(midiPath)
    file_seq=os.path.basename(midiPath).split('.')[0]
    midi_offset=cfg.EVALUATE_MAP[file_seq]['midi_offset']

    timeCount = 0
    dataList = []

    for msg in mid:
        if not msg.is_meta:
            if msg.type == 'control_change':
                timeCount = timeCount + msg.time
            elif msg.type == 'note_on' or msg.type == 'note_off':
                timeCount = timeCount + msg.time
                data = [msg.type, msg.note - 20, msg.velocity, timeCount]
                # print('the frame is {}'.format(timeCount*fps))
                # print(data)    
                # embed()            
                dataList.append(data)
    # print(dataList)
    dict1 = {}
    result = []
    for data in dataList:
        if data[0] == 'note_on' and data[2] > 0:
            dict1[data[1]] = data[1:]
        else:
            #---noteoff中后面的时间应该是当前时间啊，减掉对应的noteon中的时间才是持续时间
            dict1[data[1]].append(data[3])
            result.append(dict1.pop(data[1]))
    #---result->(按键,按下速度,按键起始时间,按键结束时间)
    result = sorted(result, key = lambda x : x[2])
    pitch_onset = []
    for item in result:
        #- midi_offset相当于对应到视频帧中第一个按下键的位置去了
        po = [item[2] - midi_offset , item[0]]  #---按下的时间/按键
        pitch_onset.append(po)
    #---时间和按键，没有包括结束时间
    pitch_onset = sorted(pitch_onset, key=lambda x: (x[0], x[1]))
    #print(pitch_onset) 
    pitch_onset_offset = []
    for item in result:
        #--起始时间/结束时间/按键
        po = [item[2] - midi_offset, item[3] - midi_offset,item[0]]
        pitch_onset_offset.append(po)
    pitch_onset_offset = sorted(pitch_onset_offset,key=lambda x:(x[0],x[1],x[2]))
    return pitch_onset,pitch_onset_offset


#---将midi信息转换为note和pitch信息
def save_midi(midiPath):
    file_seq=os.path.basename(midiPath).split('.')[0]
    fps=cfg.EVALUATE_MAP[file_seq]['fps']
    pframe_time=1.0/fps

    _,pitch_onset_offset=processMidi(midiPath,fps)
    label_save_path = os.path.join(os.path.dirname(midiPath), 'labels')
    if not os.path.exists(label_save_path): os.makedirs(label_save_path)
    print(label_save_path)
    note_path=os.path.join(label_save_path,file_seq+'_note.txt')
    label_path=os.path.join(label_save_path,file_seq+'_label.txt')
    with open(note_path,'w') as fout:
        for i,po in enumerate(pitch_onset_offset):
            # end_pitch
            p1=po[0]/pframe_time
            p2=po[1]/pframe_time
            if p1 <0 or p2<0:continue
            count_frame = int(np.ceil(p1)) if math.modf(p1)[0]>=0.2 else int(np.floor(p1))
            # end_frame = int(np.ceil(p2)) if math.modf(p2)[0]>=0.5 else int(np.floor(p2))
            end_frame = int(np.ceil(p2))
            data = 'frame{:0>4d}\t{:.2f}\tframe{:0>4d}\t{:.2f}\t{}\n'.format(count_frame,po[0],end_frame,po[1],po[2])
            fout.write(data)
    images_path = os.path.dirname(midiPath).replace('midi', 'images')
    img_path = os.path.join(images_path, file_seq)
    img_lists=[os.path.join(img_path,x) for x in os.listdir(img_path)
               if x.endswith('.jpg')]
    img_lists.sort()
    pitch_dic={}
    for i in range(len(img_lists)):
        pitch_dic[i]=[]

    with open(label_path,'w') as fout:
        for i,po in enumerate(pitch_onset_offset):
            # end_pitch
            p1=po[0]/pframe_time
            p2=po[1]/pframe_time
            if p1 <-0.9 or p2<0:continue
            if p1<0:
                count_frame=0   
            else:
                count_frame = int(np.ceil(p1)) if math.modf(p1)[0]>=0.2 else int(np.floor(p1))
            end_frame = min(int(np.ceil(p2)),len(img_lists)-1)
            for j in range(count_frame,end_frame+1):
                pitch_dic[j].append(po[2])
                

        for i,path in enumerate(img_lists):
            fout.write('{} '.format(path))
            if len(pitch_dic[i])==0:
                fout.write('{} '.format(0))
            else:
                for key in pitch_dic[i]:
                    fout.write('{} '.format(key))
            fout.write('\n')

if __name__=='__main__':
    args = parser()
    args.label_path='/home/lj/cy/data/piano/new/videos/Tencent/labels'
    if args.label_path is not None:
        label_lists=[os.path.join(args.label_path,x) for x in os.listdir(args.label_path)
                     if 'label' in x]
        label_lists.sort()
        for path in label_lists:
            if not 'level_1_no_02' in path: continue
            print(path)
            # Pad_zero(path)            
            ToNote(path)
    
    #--这个midi_offset还不一定一样.0
    # args.midi_path = '/home/lj/cy/data/piano/new/videos/Record/midi'
    if args.midi_path is not None:
        midi_lists=[os.path.join(args.midi_path,x) for x in os.listdir(args.midi_path)
                     if 'mid' in x or 'MID' in x]
        midi_lists.sort()
        for path in midi_lists:
            # if not '/27.mid' in path:continue
            print(path)
            save_midi(path)      
    
    # video_path = '/home/lj/cy/data/piano/new/videos/Record'
    # midi_path = os.path.join(video_path, 'midi')
    # videos = [os.path.basename(x).split('.')[0] for x in os.listdir(
    #           video_path) if x.endswith('.mp4')]
    # midi_file = [os.path.join(midi_path, x) for x in os.listdir(midi_path)
    #              if x.endswith('.MID')]
    # for midi in midi_file:
    #     file_seq = os.path.basename(midi).split('.')[0]
    #     if not file_seq in videos:
    #         os.remove(midi)

    