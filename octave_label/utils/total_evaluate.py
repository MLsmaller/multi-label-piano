#-*- coding:utf-8 -*-
import cv2
import numpy
import argparse
import os
import shutil
import mido
from IPython import embed
import sys 
sys.path.append('../')
from config import cfg 
import time, math
import numpy as np

class Accuracy(object):
    def __init__(self, 
                midiPath=None, 
                detectPath=None, 
                pframe_time=0.04,
                start_frame = 0,
                midi_offset = 1.5,
                frameOffset=2, 
                tolerance=0.2):  #--即0.2s的误差，对于fps=20，则是(1/20)*4，相当于4帧的误差
                                  #--对于fps=25，则是(1/25)*5，相当于5帧的误差
        self.midiPath = midiPath    
        self.detectPath = detectPath
        self.pframe_time = pframe_time 
        self.offTime = start_frame * pframe_time

        self.frameOffset = frameOffset  #前面两帧没出现则认为是新的note
        self.tolerance = tolerance  #误差计算误差为2s
        self.tol_frame = 3  #相差3帧之内检测到认为是正确的
        self.midi_offset = midi_offset 

        self.black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                          31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                          60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
        self.white_num = [x for x in range(1, 89) if x not in self.black_num]

        self.run()
        print('the tolerance is {} and self.tol_frame is {}'.format(tolerance,self.tol_frame))
    
    def run(self):
        if self.midiPath is not None:
            if self.midiPath.endswith('txt'):
                #---for label txt
                self.pitch_onset = self.process_label_txt(self.midiPath,self.offTime,self.midi_offset) 
            else:
                self.pitch_onset = self.processMidi(self.midiPath,self.offTime,self.midi_offset)
            evalfile = os.path.join(os.path.dirname(self.detectPath),'evalresult.txt') 
            self.evalout = open(evalfile,'w')

            #---计算帧级的准确率
            self.evaluate_frame_precision()
            self.Total_precision()
            self.save_midi()
            self.pitch2note()
            self.evalout.close()
        else:
            self.b_pro_onset=self.processDetect1(self.b_detectPath,mode='black')
            self.pro_onset = self.processDetect1(self.detectPath) 
            self.pitch2note()

    def get_frame_result(self):
        result = {
                'black':{'precies':self.b_precies,'recall':self.b_recall,'F':self.b_F},
                'white':{'precies':self.w_precies,'recall':self.w_recall,'F':self.w_F}
        }
        return result 

    def get_note_result(self):
        result = {
                'black':{'precies':self.noteb_precies,'recall':self.noteb_recall,'F':self.noteb_F},
                'white':{'precies':self.notew_precies,'recall':self.notew_recall,'F':self.notew_F}
        }
        return result 

    #---用以验证自己标注的数据集
    def process_label_txt(self, midiPath, offTime=0, midi_offset=0):
        with open(midiPath,'r') as fr:
            items = fr.readlines()

        self.pitch_onset_offset = []
        pitch_onset = []
        for item in items:
            item = item.strip().split()
            po = [float(item[1]) - midi_offset + offTime, float(item[3]) - midi_offset + offTime, int(item[-1])]
            self.pitch_onset_offset.append(po)
            pitch_onset.append([float(item[1]) - midi_offset + offTime, int(item[-1])])
        
        pitch_onset = sorted(pitch_onset, key=lambda x: (x[0], x[1]))
        self.pitch_onset_offset = sorted(self.pitch_onset_offset, key=lambda x: (x[0], x[1], x[2]))
        return pitch_onset         

    def processMidi(self, midiPath, offTime, midi_offset):
        mid = mido.MidiFile(midiPath)
        timeCount = 0
        dataList = []
        for msg in mid:
            if not msg.is_meta:
                if msg.type == 'control_change':
                    timeCount = timeCount + msg.time
                elif msg.type == 'note_on' or msg.type == 'note_off':
                    timeCount = timeCount + msg.time
                    if 'wmv' in self.midiPath:
                        data = [msg.type, msg.note - 8, msg.velocity, timeCount]
                    else:
                        data = [msg.type, msg.note - 20, msg.velocity, timeCount]
                    dataList.append(data)
        dict1 = {}
        result = []
        for data in dataList:
            if data[0] == 'note_on' and data[2] > 0:
                dict1[data[1]] = data[1:]
            else:
                #---noteoff中后面的时间应该是当前时间啊，减掉对应的noteon中的时间才是持续时间啊
                dict1[data[1]].append(data[3])
                result.append(dict1.pop(data[1]))
        #---result->(按键,按下速度,按键起始时间,按键结束时间)
        result = sorted(result, key = lambda x : x[2])
        pitch_onset = []
        for item in result:
            #---+offTime相当于对应到视频帧中第一个按下键的位置去了
            po = [item[2] - midi_offset + offTime, item[0]]  #---按下的时间/按键
            pitch_onset.append(po)
        #---时间和按键，没有包括结束时间
        pitch_onset = sorted(pitch_onset, key=lambda x: (x[0], x[1]))
        #print(pitch_onset) 
        self.pitch_onset_offset = []
        for item in result:
            #--起始时间/结束时间/按键
            po = [item[2] - midi_offset + offTime, item[3] - midi_offset + offTime,item[0]]
            self.pitch_onset_offset.append(po)
        self.pitch_onset_offset = sorted(self.pitch_onset_offset,key=lambda x:(x[0],x[1],x[2]))
        return pitch_onset
    
    def save_midi(self):
        whitepath = os.path.join(os.path.dirname(self.detectPath),'midi_Total.txt')

        #---只记录按下的键/onset
        pitch_white = [[line[0],line[1]]
                       for line in self.pitch_onset]

        #---记录了onset和持续时间
        pitch_white_on_off = [[line[0],line[1],line[2]]
                for line in self.pitch_onset_offset]
    
        with open(whitepath,'w') as fout:
            for i,po in enumerate(pitch_white_on_off):
                # end_pitch
                p1 = po[0] / self.pframe_time
                p2 = po[1] / self.pframe_time
                if p1 <0 or p2<0:continue
                if p1<0:
                    count_frame=0   
                else:
                    count_frame = int(np.ceil(p1)) if math.modf(p1)[0]>=0.5 else int(np.floor(p1))
                end_frame = int(np.ceil(p2)) if math.modf(p2)[0]>=0.5 else int(np.floor(p2))
                dat = 'frame{} {:.5} frame{} {:.5} {}\n'.format(count_frame,po[0],end_frame,po[1],po[2])
                fout.write(dat)

    def pitch2note(self):
        whitepath = os.path.join(os.path.dirname(self.detectPath),'note_Total.txt')
        with open(whitepath,'w') as fout:
            for po in self.pro_onset:
                count_frame = int(po[0]/self.pframe_time)
                dat = 'frame{} {:.5} {:.5} {}\n'.format(count_frame,po[0],po[2],po[1])
                fout.write(dat)

    def processDetect(self, detectPath,mode='white'):
        with open(detectPath, 'r') as f:
            lines = f.readlines()
        filenames = []
        times = []
        keys = []
        for i, line in enumerate(lines):
            key = []
            line = line.strip().split()
            filenames.append(line[0])
            frame = int(os.path.basename(line[0]).split('.')[0])
            #times.append(float(line[1]))
            times.append(float(frame*self.pframe_time))
            for j in range(2, len(line)):
                key.append(int(line[j]))
            keys.append(key)

        pro_onset = []
        #-------------TIP------------------------------
        #--如果第一帧就按下的话这里是会漏掉的，要先加上第一帧的结果，后面是自己看第二针比第一帧多出的键

        #---从持续检测的帧中得到对应的onset
        for i in range(1,len(times)):
            current_keys = set(keys[i])
            last_keys = set(keys[i - 1])
            #---当前帧比上一帧多检出来的键(不包括0)
            difSet = [x for x in list(current_keys.symmetric_difference(last_keys)) if not (x == 0 or x in last_keys)]
            if (len(difSet) > 0):
                final_set = difSet[:]
                for key in difSet:
                    if (i - self.frameOffset) > 0:  
                        #--1
                        for m in range(2, self.frameOffset + 1):
                            if key in keys[i - m]:
                                #---去掉前m帧中出现的按键，认为隔了m帧出现的按键是新的按键
                                final_set = [x for x in final_set if x is not key]
                                break       
                    else:
                        for m in range(i):   #---对于视频的前m帧,只需要判断前面几帧是否有重复
                            if key in keys[m]:
                                final_set = [x for x in final_set if x is not key]
                                break

                final_set = sorted(final_set)
                for pressed_key in final_set:
                    count = 0
                    end_frame = i+1 
                    #---看该按键持续了多少帧被按下
                    for j in range(i+1,len(times)):
                        if pressed_key in keys[j]:
                            count+=1
                            end_frame = j 
                        else:break 
                    if mode=='white':
                        if count>=1:
                            #---记录了按键起始和结束时间
                            data = [times[i], pressed_key,times[end_frame]]
                            pro_onset.append(data)
                    else:
                        if count>=1:
                            data = [times[i], pressed_key,times[end_frame]]
                            pro_onset.append(data)
                            
        # if len(keys)>0 and len(pro_onset)>0:
        #     for note in keys[0]:
        #         count = 0
        #         end_frame = 1
        #         if note==0 or note==pro_onset[0][1]:continue
        #         for j in range(1,len(times)):
        #             if note in keys[j]:
        #                 count+=1
        #                 end_frame = j
        #             else:break 
        #         if count>=1:
        #             data = [times[0],note,times[end_frame]]
        #             pro_onset.append(data)
        if len(pro_onset) > 0:
            pro_onset = sorted(pro_onset, key=lambda x: (x[0], x[1],x[2]))
        return pro_onset

    def processDetect1(self,detectPath):
        with open(detectPath, 'r') as f:
            lines = f.readlines()
        filenames = []
        times = []
        keys = []
        for i, line in enumerate(lines):
            key = []
            line = line.strip().split()
            filenames.append(line[0])
            frame = int(os.path.basename(line[0]).split('.')[0])
            #times.append(float(line[1]))
            times.append(float(frame*self.pframe_time))
            for j in range(2, len(line)):
                key.append(int(line[j]))
            keys.append(key)

        pro_onset = []
        #---第一帧若检测到按键，则加进去0.0
        first_line = lines[0].strip().split()
        cur_key = int(first_line[-1])
        end_time = float(first_line[1])
        if cur_key != 0:
            for line in lines[1:]:
                line = line.strip().split()
                if int(line[-1]) == cur_key:
                    end_time = float(line[1])
                else: break
            data = [float(first_line[1]), cur_key, end_time]
            pro_onset.append(data)

        #---从持续检测的帧中得到对应的onset(第二帧及以后)
        for i in range(1,len(times)):
            current_keys = set(keys[i])
            last_keys = set(keys[i - 1])
            #---当前帧比上一帧多检出来的键(不包括0)
            difSet = [x for x in list(current_keys.symmetric_difference(last_keys)) if not (x == 0 or x in last_keys)]
            if (len(difSet) > 0):
                final_set = difSet[:]
                for key in difSet:
                    if (i - self.frameOffset) > 0:  
                        #--1
                        for m in range(2, self.frameOffset + 1):
                            if key in keys[i - m]:
                                #---去掉前m帧中出现的按键，认为隔了m帧出现的按键是新的按键
                                final_set = [x for x in final_set if x is not key]
                                break       
                    else:
                        for m in range(i):   #---对于视频的前m帧,只需要判断前面几帧是否有重复
                            if key in keys[m]:
                                final_set = [x for x in final_set if x is not key]
                                break

                final_set = sorted(final_set)
                for pressed_key in final_set:
                    count = 0
                    # end_frame = i+1   #--如果只在当前帧，应该是end_frame=i,就只有当前帧
                    end_frame=i 
                    #---看该按键持续了多少帧被按下
                    for j in range(i+1,len(times)):
                        if pressed_key in keys[j]:
                            count+=1
                            end_frame = j 
                        else:break 
                    # if mode=='white':
                    if count==0:  #--只在当前帧被按下
                        data = [times[i], pressed_key,times[end_frame]]
                    else :
                        #---记录了按键起始和结束时间,其实对于那种中间就断了一帧的情况还是连着按下的，可以把后面断的时间加进来0.0
                        data = [times[i], pressed_key,times[end_frame]]
                    pro_onset.append(data)

        if len(pro_onset) > 0:
            pro_onset = sorted(pro_onset, key=lambda x: (x[0], x[1], x[2]))
        return pro_onset
    

    def cal_F(self,recall,precise):
        if recall==0 or precise==0:
            return 0.0 
        F = 2.0*recall*precise/(recall+precise)
        return F

    def cuont_acu(self, pitch_onset, pro_onset):
        right_keys_precies = []
        right_keys_recall = []
        if len(pitch_onset) == 0 or len(pro_onset) == 0:
            return 0.0, 0.0, 0.0,0.0
        else:
            recall_index=[]
            pre_index=[]
            for i, w_key in enumerate(pitch_onset):
                cur_time = w_key[0]
                cur_key = w_key[1]
                cur_matchs = []

                for key_index,pro_key in enumerate(pro_onset):
                    if key_index in recall_index:continue
                    if (abs(cur_time - pro_key[0]) < self.tolerance) and (cur_key == pro_key[1]):
                        cur_matchs.append(pro_key)
                        recall_index.append(key_index)
                        break
                            
                if len(cur_matchs)>0:
                    right_keys_recall.append(cur_matchs)
                else:
                    count_frame = int(w_key[0]/self.pframe_time)
                    #print(count_frame,w_key[1])
            for i, w_key in enumerate(pro_onset):
                cur_time = w_key[0]
                cur_key = w_key[1]
                cur_matchs = []
                for key_index,pitch_key in enumerate(pitch_onset):
                    if key_index in pre_index:continue
                    if (abs(cur_time - pitch_key[0]) < self.tolerance) and (cur_key == pitch_key[1]):
                            cur_matchs.append(pitch_key)
                            pre_index.append(key_index)
                            break                            
                if len(cur_matchs)>0:
                    right_keys_precies.append(cur_matchs)
                else:
                    count_frame = int(w_key[0]/self.pframe_time)
                    #print(count_frame,w_key[1])
   
            #right_keys1 = sorted(right_keys1, key=lambda x: (x[0],x[1]))
            #right_keys1=list(set([tuple(t) for t in right_keys1]))
            #right_keys1 = sorted(right_keys1, key=lambda x: (x[0],x[1]))
            conf1 = len(right_keys_precies) / len(pro_onset)
            conf2 = len(right_keys_recall) / len(pitch_onset)
            return (conf1, conf2, len(right_keys_precies), len(right_keys_recall))


    def Total_precision(self):
        DetectNotes = self.processDetect1(self.detectPath)
        #---检测出来的按键数量
        Prokeys_nums = len(DetectNotes)
        #---midi中的按键数量
        Pitchkeys_num = len(self.pitch_onset)
        self.pro_onset = [po for po in DetectNotes if po[1]!=0]
        pitch_onte = [[line[0],line[1]] for line in self.pitch_onset]
        self.note_precies, self.note_recall, keys_pres,keys_recall = self.cuont_acu(pitch_onte, self.pro_onset)
        self.note_F = self.cal_F(self.note_recall,self.note_precies)
        data = 'note Total\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(self.note_precies,self.note_recall,self.note_F)
        self.evalout.write(data+'\n')
        print(data)        
    
    
    def evaluate_frame_precision(self):
        def parse_detect_file(detectPath):
            with open(detectPath, 'r') as f:
                lines = f.readlines()
            frames = []
            keys = []
            for i, line in enumerate(lines):
                line = line.strip().split()
                frame = int(os.path.basename(line[0]).split('.')[0])
                for j in range(2, len(line)):
                    if int(line[j])>0:
                        keys.append(int(line[j]))
                        frames.append(frame) 
            return frames, keys
            
        def cal_acc(det_frames,det_keys,midi_frames,midi_keys):
            recall_count = 0 
            precise_count = 0
            r_match_index=[]
            p_match_index=[]
            a=time.time()
            for idx,mframe in enumerate(midi_frames):
                match = False 
                for idy,dframe in enumerate(det_frames):
                    if idy in r_match_index:
                        continue
                    if abs(mframe-dframe)<=self.tol_frame and det_keys[idy]==midi_keys[idx]:  
                        match = True 
                        r_match_index.append(idy)
                        recall_count+=1
                        break
            # print('it cost {} minutes'.format((time.time()-a)/60))
            for idx,dframe in enumerate(det_frames):
                match = False 
                for idy,mframe in enumerate(midi_frames):
                    if idy in p_match_index:
                        continue
                    if abs(mframe-dframe)<=self.tol_frame and det_keys[idx]==midi_keys[idy]:
                        match = True 
                        p_match_index.append(idy)
                        precise_count+=1
                        break                

            precise,recall = 0.0,0.0
            if len(det_frames)>0:
                precise = precise_count/len(det_frames)
            if len(midi_frames)>0:
                recall = recall_count/len(midi_frames)
            # print('precise_count :\t {} \tdet_frames :\t {}'.format(precise_count,len(det_frames)))
            # print('recall_count  :\t {} \tmidi_frames :\t {}'.format(recall_count,len(midi_frames)))
            F = self.cal_F(recall,precise)
            return precise,recall,F 

        midi_frames,midi_keys = [],[]
        for pof in self.pitch_onset_offset:
            start_frame = round(pof[0]/self.pframe_time)
            end_frame = round(pof[1]/self.pframe_time)
            #---end_frame+1确保结束帧也加上去了
            for frame in range(start_frame, end_frame+1):
                midi_frames.append(frame)
                # midi_keys.append(self.white_num.index(pof[2])+1)
                midi_keys.append(pof[2])
        
        if self.detectPath is not None:
            det_frames, det_keys = parse_detect_file(self.detectPath)
            self.w_precies, self.w_recall, self.w_F = cal_acc(det_frames, det_keys, midi_frames, midi_keys)
            data = 'frame Total\tprecies:{:.2}\trecall:{:.2}\tFscore:{:.2}'.format(self.w_precies,self.w_recall,self.w_F)
            print(data) 
            self.evalout.write(data+'\n')            

      

if __name__ == '__main__':
    file_seqs = ['level_1_no_02', '25', '26', 'level_2_no_02']
    for file_seq in file_seqs:
        midiPath = cfg.EVALUATE_MAP[file_seq]['label_path']
        midiPath = midiPath.replace('_label', '_note')
        # midi_offset = cfg.EVALUATE_MAP[file_seq]['midi_offset']
        midi_offset = 0  #由于这里都是txt文件，相当于都已经是对应上了的
        detectPath = os.path.join('/home/lj/cy/data/piano/new/saved/YouTube/',
                                  file_seq + '/octave_pitch.txt')
        fps = float(cfg.EVALUATE_MAP[file_seq]['fps'])
        pframe_time = 1.0 / fps
        print(file_seq)
        evaluate = Accuracy(midiPath=midiPath,
                            detectPath=detectPath,
                            pframe_time=pframe_time,
                            start_frame = 0,
                            midi_offset = midi_offset,
                            frameOffset=2, 
                            tolerance=0.2                        
                            )
