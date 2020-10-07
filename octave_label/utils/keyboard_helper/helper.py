#-*- coding:utf-8 -*-
import cv2
import numpy as np 
import os 
import time 
from PIL import Image 
from config import cfg 
from IPython import embed 

def vis_bw_key(img,
        white_loc,
        black_boxes,
        white_top_boxes,
        white_bottom_boxes):
    img_copy = img.copy()
    height,width,_ = img.shape 
    for loc in white_loc:
        cv2.line(img_copy,(loc,9),(loc,height),(0,0,255),2)
    
    for box in black_boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,0),2)

    for box in white_top_boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(255,0,255),1)

    for box in white_bottom_boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(255,255,0),2)

    return img_copy

def vis_white_loc(img,white_loc):
    img_copy = img.copy()
    height,width,_ = img.shape 
    for loc in white_loc:
        cv2.line(img_copy,(loc,9),(loc,height),(0,0,255),2)
    return img_copy 

def vis_boxes(img,boxes,color): 
    img_copy = img.copy()
    for box in boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),color,2)
    return img_copy 

#---将白键和黑键的位置对应标出来，方便标注数据
def vis_white_black_loc(white_loc,boxes,base_info,save_path,img_lists):
    warp_M = base_info['warp_M']
    rote_M = base_info['rote_M']
    keyboard_rect = base_info['rect']

    black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                          31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                          60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
    white_num = [x for x in range(1, 89) if x not in black_num]
    txt_save_path=os.path.split(save_path)[0]
    frame_path=os.path.join(txt_save_path,os.path.basename(txt_save_path)+'_label.txt')
    fout=open(frame_path,'w')
    for img_file in img_lists:
        print(img_file)
        data='{}\n'.format(img_file)
        fout.write(data)
        img = Image.open(img_file)
        w,h = img.size 
        file_seq = os.path.basename(img_file).split('.')[0]
        opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        if rote_M is not None:
            opencv_img = cv2.warpAffine(opencv_img,rote_M,(w,h))
        if warp_M is not None:
            opencv_img = cv2.warpPerspective(opencv_img,warp_M,(w,h))
        
        img_copy=opencv_img[keyboard_rect[1]:keyboard_rect[3], keyboard_rect[0]:keyboard_rect[2]]
        height,width,_ = img_copy.shape 
        for i,loc in enumerate(white_loc):
            if i==len(white_loc)-1:break
            key_num=white_num[i]
            cv2.putText(img_copy,str(key_num),(loc+2,height-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

        for i,box in enumerate(boxes):
            x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
            key_num=black_num[i]
            cv2.putText(img_copy,str(key_num),(x1+3,y1+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        cv2.imwrite(os.path.join(save_path,os.path.basename(img_file)),img_copy)
    fout.close()


#---将白键和黑键的位置对应标出来，对应youtube上的数据
def vis_white_black_loc_youtube(white_loc,boxes,save_path,img_lists):


    black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                          31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                          60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
    white_num = [x for x in range(1, 89) if x not in black_num]
    txt_save_path=os.path.split(save_path)[0]
    for img_file in img_lists:
        print(img_file)

        img = Image.open(img_file)
        w,h = img.size 
        file_seq = os.path.basename(img_file).split('.')[0]
        opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        
        img_copy=opencv_img.copy()
        height,width,_ = img_copy.shape 
        for i,loc in enumerate(white_loc):
            if i==len(white_loc)-1:break
            key_num=white_num[i]
            cv2.putText(img_copy,str(key_num),(loc+2,height-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        for i,box in enumerate(boxes):
            x1,y1= box[0],box[1]
            key_num=black_num[i]
            cv2.putText(img_copy,str(key_num),(x1+3,y1+10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        cv2.imwrite(os.path.join(save_path,os.path.basename(img_file)),img_copy)
        break

def vis_white_loc_boxes(img,white_loc,boxes):
    img_copy = img.copy()
    height,width,_ = img.shape 
    for loc in white_loc:
        cv2.line(img_copy,(loc,9),(loc,height),(0,0,255),2)
    for box in boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,0),2)
    return img_copy 

def vis_detect_white_key(img,
                        hand_boxes,
                        key_indexs,
                        white_loc,
                        rect,
                        total_top,
                        total_bottom,
                        warp=False):
    img_copy = img.copy()
    height,width = img.shape[:2]
    ratio = width/1920 
    hratio = int(round(2.2*width/1920))
    for ind,loc in enumerate(white_loc):
        if ind ==len(white_loc)-1:
            break 
        if not warp:
            cv2.putText(img_copy, str(ind + 1), (int(white_loc[ind] + 2+rect[0]), int(0.9 *(rect[3]-rect[1]) +rect[1])), cv2.FONT_HERSHEY_PLAIN, ratio, (0, 0, 255), 1)
        else:
            height,width = img_copy.shape[:2]
            cv2.putText(img_copy, str(ind + 1), (int(white_loc[ind] + 2), int(0.9 *height)), cv2.FONT_HERSHEY_PLAIN, ratio, (0, 0, 255), 1)
    if cfg.VISION_DETECT:    
        for index in key_indexs:
            box1 = total_top[index-1]
            box2 = total_bottom[index-1]
            if not warp:
                x1_loc = int(box1[0]+rect[0])
                y1_loc = int(box1[1]+rect[1])
                x2_loc = int(box2[0]+rect[0])
                y2_loc = int(box2[1]+rect[1])
            else:
                x1_loc,y1_loc = int(box1[0]),int(box1[1])
                x2_loc,y2_loc = int(box2[0]),int(box2[1])
            cv2.rectangle(img_copy, (x1_loc, y1_loc), (x1_loc + int(box1[2]), y1_loc + int(box1[3])), (0, 0, 255), 1)
            cv2.rectangle(img_copy, (x2_loc, y2_loc), (x2_loc + int(box2[2]), y2_loc + int(box2[3])), (0, 0, 255), 1)
    if not warp:
        font = True if width<900 else False 
        for box in hand_boxes:
            cv2.rectangle(img_copy,(box[0][0]+rect[0],box[0][1]),(box[1][0]+rect[0],box[1][1]),(0,255,0),hratio)
    else:
        for box in hand_boxes:
            cv2.rectangle(img_copy,box[0],box[1],(0,255,0),hratio) 
    return img_copy 

def vis_detect_black_key(img,
                        hand_boxes,
                        key_indexs,
                        rect,
                        black_boxes,
                        warp=False):
    img_copy = img.copy()
    height,width = img.shape[:2]
    ratio = width/1920
    hratio = int(round(2.2*width/1920)) 
    for ind,box in enumerate(black_boxes):
        x1,y1,w,h = box
        if not warp:
            cv2.putText(img_copy, str(ind + 1), (int(x1 - 2+rect[0]), int(0.1 *(rect[3]-rect[1]) +rect[1])), cv2.FONT_HERSHEY_PLAIN, ratio, (0, 255, 0), 1)
        else: 
            height,width = img_copy.shape[:2]
            cv2.putText(img_copy, str(ind + 1), (int(x1 - 2), int(0.1 *height)), cv2.FONT_HERSHEY_PLAIN,ratio, (0, 255, 0), 1)
    if cfg.VISION_DETECT:
        for index in key_indexs:
            box = black_boxes[index-1]
            if not warp:
                x1_loc = int(box[0]+rect[0])
                y1_loc = int(box[1]+rect[1])
            else:
                x1_loc,y1_loc = int(box[0]),int(box[1])
            cv2.rectangle(img_copy, (x1_loc, y1_loc), (x1_loc + int(box[2]), y1_loc + int(box[3])), (0, 255, 255), 1)
    if not warp:
        for box in hand_boxes:
            cv2.rectangle(img_copy,(box[0][0]+rect[0],box[0][1]),(box[1][0]+rect[0],box[1][1]),(0,255,0),hratio)
    else:
        for box in hand_boxes:
            cv2.rectangle(img_copy,box[0],box[1],(0,255,0),hratio)
    return img_copy 


def vis_diff_img_key(img,
                    img_name,
                    hand_boxes,
                    white_loc,
                    black_boxes,
                    rect):
    img_copy = img.copy()
    height,width = img.shape[:2]
    ratio = width/1920 
    hratio = int(round(2.2*width/1920))
    for ind,loc in enumerate(white_loc):
        if ind ==len(white_loc)-1:
            break 
        cv2.putText(img_copy, str(ind + 1), (int(white_loc[ind] + 2), int(0.9*height)), cv2.FONT_HERSHEY_PLAIN, ratio, (0, 0, 255), 1)
    for ind,box in enumerate(black_boxes):
        x1,y1,w,h = box
        cv2.putText(img_copy, str(ind + 1), (int(x1 - 2), int(0.1*height)), cv2.FONT_HERSHEY_PLAIN, ratio, (0, 255, 0), 1)
    for box in hand_boxes:
        cv2.rectangle(img_copy,(box[0][0],box[0][1]-rect[1]),(box[1][0],box[1][1]-rect[1]),(0,255,0),hratio)
    cv2.putText(img_copy,img_name,(30,int(height//4)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
    return img_copy 


def vis_detect_total_key(img,
                        img_name,
                        hand_boxes,
                        white_key_indexs,
                        black_key_indexs,
                        white_loc,
                        rect,
                        total_top,
                        total_bottom,
                        black_boxes,
                        warp=False):
    img_copy = img.copy()
    img_copy = vis_detect_white_key(img_copy,
                                    hand_boxes,
                                    white_key_indexs,
                                    white_loc,
                                    rect,
                                    total_top,total_bottom,warp)
    img_copy = vis_detect_black_key(img_copy,hand_boxes,black_key_indexs,rect,black_boxes,warp)
    height,width,_ = img_copy.shape 
    ratio = 0.45 if warp else 0.8
    size = 1 if warp else 3 
    cv2.putText(img_copy,img_name,(int(width//2-5),int(height//4)),cv2.FONT_HERSHEY_SIMPLEX,ratio,(0,0,255),size)
    return img_copy 


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def new_save_dir(root,file_mark):
    base_img_dir = os.path.join(root,file_mark,'base_img')
    diff_img_dir = os.path.join(root,file_mark,'diff_img')
    detect_total_img_dir = os.path.join(root,file_mark,'detect_total_img')
    detect_white_img_dir = os.path.join(root,file_mark,'detect_white_img')
    detect_black_img_dir = os.path.join(root,file_mark,'detect_black_img')
    press_white_img_dir = os.path.join(root,file_mark,'press_white_img')
    press_black_img_dir = os.path.join(root,file_mark,'press_black_img')

    ensure_dir(base_img_dir)
    ensure_dir(diff_img_dir)
    ensure_dir(detect_total_img_dir)
    ensure_dir(detect_white_img_dir)
    ensure_dir(detect_black_img_dir)

    ensure_dir(press_white_img_dir)
    ensure_dir(press_black_img_dir)

    detect_img_dir = (detect_total_img_dir,detect_white_img_dir,detect_black_img_dir)
    press_img_dir = (press_white_img_dir,press_black_img_dir)
    return base_img_dir,diff_img_dir,detect_img_dir,press_img_dir 


def near_white(white_loc,boxes):
    #---[((15, 234), (67, 303))]

    index_list = []
    if len(boxes)==0:
        return index_list 
    for box in boxes:
        min_loc = box[0][0]
        max_loc = box[1][0]
        diffs1,diffs2 = [],[]
        for w_loc in white_loc:
            diff1 = abs(min_loc-w_loc)
            diff2 = abs(max_loc-w_loc)
            diffs1.append(diff1)
            diffs2.append(diff2)
        '''
        left_index,right_index = 0,0
        for i,w_loc in enumerate(white_loc):
            if i == len(white_loc)-1:
                break 
            if white_loc[i]<=min_loc and white_loc[i+1]>=min_loc:
                left_index = i 
            if white_loc[i]<=max_loc and white_loc[i+1]>=max_loc:
                right_index = i 
        '''
        left_index = diffs1.index(min(diffs1))-1 
        right_index = diffs2.index(min(diffs2))+1
        index_list.append((left_index,right_index))
    return index_list 

def near_black(black_boxes,boxes):
    index_list = []
    if len(boxes)==0:
        return index_list 
    for box in boxes:
        minx = box[0][0]
        maxx = box[1][0]
        diffs1,diffs2 = [],[]
        for index,blbox in enumerate(black_boxes):
            x1,y1,w,h = blbox
            x2,y2 = x1+w,y1+h 
            diff1 = abs(x1-minx)
            diff2 = abs(x2-maxx)
            diffs1.append(diff1)
            diffs2.append(diff2)
        left_index = max(0,diffs1.index(min(diffs1))-1)
        right_index = min(len(black_boxes)-1,diffs2.index(min(diffs2))+1)
        index_list.append((left_index,right_index))
    return index_list 

class timer(object):
    def __init__(self):
        self.count = 0 
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0 
        self.avg_time = 0 
    
    def reset():
        self.start_time = 0
        self.end_time = 0 
        self.total_time = 0
        self.avg_time = 0 
        self.count = 0

    def tic(self):
        self.start_time = time.time()
    
    def toc(self):
        self.end_time = time.time()
        self.total_time+=(self.end_time-self.start_time)
        self.count+=1 
        return self.end_time-self.start_time 

    def elapsed(self,avg=True):
        if avg:
            return self.total_time/self.count 
        return self.total_time 

def save_detect_result(keys_list,img_path,fout,fps):
    img_path = os.path.realpath(img_path)
    count_frame = int(os.path.splitext(os.path.basename(img_path))[0])
    per_frame_time = 1.0/fps 
    cur_time = per_frame_time*(count_frame)
    if len(keys_list)==0:
        data = '{} {:.4} {}\n'.format(img_path,cur_time,0)
        fout.write(data)
        fout.flush()
        return 
    fout.write('{} {:.4} '.format(img_path,cur_time))
    for key in keys_list:
        fout.write('{} '.format(key))
    fout.write('\n')
    fout.flush()


def img2video(img_dir,video_file):
    imgs = [os.path.join(img_dir,x) for x in os.listdir(img_dir) if x.endswith('jpg')]
    assert len(imgs)>0,print('imgs error')
    img = cv2.imread(imgs[0])
    h,w,_ = img.shape 
    size = (w,h)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter(video_file,fourcc=fourcc, fps=25.0, frameSize=size)
    imgs.sort()
    for img_file in imgs:
        img = cv2.imread(img_file)
        fimg = Image.fromarray(img)  
        fimg = fimg.resize(list(size),resample=Image.NONE)
        img = np.array(fimg)
        videoWriter.write(img)
    videoWriter.release()


def black_white_index_dict():
    index_dict = {}
    index_dict['1'] = [1]
    index_dict['2'] = [1]
    ib,iw = 3,2
    for i in range(7):
        index_dict[str(ib)] = [iw]
        ib+=1
        index_dict[str(ib)] = [iw]
        iw+=1
        index_dict[str(ib)].append(iw)
        ib+=1
        index_dict[str(ib)] = [iw]
        ib+=1
        iw+=1
        index_dict[str(ib)] = [iw]
        ib+=1
        index_dict[str(ib)] = [iw]
        iw+=1
        index_dict[str(ib)].append(iw)
        ib+=1
        index_dict[str(ib)] = [iw]
        iw+=1
        index_dict[str(ib)].append(iw)
        ib+=1
        index_dict[str(ib)] = [iw]
        iw+=1
        ib+=1
    index_dict[str(ib)] = [iw-1]
    return index_dict

def paper_black_white_index_dict():
    index_dict = {}
    ib,iw = 1,1
    for i in range(5):
        index_dict[str(ib)] = [iw]
        ib+=1 
        index_dict[str(ib)] = [iw]
        iw+=1 
        index_dict[str(ib)].append(iw)
        ib+=1 
        index_dict[str(ib)] = [iw]
        ib+=1
        iw+=1 
        index_dict[str(ib)] = [iw]
        ib+=1  
        index_dict[str(ib)] = [iw]
        iw+=1 
        index_dict[str(ib)].append(iw)
        ib+=1 
        index_dict[str(ib)] = [iw]
        iw+=1 
        index_dict[str(ib)].append(iw)
        ib+=1 
        index_dict[str(ib)] = [iw]
        iw+=1
        ib+=1 
    index_dict[str(ib)] = [iw-1]
    return index_dict 


def vertify_press_white(file_seq,key_index,
                        mask,
                        index_dict,
                        black_boxes,
                        hand_boxes,
                        white_loc,
                        prob):

    #---index_dict->键盘中白键对应的黑键
    bindex = index_dict[str(key_index)]
    if len(bindex)==2:
        box1 = black_boxes[bindex[0]-1]
        box2 = black_boxes[bindex[1]-1]
        if box1[0]<box2[0]:
            b1,b2 = box1,box2
        else:
            b1,b2 = box2,box1
        cx1,cx2= b1[0]+b1[2],b2[0]
    else:
        box = black_boxes[bindex[0]-1]
        x1,x2 = box[0],box[0]+box[2]
        w1 = white_loc[key_index-1]
        w2 = white_loc[key_index]

        if abs(x1-w1)+abs(x2-w1)<abs(x1-w2)+abs(x2-w2):
            if abs(w2-x1)<abs(w2-x2):
                cx1 = min(w2,x1)
                cx2 = max(w2,x1)
            else:
                cx1 = min(w2,x2)
                cx2 = max(w2,x2)
        else:
            if abs(w1-x1)<abs(w1-x2):
                cx1 = min(w1,x1)
                cx2 = max(w1,x1)
            else:
                cx1 = min(w1,x2)
                cx2 = max(w1,x2)

    #---相当于这里也是判断就是说白键中线位置手的像素有多少
    #--如果是相邻黑键夹着白键->则是两个黑键的边界的中间位置
    #--对于其他白键->则是白键区域中去掉黑键在白键内的部分像素再取中间位置

    #---这个就是说判断白键区域内(去掉黑键部分)有没有手的Mask像素，有才认为是被按下的0.0
    box = black_boxes[bindex[0]-1]
    bend = int(box[1]+box[3])
    wc = int((cx1+cx2)/2)
    length = len(np.where(mask[0:bend,wc]!=0)[0])
    '''
    if length<50:
        for index in bindex:
            box = black_boxes[index-1]
            xcenter = int(box[0]+box[2]/2.0)
            ystart = int(box[1])
            yend = int(box[1]+box[3])
            count = 0
            for i in range(ystart,yend):
                if mask[i,xcenter]!=0:
                    count+=1 
            if count>2 and prob<0.85:
                return False,0 
            #elif count>2:
            #    return False,1 
    '''

    #---mask是键盘区域内手的分割mask
    #---这个是排除由黑键按下导致相邻的白键被误检的情况
    
    # if int(file_seq)==148:
    #     embed()

    for index in bindex:
        box = black_boxes[index-1]
        xcenter = int(box[0]+box[2]/2.0)
        ystart = int(box[1])
        yend = int(box[1]+box[3])
        count = 0
        #--该位置有像素说明可能是按下了黑键然后导致隔壁白键被误检，因为按下白键一般手在下面
        #---但是还有一种情况就是说本身白键被按下，白键位置也是会有手的像素点，同时隔壁黑键也会有手的像素，但白键确实是被按下的
        for i in range(ystart,yend):
            if mask[i,xcenter]!=0:
                count+=1 
        if count>2 and prob<0.8: #--同时该白键检测得到的prob<0.8
            return False,0


    white1,white2 = white_loc[key_index-1],white_loc[key_index]
    for box in hand_boxes:
        x1,x2 = box[0][0]-10,box[1][0]+10
        #---就是说白键是在手的检测boxes区域内才算正确/或者刚看垮在boxes中间
        if (white1<=x1 and white2>=x1) or (white1>=x1 and white2<=x2) or (white1<=x2 and white2>=x2):
            return True,0
    return False,0


def vertify_press_black(key_index,mask,black_boxes):
    #---相当于就是判断该黑键被按下时该区域中有没有手的mask部分像素，被按下肯定是有手在上面的

    box = black_boxes[key_index]
    xcenter = int(box[0]+box[2]/2.0)
    ystart,yend = int(box[1]),int(box[1]+box[3])
    count = 0
    for i in range(ystart,yend):
        if mask[i,xcenter]!=0:
            count+=1 
    #--应该再加一个阈值的判断和Mask一起决定是否返回？？
    if count>2:return True 

    return False 


def update_base_img(file_seq,base_img,cur_img,white_loc,hand_boxes):
    h,w = base_img.shape[:2]
    cur_img_ = cur_img.copy()
    if len(hand_boxes)>0:
        #---出现这种情况就是说查分图没检测到手然后分割检测到手了，改一下就行
        # embed()
        index_list = near_white(white_loc,hand_boxes)
        for index_pair in index_list:
            index1 = max(index_pair[0],0)
            index2 = min(len(white_loc)-1,index_pair[1])
            x1,x2 = white_loc[index1],white_loc[index2]
            cur_img_[:,x1:x2] = base_img[:,x1:x2].copy()
    base_img = cur_img_
    return base_img 

#--返回视频帧的平均亮度
def get_img_light(img_list):
    mean_lights=[]
    for img_path in img_list:
        mean_lights.append(np.mean(cv2.imread(img_path)))
    
    return np.mean(mean_lights)


def find_begin_frame(keyboard_model,img_list):
    base_img,rect = None,None 
    count_frame = 0
    frame_num=1
    frameTostop=20
    keyboard_frame=0

    avg_light = get_img_light(img_list)
    light_thresh = 25
    for i,img_path in enumerate(img_list):
        # print(img_path)
        img = Image.open(img_path)
        w,h = img.size 
        count_frame+=1 
        if abs(np.mean(np.array(img))-avg_light)>light_thresh:continue
        keyboard_info = keyboard_model.detect_keyboard(img)
        if not keyboard_info['flag']:
            continue
        begin_frame = count_frame - 1
        break
    for img_path in img_list[::-1]:
        # print(img_path)
        img = Image.open(img_path)
        w,h = img.size 
        count_frame+=1 
        if abs(np.mean(np.array(img))-avg_light)>light_thresh:continue
        keyboard_info = keyboard_model.detect_keyboard(img)
        if not keyboard_info['flag']:
            continue
        end_frame=int(os.path.basename(img_path).split('.')[0])
        break
    return begin_frame, end_frame


def find_base_img(keyboard_model,
                  hane_seg_model,
                  img_list):
    base_img,rect = None,None 
    count_frame = 0
    frame_num=1
    frameTostop = 20    #---正序查找带键盘图像20帧变为倒序
    ReverseTostop = 0   #---倒序查找带键盘图像最多20帧
    keyboard_frame=0
    result = {'base_img':base_img,'img':None,'count_frame':count_frame,'rect':rect,'rote_M':None,'warp_M':None}
    avg_light=get_img_light(img_list)
    light_thresh=25
    begin_flag = True
    for i,img_path in enumerate(img_list):
        #---倒序查找
        if keyboard_frame>frameTostop:
            img_path=img_list[-frame_num]
            frame_num += 1
            #---如果倒序查找超过20帧还没有找到键盘图像，则认为视频中没有不包含手的视频图像,通过其他方法来解决
            if ReverseTostop > 20:
                return None
        
        #---标记好了背景帧，测试为了方便
        file_mark = os.path.basename(os.path.split(img_path)[0])
        if file_mark in cfg.BASE_FRAME:
            img_path = img_list[int(cfg.BASE_FRAME[file_mark]['base_frame'])]

        #---模型对于那篇sighttosound的论文的图像表现不行，键盘检测不出来
        # numpy_img=cv2.imread(img_path)
        # cur_h,cur_w,_=numpy_img.shape
        # new_img=np.random.randint(128,129,(int(cur_w/1.7),cur_w,3)).astype(np.uint8)
        # new_h,new_w,_=new_img.shape
        # new_img[(new_h-cur_h):new_h,:,:]=numpy_img
        # cv2.imwrite('./test1.jpg',new_img)
        # img=Image.fromarray(new_img).convert('RGB')

        img = Image.open(img_path)
        w,h = img.size 
        #--如果是正序的话count_frame就表示第一次检测到键盘的图，后续检测按键可以从这里开始
        #---倒序的话count_frame就不是表示第一次检测到键盘的图了，直接从头(第一张图像)开始检测也行
        # count_frame+=1 
        
        #---就是说如果当前帧和图像帧整体亮度相差大于light_thresh，说明不是钢琴图片，直接跳过
        #---视频中大多数是钢琴图像，因此背景帧图像亮度应该与平均亮度相差较小
        # print("the avg_light is {}".format(avg_light))
        # print("the diff is {}".format(abs(np.mean(np.array(img))-avg_light)))
        if abs(np.mean(np.array(img)) - avg_light) > light_thresh: continue
        if keyboard_frame > frameTostop: ReverseTostop += 1
        
        print('the img with keyboard is {}'.format(img_path))
        keyboard_info = keyboard_model.detect_keyboard(img)
        if not keyboard_info['flag']:
            continue

        #---从检测到键盘开始如果连续20帧都没找到背景，则倒序查找
        keyboard_frame += 1
        rect = keyboard_info['keyboard_rect']
        opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        count_frame = int(os.path.basename(img_path).split('.')[0])
        while begin_flag:
            first_frame = count_frame
            begin_flag = False
            break
        if keyboard_info['rote_M'] is None:
            base_img = opencv_img[rect[1]:rect[3],rect[0]:rect[2]]
            hand_boxes,mask = hane_seg_model.segment_detect_hand(img,rect)
            result = {'base_img':base_img,'img':opencv_img,
                    'count_frame':first_frame,'end_frame':None,
                    'rect':rect,'rote_M':None,'warp_M':None} 
        else:
            rote_M = keyboard_info['rote_M'] 
            if keyboard_info['warp_M'] is None:
                rotated_img=keyboard_info['rotated_img']
                base_img = rotated_img[rect[1]:rect[3],rect[0]:rect[2]]
                img = Image.fromarray(cv2.cvtColor(rotated_img,cv2.COLOR_BGR2RGB))
                # hand_boxes,mask = hane_seg_model.segment_detect_hand(img,rect)
                result = {'base_img':base_img,'img':rotated_img,
                    'count_frame':first_frame,'end_frame':None,
                    'rect':rect,'rote_M':rote_M,'warp_M':None}
            else:
                warp_M = keyboard_info['warp_M']
                warp_img = keyboard_info['warp_img']
                base_img = warp_img[rect[1]:rect[3],rect[0]:rect[2]]
                img = Image.fromarray(cv2.cvtColor(warp_img,cv2.COLOR_BGR2RGB))
                
                result = {'base_img':base_img,'img':warp_img,
                    'count_frame':first_frame,'end_frame':None,
                    'rect':rect,'rote_M':rote_M,'warp_M':warp_M}
            hand_boxes,mask = hane_seg_model.segment_detect_hand(img,rect)
        numpy_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if len(hand_boxes)==0:return result  
        if len(hand_boxes)==1:
            hand_xmin,hand_ymin = hand_boxes[0][0][0],hand_boxes[0][0][1]
            if hand_ymin>rect[3]+5:return result 
        if len(hand_boxes)>1:
            hand_xmin1,hand_ymin1 = hand_boxes[0][0][0],hand_boxes[0][0][1]
            hand_xmin2,hand_ymin2 = hand_boxes[1][0][0],hand_boxes[1][0][1]
            if hand_ymin1>rect[3]+5 and hand_ymin2>rect[3]+5:return result 

    return result 
    
def find_key_loc(bwlabel_model,
                base_img):
    img = base_img.copy()
    white_loc,black_boxes,total_top,total_bottom = bwlabel_model.key_loc(img)
    return white_loc,black_boxes,total_top,total_bottom 

def find_video_base_img(keyboard_model,
                        modelproduct,
                        video_file):
    base_img,rect = None,None 
    count_frame = 0
    capture = cv2.VideoCapture(video_file)
    result = {'base_img':base_img,'img':None,'count_frame':count_frame,'rect':rect,'rote_M':None,'warp_M':None}
    if not capture.isOpened():
        return result 
    while True:
        ret,frame = capture.read()
        h,w,_ = frame.shape
        count_frame+=1 
        if not ret:
            break 
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        keyboard_info = keyboard_model.detect_keyboard(img)
        if not keyboard_info['flag']:
            continue  
        if keyboard_info['rote_M'] is None:
            opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            rect = keyboard_info['keyboard_rect']
            base_img = opencv_img[rect[1]:rect[3],rect[0]:rect[2]]
            hand_boxes = modelproduct.detect_hand(img,rect)
            result = {'base_img':base_img,'img':opencv_img,
                    'count_frame':count_frame,
                    'rect':rect,'rote_M':None,'warp_M':None} 
        else:
            opencv_img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
            if keyboard_info['warp_M'] is None:
                rote_M = keyboard_info['rote_M']    
                rotated_img = cv2.warpAffine(opencv_img,rote_M,(w,h))
                rect = keyboard_info['keyboard_rect']
                base_img = rotated_img[rect[1]:rect[3],rect[0]:rect[2]]
                img = Image.fromarray(cv2.cvtColor(rotated_img,cv2.COLOR_BGR2RGB))
                hand_boxes = modelproduct.detect_hand(img,rect)
                result = {'base_img':base_img,'img':rotated_img,
                    'count_frame':count_frame,
                    'rect':rect,'rote_M':rote_M,'warp_M':None}
            else:
                rote_M = keyboard_info['rote_M']
                warp_M = keyboard_info['warp_M']
                rotated_img = cv2.warpAffine(opencv_img,rote_M,(w,h))
                warp_img = cv2.warpPerspective(rotated_img,warp_M,(w,h))
                rect = keyboard_info['keyboard_rect']
                base_img = warp_img[rect[1]:rect[3],rect[0]:rect[2]]
                img = Image.fromarray(cv2.cvtColor(warp_img,cv2.COLOR_BGR2RGB))
                hand_boxes = modelproduct.detect_hand(img,rect)
                result = {'base_img':base_img,'img':warp_img,
                    'count_frame':count_frame,
                    'rect':rect,'rote_M':rote_M,'warp_M':warp_M}
        if len(hand_boxes)==0:
            capture.release()
            return result  
        if len(hand_boxes)==1:
            hand_xmin,hand_ymin = hand_boxes[0][0][0],hand_boxes[0][0][1]
            if hand_ymin>rect[3]+5:
                capture.release()
                return result 
        if len(hand_boxes)>1:
            hand_xmin1,hand_ymin1 = hand_boxes[0][0][0],hand_boxes[0][0][1]
            hand_xmin2,hand_ymin2 = hand_boxes[1][0][0],hand_boxes[1][0][1]
            if hand_ymin1>rect[3]+5 and hand_ymin2>rect[3]+5:
                capture.release()
                return result 
    capture.release()
    return result 


def save_prob_file(wpath,bpath,wprobs,bprobs):
    def save(path,probs):
        fout = open(path,'w')
        for items in probs:
            for item in items:
                data = '{:.3}\t'.format(float(item))
                fout.write(data)
            fout.write('\n')
        fout.close()
    save(wpath,wprobs)
    save(bpath,bprobs)

if __name__=='__main__':
    index_dict = black_white_index_dict()
    print(index_dict)

