# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np 
from IPython import embed 

def remove_region(img):
    if len(img.shape) == 3:
        print("please input a gray image")
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            # if (i < 30 or i > (2.0/3) * h):
            if (i < 0.1 * h or i > (2.0/3) * h):
                img[i, j] = 255
    for i in range(h):
        for j in range(w):
            if (j < 0.01 * w or j > 0.994 * w):
                img[i, j] = 255
    return img

def near_white(white_loc,black_boxes):
    diffs = []
    for i in range(len(black_boxes)):
        diff = abs(black_boxes[i][0] - white_loc)
        diffs.append(diff)
    index = diffs.index(min(diffs))
    return index

def contrast_img(img, c, b):
    rows, cols, channels = img.shape 
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return  dst 

class BwLabel(object):
    def __init__(self):
        super(BwLabel, self).__init__()

    def white_black_dict(self):
        wh_dict=[0,0,0]
        for i in range(3,53):
            div=int(i/7)
            if i%7==3 or i%7==4 :
                wh_dict.append(div*5+1)
            elif i%7==5:
                wh_dict.append(div*5+2)
            elif i%7==6:
                wh_dict.append(div*5+3)
            elif i%7==0:
                wh_dict.append((div-1)*5+3)
            elif i%7==1:
                wh_dict.append((div-1)*5+4)
            else:
                wh_dict.append((div-1)*5+5)
        return wh_dict


    def key_loc(self, base_img,phase='test'):
        white_loc = []
        black_boxes = []
        total_top = []
        total_bottom = [] 
        black_loc = []
        ''' 
        ori_img = base_img.copy()
        height,width,_ = base_img.shape 
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        base_img = remove_region(base_img)
        _, base_img = cv2.threshold(base_img, 150, 255, cv2.THRESH_BINARY) 
        base_img = cv2.GaussianBlur(base_img, (5, 5), 0)
        black_boxes = find_connect_domain(base_img)
        black_loc = [box[0] for box in black_boxes]
        '''
        ori_img = base_img.copy()
        draw_img = base_img.copy()
        height,width,_ = ori_img.shape 
        # embed()
        black_boxes,black_loc = self.find_black_boxes(ori_img)

        if len(black_boxes)!=36:
            ori_img = contrast_img(ori_img,1.3,3)
            black_boxes,black_loc = self.find_black_boxes(ori_img)

        #---在这里如果black_boxes对应的删除了元素的话，black_loc也要删除，因为后面要用到，是对应的
        if len(black_boxes)==37:
            area1 = black_boxes[0][2]*black_boxes[0][3]
            area2 = black_boxes[-1][2]*black_boxes[-1][3]
            if area1>area2:
                del black_boxes[-1]
                del black_loc[-1]
            else:
                del black_boxes[0]
                del black_loc[0]

        # for boxes in black_boxes:
        #     x1,y1=boxes[0],boxes[1]
        #     x2,y2=boxes[0]+boxes[2],boxes[1]+boxes[3]
        #     cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,0,255),2)
        # cv2.imwrite('./1.jpg', draw_img)
        # embed()

        if not phase=='test':
            #---对于包含手的背景图，返回第3个黑键的高度
            return black_boxes[2][3]

        assert len(black_boxes)==36,'black number is wrong'
    
        #---有时候对于一些钢琴键盘，两边的键盘亮度不一致，导致检测出来的黑键的位置宽度较小
        flag='normal'
        black_width=[x[2] for x in black_boxes]
        new_black_boxes=list()
        for i,b_width in enumerate(black_width):
            thresh=4
            if i==len(black_width)-1:break
            #---前后的框比后面的大，说明后面的框像素较少，需要调整
            if b_width-black_width[i+1]>thresh:
                change_info={i:b_width-black_width[i+1]}
                flag='right'
                for key,value in change_info.items():
                    change_idx,change_value=key,value
                new_black_boxes=black_boxes[:change_idx+1]
                break
            #---后面的框比前面的大，说明前面的框像素较少，需要调整
            elif black_width[i+1]-b_width>thresh:
                change_info=i
                flag='left'
                break

        if flag=='right':
            for i in range(change_idx+1,len(black_boxes)):
                box=black_boxes[i]
                new_black_boxes.append((box[0],box[1],box[2]+change_value,box[3]))
        
        if len(new_black_boxes)>0:
            black_boxes=new_black_boxes.copy()
            black_loc = [box[0] for box in black_boxes]

        for boxes in black_boxes:
            x1,y1=boxes[0],boxes[1]
            x2,y2=boxes[0]+boxes[2],boxes[1]+boxes[3]
            cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,0,255),2)
        # cv2.imwrite('./black_boxes_img.jpg',draw_img)

        # #----得到白键的区域
        white_loc = self.find_white_loc_old(black_loc,black_boxes,width)
        #print("the number of whitekey_num is {}".format(len(white_loc)))
        #--------找到白键所在的box---
        wh_dict=self.white_black_dict()
        for i in range(1, len(white_loc)):
            white_x = white_loc[i - 1]
            white_width = white_loc[i] - white_x
            index=wh_dict[i]
            if ((((i%7== 3) or (i%7==6)) and i < 52) or i==1):
                top_box=(white_x, 0, max(int(black_boxes[index][0] - white_x),int(1)), 1.1 * black_boxes[index][3]) #---(x,y,w,h)
                bottom_box=(white_x,1.1*black_boxes[index][3],white_width,height-1.1*black_boxes[index][3])
            
            elif (i%7==4 or i%7==0 or i%7==1):
                top_box=(black_boxes[index][0]+black_boxes[index][2], 0, max(int(black_boxes[index+1][0] - (black_boxes[index][0]+black_boxes[index][2])),1), 1.1 * black_boxes[index][3])
                bottom_box=(white_x,1.1*black_boxes[index][3],white_width+2,height-1.1*black_boxes[index][3])
            
            elif (i%7==5 or i%7==2 or i==2):
                top_box=(black_boxes[index][0]+black_boxes[index][2], 0, white_loc[i] - max(int(black_boxes[index][0]+black_boxes[index][2]),1), 1.1 * black_boxes[index][3])
                bottom_box=(white_x,1.1*black_boxes[index][3],white_width+2,height-1.1*black_boxes[index][3])
            
            else:
                top_box=(white_x + 1, 0, max(int(white_loc[i] - white_x - 1),1), 1.1 * black_boxes[35][3])
                bottom_box=(white_x + 1, 1.1 * black_boxes[35][3], white_loc[i] - white_x - 1, height - 1.1 * black_boxes[35][3])
            total_top.append(top_box)
            total_bottom.append(bottom_box)            

        for i,top_box in enumerate(total_top):
            x1,y1,x2,y2=int(top_box[0]),int(top_box[1]),int(top_box[0]+top_box[2]+2),int(top_box[1]+top_box[3])
            x3,y3,x4,y4=int(total_bottom[i][0]),int(total_bottom[i][1]),int(total_bottom[i][0]+total_bottom[i][2]),int(total_bottom[i][1]+total_bottom[i][3])
            cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.rectangle(draw_img,(x3,y3),(x4,y4),(0,0,255),1)
            cv2.putText(draw_img,str(i+1),(x3+5,15),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            cv2.putText(draw_img,str(i+1),(x3+5,height-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        # cv2.imwrite('./draw_img.jpg',draw_img)

        white_loc = np.array(white_loc,dtype=np.int32)
        black_boxes = np.array(black_boxes,dtype=np.int32)
        total_top = np.array(total_top,dtype=np.int32)
        total_bottom = np.array(total_bottom,dtype=np.int32)
        return  white_loc,black_boxes,total_top,total_bottom
    
    def find_black_boxes(self,ori_img):
        #---数据集的那些图像偏暗
        # thresh = 100
        thresh = 120
        while True:
            base_img = ori_img.copy()
            base_img = cv2.cvtColor(base_img,cv2.COLOR_BGR2GRAY)
            base_img = remove_region(base_img)
            test_img=base_img.copy()
            _,base_img = cv2.threshold(base_img,thresh,255,cv2.THRESH_BINARY_INV)
            cv2.imwrite('./base_img.jpg', base_img)
            # kernel = np.ones((10,10), dtype=np.uint8)
            # cv2.imwrite('./base_img.jpg',base_img)
            # embed()
            # base_img = cv2.erode(base_img, kernel=kernel, iterations=1)
            # base_img = cv2.dilate(base_img, kernel=kernel, iterations=1) 
            # cv2.imwrite('./base_img1.jpg',base_img)

            black_boxes = self.find_black_keys(base_img)
            black_boxes = sorted(black_boxes,key = lambda x:x[0])
            black_loc = [box[0] for box in black_boxes]
            if len(black_loc)>36:
                thresh-=1
            elif len(black_loc)<36:
                thresh+=1
            else:
                break
            if thresh<90 or thresh>150:
                break
            # break
        return black_boxes,black_loc 

    def find_white_loc_old(self,black_loc,black_boxes,width):
        white_loc = []
        black_gap1 = black_loc[3] - black_loc[2]  #--第一个周期区域内的黑键间隔
        ratio = 23.0 / 41
        # ratio = 23.0 / 40
        whitekey_width1 = ratio * black_gap1  
        half_width1 = black_boxes[4][2]    #T1中第四个黑键被均分,从该位置开始算区域起始位置
        keybegin = black_loc[4] + half_width1 / 2.0-7.0 * whitekey_width1
        for i in range(10):
            if int(keybegin + i * whitekey_width1) < 0:
                white_loc.append(1)
            else:
                white_loc.append(keybegin + i * whitekey_width1)
            
        for i in range(6):  #----剩下的6个循环区域
            axis = 8 + i * 5
            black_gap2 = black_loc[axis] - black_loc[axis - 1]
            whitekey_width2 = ratio * black_gap2 
            half_width2 = black_boxes[axis + 1][2] 
            keybegin1 = black_loc[axis + 1] + float(half_width2 / 2.0) - 5.0 * whitekey_width2
            for j in range(1,8):
                white_loc.append(keybegin1 + j * whitekey_width2)
            if i == 5:  #----最后一次循环将钢琴最后一个白键加上
                white_loc.append(min(width - 1,keybegin1 + 8 * whitekey_width2))
            
        return white_loc 

    def find_white_loc(self,black_loc):
        white_loc = []
        black_gap1 = black_loc[2] - black_loc[1]
        # w_gap1 = 63.0 / 28 * black_gap1 / 3  #---T1前三个白键的间隔
        w_gap1 = 70.0 / 28 * black_gap1 / 3  #---T1前三个白键的间隔
        w_begin1 = black_loc[1] - 2.0 / 3 * black_gap1
        # w_begin1 = black_loc[1] - 1.0 / 2 * black_gap1
        for i in range(3):
            white_loc.append(w_begin1 + i * w_gap1)

        #----最开始的那两个白键
        for i in range(1,3):
            if int(w_begin1 - i * w_gap1) < 0:
                white_loc.append(1)
            else:
                white_loc.append(w_begin1 - i * w_gap1)

        #----周期内后4个白键
        black_gap2 = black_loc[5] - black_loc[4]
        w_gap2 = 94.0 / 27 * black_gap2 / 4  #---T1前三个白键的间隔
        w_begin2 = black_loc[3] - 16.0 / 27 * black_gap2
        # w_begin2 = black_loc[3] - 13.0 / 27 * black_gap2
        for i in range(4):
            white_loc.append(w_begin2 + i * w_gap2)

        #----后面的那几个周期
        for i in range(6):
            axis1 = 7 + i * 5
            black_gap3 = black_loc[axis1] - black_loc[axis1 - 1]
            w_gap3 = 70.0 / 28 * black_gap3 / 3  #---T1前三个白键的间隔
            w_begin3 = black_loc[axis1 - 1] - 2.0 / 3 * black_gap3
            #----前3个白键
            for j in range(3):
                white_loc.append(w_begin3 + j * w_gap3)

            axis2 = 10 + i * 5
            black_gap4 = black_loc[axis2] - black_loc[axis2 - 1]
            w_gap4 = 94.0 / 27 * black_gap4 / 4  #---T1前三个白键的间隔
            w_begin4 = black_loc[axis2 - 2] - 16.0 / 27 * black_gap4
            #---后4个白键
            if i == 5:   #---最后一个周期把最后两个键也加上
                for j in range(6):
                    white_loc.append(w_begin4 + j * w_gap4)
            else:
                for j in range(4):
                    white_loc.append(w_begin4 + j * w_gap4)
        white_loc.sort()
        return white_loc 

    def find_black_keys(self, base_img):
        contours,_ = cv2.findContours(base_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        black_boxes = []
        height, width = base_img.shape[:2]

        for idx,cnt in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(cnt)
            if h>height*0.3 and w>9:   #----键盘的长大于0.3*h,宽大于9
                x1,y1,x2,y2 = x,y,x+w,y+h 
                for i in range(y2,y1,-1):
                    count = 0 
                    for j in range(x1,x2):
                        if base_img[i,j]!= 0:
                            count+=1 
                    if count > (x2-x1)*0.5:
                        black_boxes.append((x1,y1,w,i-y1))
                        break
        
        # print(len(black_boxes))
        # for boxes in black_boxes:
        #     x1,y1=boxes[0],boxes[1]
        #     x2,y2=boxes[0]+boxes[2],boxes[1]+boxes[3]
        #     cv2.rectangle(ori_img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imwrite('./ori_img1.jpg', ori_img1)
            

        if len(black_boxes)!=36:
            ws = [box[2] for box in black_boxes]
            ws = np.array(ws)
            me = np.median(ws)
            for i,wd in enumerate(ws):
                if wd<me*0.5:del black_boxes[i]
        return black_boxes