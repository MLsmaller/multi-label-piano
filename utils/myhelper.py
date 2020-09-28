import os 
import torch
import numpy as np
import cv2
from PIL import Image

from config import cfg
from IPython import embed

def near_white(white_loc,boxes):
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

def find_img_key_loc(bwlabel_model,
                base_img,phase='test'):
    img = base_img.copy()
    if not phase=='test':
        boxes_height = bwlabel_model.key_loc(img,phase)
        return boxes_height
    else:
        white_loc,black_boxes,total_top,total_bottom = bwlabel_model.key_loc(img,phase)
        return white_loc,black_boxes,total_top,total_bottom 
    

def vis_white_loc_boxes(img,white_loc,boxes):
    img_copy = img.copy()
    height,width,_ = img.shape 
    for loc in white_loc:
        cv2.line(img_copy,(loc,9),(loc,height),(0,0,255),2)
    for box in boxes:
        x1,y1,x2,y2 = box[0],box[1],box[0]+box[2],box[1]+box[3]
        cv2.rectangle(img_copy,(x1,y1),(x2,y2),(0,255,0),2)
    return img_copy 


#---将白键和黑键的位置对应标出来，对应youtube上的数据
def vis_white_black_loc_youtube(white_loc, boxes, save_path, img_lists, rect, base_info):
    
    black_num = [2, 5, 7, 10, 12, 14, 17, 19, 22, 24, 26, 29,
                          31, 34, 36, 38, 41, 43, 46, 48, 50, 53, 55, 58,
                          60, 62, 65, 67, 70, 72, 74, 77, 79, 82, 84, 86]
    white_num = [x for x in range(1, 89) if x not in black_num]
    txt_save_path=os.path.split(save_path)[0]
    for img_file in img_lists:
        ori_img = Image.open(img_file)
        ori_opencvimg = cv2.imread(img_file)
        ori_w, ori_h = ori_img.size
        img = ori_img.crop(rect)
        w,h = img.size 
        file_seq = os.path.basename(img_file).split('.')[0]
        opencv_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        img_copy = opencv_img.copy()
        height,width,_ = img_copy.shape 
        if base_info is not None:
            if base_info['warp_M'] is not None:
                rotated_img = cv2.warpAffine(ori_opencvimg, base_info['rote_M'], (ori_w, ori_h))
                warp_img = cv2.warpPerspective(rotated_img, base_info['warp_M'], (ori_w, ori_h))
                img_copy = warp_img[rect[1]:rect[3], rect[0]:rect[2]]
            elif base_info['rote_M'] is not None:
                #---对原图旋转然后裁剪
                rotated_img = cv2.warpAffine(ori_opencvimg,
                                             base_info['rote_M'], (ori_w, ori_h))
                img_copy = rotated_img[rect[1]:rect[3], rect[0]:rect[2]]
            else:
                img_copy = base_info['base_img'].copy()
        
        #----对于Record中的数据集这个字体大小要设置的小一点
        font_size = 0.4 if base_info is not None else 0.5
        font_thick = 1 if base_info is not None else 2
        for i,loc in enumerate(white_loc):
            if i==len(white_loc)-1:break
            key_num=white_num[i]
            cv2.putText(img_copy, str(key_num), (loc + 2, height - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), font_thick)

        for i,box in enumerate(boxes):   
            x1,y1= box[0],box[1]
            key_num=black_num[i]
            cv2.putText(img_copy, str(key_num), (x1 + 3, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, font_size + 0.1, (0, 255, 0), font_thick)
        save_img_path = os.path.join(save_path, os.path.basename(img_file))
        print(save_img_path)
        #----对于Tencent数据集,需要将上面的视图部分加进来,有多个视角方便看
        if 'level' in file_seq:
            top_rect = [rect[0], 0, rect[2], rect[1]]
            top_img = ori_img.crop(top_rect)
            top_opencv_img = cv2.cvtColor(np.asarray(top_img), cv2.COLOR_RGB2BGR)
            img_save = np.concatenate((top_opencv_img, img_copy), axis=0)
        else:
            img_save = img_copy.copy()
        cv2.imwrite(save_img_path, img_save)
        # break

def comp_class_vec(output_vec, num_classes, index=None):
    # if not index.any():
    #     index = np.argmax(output_vec.cpu().data.numpy())
    # else:
    index = np.array(index)
    index = index[np.newaxis,:]
    index = torch.from_numpy(index)
    #---[1 0 0 0 0 ...]
    one_hot = torch.zeros(1,num_classes).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output_vec.cpu())  # one_hot = 11.8605
    return class_vec

def gen_cam(feature_map, grads,input_size):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """ 
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
    #---weights->(16,)
    weights = np.mean(grads, axis=(1, 2))  #
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_size[1], input_size[0]))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    # path_cam_img = os.path.join(out_dir, os.path.basename(img_path))
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    return 255 * cam

def get_cam_img(input_size,img_draw,output,final_index,grad_block,fmap_block,num_classes):
    length = len(final_index)
    class_loss=comp_class_vec(output,num_classes,final_index)
    class_loss.backward()
    #--grads_val->(1,512,5,25)  就是网络最后一个卷积层的输出维度大小 c,h,w
    grads_val = grad_block[-1].cpu().data.numpy().squeeze(0)
    fmap = fmap_block[-1].cpu().data.numpy().squeeze(0)
    
    cam = gen_cam(fmap, grads_val,input_size)
    h, w = input_size
    img_show = np.float32(cv2.resize(img_draw, (w, h))) / 255
    cam_img=show_cam_on_image(img_show, cam)
    return cam_img

#----将音频算法检测得到的结果转换为对应的帧的结果---
def ConvertToNote(txt_path,file_mark,fps,offNum=20):
    with open(txt_path,'r') as f:
        lines=f.readlines()
    new_line=[]
    frame_nums=[]
    time_fre=1.0/fps
    for line in lines:
        line=line.strip().split()
        line[0]=int(float(line[0])/time_fre)
        data='frame{:0>4d}\t{}\n'.format(line[0],int(line[2])-offNum)
        new_line.append(data)
        frame_nums.append(int(line[0]))
    object_path=os.path.join(os.path.split(txt_path)[0],file_mark+'.txt')
    # label_path=os.path.join(os.path.split(txt_path)[0],file_mark+'_label.txt')
    with open(object_path,'w') as f:
        f.writelines(new_line)


if __name__=='__main__':
    audio_txt_path='/home/ccy/data/piano/videos/Tencent'
    txt_lists=[os.path.join(audio_txt_path,x ) for x in os.listdir(audio_txt_path)
               if x.endswith('.res')]
    txt_lists.sort()
    for txt_path in txt_lists:
        print(txt_path)
        file_mark=os.path.basename(txt_path).split('.')[0]
        fps=cfg.EVALUATE_MAP[file_mark]['fps']
        ConvertToNote(txt_path,file_mark,fps)