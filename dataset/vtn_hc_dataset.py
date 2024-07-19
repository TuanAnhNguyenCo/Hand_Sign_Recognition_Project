import json
import math
import os
from argparse import ArgumentParser
import numpy as np
from decord import VideoReader
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from dataset.dataset_utils import get_selected_indexs, pad_index, crop_hand
import cv2
import torchvision
from utils.video_augmentation import *





class VTN_HC_Dataset(Dataset):
    def __init__(self, base_url,split,dataset_cfg,train_labels = None,**kwargs):
     
        print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
        self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
   

        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)
        
    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                RandomVerticalFlip(),
                                RandomRotate(p=0.3),
                                RandomShear(0.3,0.3,p = 0.3),
                                Salt( p = 0.3),
                                GaussianBlur( sigma=1,p = 0.3),
                                ColorJitter(0.5, 0.5, 0.5,p = 0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']), 
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def count_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        cap.release()
        return total_frames,width,height
    
    def read_videos(self,name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
       
        path = f'{self.base_url}/videos/{name}'   
        vlen,width,height = self.count_frames(path)
       
        selected_index, pad = get_selected_indexs(vlen-5,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()
        vr = VideoReader(path,width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()

        poseflow_clip = []
        clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        for frame,frame_index in zip(frames,selected_index):
            if self.data_cfg['crop_two_hand']:
                
                kp_path = os.path.join(self.base_url,'poses',name.replace(".mp4",""),
                                    name.replace(".mp4","") + '_{:06d}_'.format(frame_index) + 'keypoints.json')
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())
                    
                    keypoints = np.array(value['pose_threshold_02']) # 26,3
                    x = 320*keypoints[:,0]/width
                    y = 256*keypoints[:,1]/height
                   
                keypoints = np.stack((x, y), axis=0)
               
           
            crops = None
            if self.data_cfg['crop_two_hand']:
                crops,missing_wrists_left, missing_wrists_right = crop_hand(frame,keypoints,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                       self.transform,len(clip),missing_wrists_left,missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

             # Try to impute hand crops from frames where the elbow and wrist weren't missing as close as possible temporally
            for clip_index in range(len(clip)):
                if clip_index in missing_wrists_left:
                    # Find temporally closest not missing frame for left wrist
                    replacement_index = -1
                    distance = np.inf
                    for ci in range(len(clip)):
                        if ci not in missing_wrists_left:
                            dist = abs(ci - clip_index)
                            if dist < distance:
                                distance = dist
                                replacement_index = ci
                    if replacement_index != -1:
                        clip[clip_index][0] = clip[replacement_index][0]
                # Same for right crop
                if clip_index in missing_wrists_right:
                    # Find temporally closest not missing frame for right wrist
                    replacement_index = -1
                    distance = np.inf
                    for ci in range(len(clip)):
                        if ci not in missing_wrists_right:
                            dist = abs(ci - clip_index)
                            if dist < distance:
                                distance = dist
                                replacement_index = ci
                    if replacement_index != -1:
                        clip[clip_index][1] = clip[replacement_index][1]


         
            
        clip = torch.stack(clip,dim = 0)
        return clip


    def __getitem__(self, idx):
        self.transform.randomize_parameters()

        data = self.train_labels.iloc[idx].values
        name,label = data[0],data[1]

        clip = self.read_videos(name)
        
        return clip,torch.tensor(label)

    
    def __len__(self):
        return len(self.train_labels)




