import os
from tqdm.auto import tqdm
import numpy as np
import torchvision
import yaml
import torchvision
import pandas as pd
from dataset_utils import build_video_transform,get_selected_indexs,pad_index,crop_hand
import torch
import random
import time
from decord import VideoReader
from mmpose.apis import MMPoseInferencer


class VTNInferenceDataset():
    def __init__(self,video_url,label_url,pose_detector):
        self.frames, _, _ = torchvision.io.read_video(video_url,pts_unit='sec')
        self.video_url = video_url
        self.label_url = label_url
        self.pose_detector = pose_detector
        self.__setup__()
    
    def __setup__(self):
        
        self.keypoints = self.detect_keypoints()
        l_center,r_center,LHips,RHips = self.get_finger_hip(self.keypoints)
        self.start_idx,self.end_idx = self.get_start_and_end_position(l_center,r_center,LHips,RHips)
        labels = pd.read_csv(self.label_url)['label'].values
        self.idx_to_classes = {idx: label for idx,label in enumerate(labels)}
        
    def get_finger_hip(self,keypoints):
        l_center = []
        r_center = []
        LHips = []
        RHips = []
        left_wrist_index = 9
        left_elbow_index = 7
        right_wrist_index = 10
        right_elbow_index = 8

        for kp in keypoints: # n,26,3
        
            kp = kp[:,:2].T # 2,26

            # Crop out both wrists and apply transform
            left_wrist = kp[0:2, left_wrist_index]
            left_elbow = kp[0:2, left_elbow_index]

            left_hand_center = left_wrist + 0.5*(left_wrist - left_elbow)
            l_center.append(left_hand_center)
            
            right_wrist = kp[0:2, right_wrist_index]
            right_elbow = kp[0:2, right_elbow_index]
            right_hand_center = right_wrist + 0.5*(right_wrist - right_elbow)
            r_center.append(right_hand_center)
            
            LHips.append( kp[0:2, 11])
            RHips.append( kp[0:2, 12])

        return l_center,r_center,LHips,RHips

    def get_start_and_end_position(self,l_center,r_center,LHips,RHips):
        frame_status = []
        for Lfinger,RFinger,LHip,RHip in zip(l_center,r_center,LHips,RHips):
            
            if (Lfinger[1] > LHip[1] and RFinger[1] > LHip[1]) or (Lfinger[1] > RHip[1] and RFinger[1] > RHip[1]):
                frame_status.append(0) # rest
            else:  
                frame_status.append(1) # do action

        start_idx = []
        end_idx = []
        is_start = False
        n_frame = len(frame_status)
        frame_status[0] = 0 # rest
        for i in range(len(frame_status)-1):
            if frame_status[i+1] - frame_status[i] != 0:
                if not is_start:
                    start_idx.append(max(0,i-1)) # shift left 2 frames
                    is_start = True
                else:
                    is_start = False
                    end_idx.append(min(n_frame-1,i+2)) # shift right 2 frames
        frame_status = np.array(frame_status)
        # if the last frame is still a action
        if len(end_idx) < len(start_idx):
            end_idx.append(n_frame-1)
        for s,e in zip(start_idx,end_idx):
            frame_status[s:e+1] = 1

        start_idx = []
        end_idx = []    
        is_start = False
        frame_status[0] = 0
        for i in range(len(frame_status)-1):
            if frame_status[i+1] - frame_status[i] != 0:
                if not is_start:
                    start_idx.append(i)
                    is_start = True
                else:
                    is_start = False
                    end_idx.append(i+1)
        # if the last frame is still a action
        if len(end_idx) < len(start_idx):
            end_idx.append(n_frame-1)
        # preprocessing
        new_start = []
        new_end = []
        for s,e in zip(start_idx,end_idx):
            if e - s >= 10:
                new_start.append(s)
                new_end.append(e)
        return  new_start, new_end
        
    def detect_keypoints(self): 

        keypoints  = []
        pose_results = self.pose_detector(self.video_url)
        for idx,pose_result in enumerate(pose_results):
            kp = pose_result['predictions'][0][0]['keypoints']
            prob = pose_result['predictions'][0][0]['keypoint_scores']
            pose_threshold_02 = [[value[0],value[1],0] if prob[idx] > 0.2 else [0,0,0] for idx,value in enumerate(kp)]
            keypoints.append(pose_threshold_02)
        return np.array(keypoints)  
    
    
class Inference:
    def __init__(self,cfg_url = 'inference_config/config.yaml',pose_model=None):
        self.cfg_url = cfg_url
        self.device = 'cpu'
        self.__setup_preprocess_model__()
        self.__setup__()
    
    def __setup__(self):
        with open(self.cfg_url, "r", encoding="utf-8") as ymlfile:
            self.cfg = yaml.safe_load(ymlfile)
        self.seed_everything(self.cfg['training']['random_seed'])
        self.data_cfg = self.cfg['data']
        self.vid_transform = build_video_transform(self.cfg['data'],'test')
        
        self.sign_recognition_model = torch.jit.load("Weights/vtn_vn_sign.pt")
        self.sign_recognition_model.eval()
        
        
    def __setup_preprocess_model__(self):
        
        self.pose_detector = MMPoseInferencer( "rtmpose-m_8xb512-700e_body8-halpe26-256x192")

    

        
    def seed_everything(self,seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def get_frames(self,s,e,frames,keypoints):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        selected_index, pad = get_selected_indexs(e-s+1,self.data_cfg['num_output_frames'],False,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
        if pad is not None:
            selected_index  = pad_index(selected_index,pad).tolist()
        clip = []
        missing_wrists_left = []
        missing_wrists_right = []
        for idx in selected_index:
            frame = frames[s+idx]
            kp = keypoints[s+idx].T
            crops,missing_wrists_left,missing_wrists_right = crop_hand(frame,kp,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                    self.vid_transform,len(clip),missing_wrists_left,missing_wrists_right)
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

        clip = torch.stack(clip, dim=0)

        return clip


    
    def predict(self,video_url):
        self.seed_everything(self.cfg['training']['random_seed'])
        print("Start Inference")
        start_time = time.time()
        dataset = VTNInferenceDataset(video_url,'inference_config/200_sign_words.csv',self.pose_detector)
        predictions = []
        for s,e in zip(dataset.start_idx,dataset.end_idx):
            clip = self.get_frames(s,e,dataset.frames,dataset.keypoints).unsqueeze(dim = 0)
            with torch.no_grad():
                outputs = self.sign_recognition_model(clip)
            pred = outputs[0].argmax(axis = -1).item()
            predictions.append(pred)
        end_time = time.time()
        print("Time:",end_time-start_time)
        sentence = [dataset.idx_to_classes[idx] for idx in predictions]
        print("Finish Inference")
        return sentence,end_time-start_time
       
      
        
if __name__ == "__main__":
    infer = Inference()
    sentence = infer.predict('chungtoi_muon_baogio.mp4')
    print(sentence)
    
    