import os, numpy as np
import io, torch, torchvision
from PIL import Image
from video_augmentation import *


def build_video_transform(dataset_cfg,split):
    transform = Compose(
                        Scale(dataset_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                        CenterCrop(dataset_cfg['vid_transform']['IMAGE_SIZE']), 
                        ToFloatTensor(),
                        PermuteImage(),
                        Normalize(dataset_cfg['vid_transform']['NORM_MEAN_IMGNET'],dataset_cfg['vid_transform']['NORM_STD_IMGNET']))
    return transform

def get_selected_indexs(vlen, num_frames=64, is_train=True, setting=['consecutive', 'pad', 'central', 'pad'],temporal_stride = 2):
    pad = None  #pad denotes the number of padding frames
    assert len(setting) == 4
    # denote train > 64, test > 64, test < 64
    train_p, train_m, test_p, test_m = setting
    if train_p == 'fusion': 
        train_p = np.random.choice(['consecutive', 'random','segment','center_stride'])
    assert train_p in ['consecutive', 'random','segment','center_stride']
    assert train_m in ['pad']
    assert test_p in ['central', 'start', 'end','segment','center_stride']
    assert test_m in ['pad', 'start_pad', 'end_pad']
    if num_frames > 0:
        assert num_frames%4 == 0
        if is_train:
            if vlen > num_frames:
                
                if train_p == 'consecutive':
                    start = np.random.randint(0, vlen - num_frames, 1)[0]
                    selected_index = np.arange(start, start+num_frames)
                elif train_p == 'center_stride':
                    frame_start = (vlen - num_frames) // (2 * temporal_stride)
                    frame_end = frame_start + num_frames * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > vlen:
                        frame_end = vlen
                    selected_index = list(range(frame_start, frame_end, temporal_stride))
                    while len(selected_index) < num_frames:
                        selected_index.append(selected_index[-1])
                    selected_index = np.array(selected_index)
                elif train_p == 'random':
                    # random sampling
                    selected_index = np.arange(vlen)
                    np.random.shuffle(selected_index)
                    selected_index = selected_index[:num_frames]  #to make the length equal to that of no drop
                    selected_index = sorted(selected_index)
                elif train_p == "segment":
                    data_chunks = np.array_split(range(vlen), num_frames)
                    random_elements = np.array([np.random.choice(chunk) for chunk in data_chunks])
                    selected_index = sorted(random_elements)
                else:
                    selected_index = np.arange(0, vlen)
            elif vlen < num_frames:
                if train_m == 'pad':
                    remain = num_frames - vlen
                    selected_index = np.arange(0, vlen)
                    pad_left = np.random.randint(0, remain, 1)[0]
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
            else:
                selected_index = np.arange(0, vlen)
        
        else:
            if vlen >= num_frames:
                start = 0
                if test_p == 'central':
                    start = (vlen - num_frames) // 2
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == 'center_stride':
                    frame_start = (vlen - num_frames) // (2 * temporal_stride)
                    frame_end = frame_start + num_frames * temporal_stride
                    if frame_start < 0:
                        frame_start = 0
                    if frame_end > vlen:
                        frame_end = vlen
                    selected_index = list(range(frame_start, frame_end, temporal_stride))
                    while len(selected_index) < num_frames:
                        selected_index.append(selected_index[-1])
                    selected_index = np.array(selected_index)
                elif test_p == 'start':
                    start = 0
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == 'end':
                    start = vlen - num_frames
                    selected_index = np.arange(start, start+num_frames)
                elif test_p == "segment":
                    data_chunks = np.array_split(range(vlen), num_frames)
                    random_elements = np.array([np.random.choice(chunk) for chunk in data_chunks])
                    selected_index = sorted(random_elements)
                else: 
                    selected_index = np.arange(start, start+num_frames)
            else:
                remain = num_frames - vlen
                selected_index = np.arange(0, vlen)
                if test_m == 'pad':
                    pad_left = remain // 2
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'start_pad':
                    pad_left = 0
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                elif test_m == 'end_pad':
                    pad_left = remain
                    pad_right = remain - pad_left
                    pad = (pad_left, pad_right)
                else:
                    selected_index = np.arange(0, vlen)
    else:
        # for statistics
        selected_index = np.arange(vlen)

    return selected_index, pad


def pil_list_to_tensor(pil_list, int2float=True):
    func = torchvision.transforms.PILToTensor()
    tensors = [func(pil_img) for pil_img in pil_list]
    #list of C H W
    tensors = torch.stack(tensors, dim=0)
    if int2float:
        tensors = tensors/255
    return tensors #T,C,H,W


def pad_array(array, l_and_r):
    left, right = l_and_r
    if left > 0:
        pad_img = array[0]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([left]+[1]*(len(array.shape)-1)))
        array = np.concatenate([pad, array], axis=0)
    if right > 0:
        pad_img = array[-1]
        pad = np.tile(np.expand_dims(pad_img, axis=0), tuple([right]+[1]*(len(array.shape)-1)))
        array = np.concatenate([array, pad], axis=0)
    return array

def pad_index(index_arr, l_and_r) :
    left, right = l_and_r
    index_arr = index_arr.tolist()
    index_arr = left*[index_arr[0]] + index_arr + right*[index_arr[-1]]
    return np.array(index_arr)

def crop_hand(frame,keypoints,WRIST_DELTA,SHOULDER_DIST_EPSILON,
              transform,clip_len,missing_wrists_left,missing_wrists_right):
    left_wrist_index = 9
    left_elbow_index = 7
    right_wrist_index = 10
    right_elbow_index = 8

    # Crop out both wrists and apply transform
    left_wrist = keypoints[0:2, left_wrist_index]
    left_elbow = keypoints[0:2, left_elbow_index]

    left_hand_center = left_wrist + WRIST_DELTA * (left_wrist - left_elbow)
    left_hand_center_x = left_hand_center[0]
    left_hand_center_y = left_hand_center[1]
    shoulder_dist = np.linalg.norm(keypoints[0:2, 5] - keypoints[0:2, 6]) * SHOULDER_DIST_EPSILON
    left_hand_xmin = max(0, int(left_hand_center_x - shoulder_dist // 2))
    left_hand_xmax = min(frame.shape[1], int(left_hand_center_x + shoulder_dist // 2))
    left_hand_ymin = max(0, int(left_hand_center_y - shoulder_dist // 2))
    left_hand_ymax = min(frame.shape[0], int(left_hand_center_y + shoulder_dist // 2))

    if not np.any(left_wrist) or not np.any(
                    left_elbow) or left_hand_ymax - left_hand_ymin <= 0 or left_hand_xmax - left_hand_xmin <= 0:
                # Wrist or elbow not found -> use entire frame then
                left_hand_crop = frame
                missing_wrists_left.append(clip_len) # I tried this and achived 93% on test
                
    else:
        left_hand_crop = frame[left_hand_ymin:left_hand_ymax, left_hand_xmin:left_hand_xmax, :]
    if not isinstance(left_hand_crop,np.ndarray):
        left_hand_crop = transform(left_hand_crop.numpy())
    else:
        left_hand_crop = transform(left_hand_crop)

    right_wrist = keypoints[0:2, right_wrist_index]
    right_elbow = keypoints[0:2, right_elbow_index]
    right_hand_center = right_wrist + WRIST_DELTA * (right_wrist - right_elbow)
    right_hand_center_x = right_hand_center[0]
    right_hand_center_y = right_hand_center[1]
    right_hand_xmin = max(0, int(right_hand_center_x - shoulder_dist // 2))
    right_hand_xmax = min(frame.shape[1], int(right_hand_center_x + shoulder_dist // 2))
    right_hand_ymin = max(0, int(right_hand_center_y - shoulder_dist // 2))
    right_hand_ymax = min(frame.shape[0], int(right_hand_center_y + shoulder_dist // 2))

    if not np.any(right_wrist) or not np.any(
                right_elbow) or right_hand_ymax - right_hand_ymin <= 0 or right_hand_xmax - right_hand_xmin <= 0:
            # Wrist or elbow not found -> use entire frame then
            right_hand_crop = frame
            missing_wrists_right.append(clip_len) # I tried this and achived 93% on test
            
    else:
        right_hand_crop = frame[right_hand_ymin:right_hand_ymax, right_hand_xmin:right_hand_xmax, :]
    if not isinstance(right_hand_crop,np.ndarray):
        right_hand_crop = transform(right_hand_crop.numpy())
    else:
        right_hand_crop = transform(right_hand_crop)

    crops = torch.stack((left_hand_crop, right_hand_crop), dim=0)
    # left_hand_crop = cv2.resize(left_hand_crop,(224,224))
    # right_hand_crop = cv2.resize(right_hand_crop,(224,224))
    # new_img = np.concatenate((right_hand_crop,left_hand_crop),axis = 1)
    # crops = transform(crops)

   

    return crops,missing_wrists_left,missing_wrists_right