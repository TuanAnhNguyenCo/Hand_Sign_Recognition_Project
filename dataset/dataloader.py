from dataset.dataset_utils import get_selected_indexs,pad_array,pad_index
from dataset.dataset import build_dataset
import torch
from functools import partial
import random
import numpy as np
import os
import torchvision
import json
import math      
import random
import cv2



def vtn_hc_collate_fn_(batch):
    labels = torch.stack([s[1] for s in batch],dim = 0)
    clip = torch.stack([s[0] for s in batch],dim = 0) 
    return {'video':clip},labels


def build_dataloader(cfg, split, is_train=True, model = None,labels = None):
    dataset = build_dataset(cfg['data'], split,model,train_labels = labels)

    if cfg['data']['model_name'] == 'vtn_hc':
        collate_func = vtn_hc_collate_fn_
  

    dataloader = torch.utils.data.DataLoader(dataset,
                                            collate_fn = collate_func,
                                            batch_size = cfg['training']['batch_size'],
                                            num_workers = cfg['training'].get('num_workers',2),
                                            shuffle = is_train,
                                            prefetch_factor = cfg['training'].get('prefetch_factor',2),
                                            # pin_memory=True,
                                            persistent_workers =  True,
                                            # sampler = sampler
                                            )
    # return dataloader, sampler
    return dataloader
