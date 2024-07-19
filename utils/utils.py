import torch.nn as nn
import torch.optim as optim
from model.vtn_hc import VTN_HC
import torch
from trainer.tools import MyCustomLoss,OLM_Loss
from torchvision import models
from torch.nn import functional as F
from collections import OrderedDict

def load_criterion(train_cfg):
    criterion = None
    if train_cfg['criterion'] == "MyCustomLoss":
        criterion = MyCustomLoss(label_smoothing=train_cfg['label_smoothing'])
    if train_cfg['criterion'] == "OLM_Loss": 
        criterion = OLM_Loss(label_smoothing=train_cfg['label_smoothing'])
    assert criterion is not None
    return criterion

def load_optimizer(train_cfg,model):
    optimzer = None
    if train_cfg['optimzer'] == "SGD":
        optimzer = optim.SGD(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']),momentum=0.9,nesterov=True)
    if train_cfg['optimzer'] == "Adam":
        optimzer = optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']))
    assert optimzer is not None
    return optimzer

def load_lr_scheduler(train_cfg,optimizer):
    scheduler = None
    if train_cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
    if train_cfg['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['gamma'])
    assert scheduler is not None
    return scheduler




def load_model(cfg):
    if cfg['training']['pretrained']:
        print(f"load pretrained model: {cfg['training']['pretrained_model']}")
        if cfg['data']['model_name'] == 'vtn_hc':
            if '.ckpt' in cfg['training']['pretrained_model']:
                model = VTN_HC(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                new_state_dict = {}
                for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        if 'bottle_mm' in key:
                            continue
                        new_state_dict[key.replace('model.','')] = value
                model.reset_head(226) # AUTSL
                model.load_state_dict(new_state_dict,strict = False)
                model.reset_head(model.num_classes)
                
            else:
                model = VTN_HC(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
       
    else:
        if cfg['data']['model_name'] == 'vtn_hc':
            model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
       
    
    assert model is not None
    print("loaded model")
    return model
        