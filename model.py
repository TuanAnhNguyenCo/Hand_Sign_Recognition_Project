from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F



class MMTensorNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, x):
        mean = torch.mean(x, dim=self.dim).unsqueeze(self.dim)
        std = torch.std(x, dim=self.dim).unsqueeze(self.dim)
        return (x - mean) / std


class VTN_HC(nn.Module):
    def __init__(self, num_classes=199, num_heads=8, num_layers=4, embed_size=512, sequence_length=16, cnn='rn34',
                freeze_layers=0, dropout=0 ,**kwargs):
        super().__init__()

        self.sequence_length = sequence_length
        self.embed_size = embed_size
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(cnn, embed_size, freeze_layers)

        num_attn_features = embed_size*2
        self.norm = MMTensorNorm(-1)
        self.bottle_mm = nn.Linear(num_attn_features, num_attn_features)

        self.self_attention_decoder = SelfAttention(num_attn_features, num_attn_features,
                                                    [num_heads] * num_layers,
                                                    self.sequence_length, layer_norm=True)
        self.classifier = LinearClassifier(num_attn_features, num_classes, dropout)
        self.relu = F.relu
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, video = None,**kwargs):
        
        bs,t,x,c,w,h = video.shape # two hands with x = 2
        features = self.feature_extractor(video.view(bs,t*x,c,w,h))
        z = features.view(bs,t,-1)
        zp = self.self_attention_decoder(z)

        # cls
        y = self.classifier(zp) # 1->bs is video1

        return {
            'logits':y.mean(1),
        } # train