import timm

import torch
import torch.nn as nn
from torch.nn import functional as F

import math

import config
import utils

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform(self.weight)

    def forward(self, features):
        '''
        logits = cosine, as:
        W = Weight at last layer
        x = last layer feature
        Z = logits
        logits = Z = W*x = |W||x|cos(theta)
        theta = angle between W and x
        normalised weights W and normalised features x, |W|=1 |x|=1
        thus logits become cos(theta)
        '''
        logits = F.linear(F.normalize(features), F.normalize(self.weight))
        return logits

class Backbone(nn.Module):
    def __init__(self, pretrained=True, backbone_out_features=1):
        super(Backbone, self).__init__()
        self.backbone = timm.create_model(config.MODEL_NAME,
                                            pretrained=pretrained, in_chans=config.CHANNELS)

        if config.MODEL_NAME in ['resnet18', 'resnet18d']:
            self.in_features = self.backbone.fc.in_features
        else:
            self.in_features = self.backbone.classifier.out_features
        self.backbone.fc = nn.Linear(self.in_features, self.backbone_out_features)
        print(f'\n Using {config.MODEL_NAME} as backbone, model output layer: {self.output_layer} with DROPOUT {config.DROPOUT}\n')

    def forward(self, x):
        backbone_logits = self.backbone(x)
        return backbone_logits

class Net(nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        self.backbone_out_features = config.TARGET_SIZE
        self.backbone = Backbone(pretrained=pretrained, backbone_out_features=self.backbone_out_features)

        self.arcface_metric = ArcMarginProduct(self.backbone.fc.out_features, config.TARGET_SIZE)

    def forward(self, x):
        backbone_logits = self.backbone(x)
        arcface_metric = self.arcface_metric(backbone_logits)
        if config.DROPOUT:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    logits = self.output_layer(dropout(base_model_out))
                else:
                    logits += self.output_layer(dropout(base_model_out))
            logits /= len(self.dropouts)
            return logits
        else:
            return backbone_logits, arcface_metric
        

# class Model(nn.Module):
#     def __init__(self, pretrained=True ):
#         super().__init__()
#         self.base_model = timm.create_model(config.MODEL_NAME,
#                                             pretrained=pretrained, in_chans=config.CHANNELS)
#         if config.MODEL_NAME in ['resnet18', 'resnet18d']:
#             self.in_features = self.base_model.fc.in_features
#         else:
#             self.in_features = self.base_model.classifier.out_features
            
            
#         if config.DROPOUT:
#             self.base_model_out_features = 1024*10
#             self.base_model.fc = nn.Linear(self.in_features, self.base_model_out_features)
#             self.output_layer = nn.Linear(self.base_model_out_features, config.TARGET_SIZE)
#         else:
#             self.base_model_out_features = config.TARGET_SIZE
#             self.base_model.fc = nn.Linear(self.in_features, self.base_model_out_features)
#             self.output_layer = self.base_model.fc
        
        
#         self.dropouts = nn.ModuleList([
#             nn.Dropout(0.7) for _ in range(16)
#         ])
# #         print(self.model)
#         print(f'\nUsing {config.MODEL_NAME}, model output layer: {self.output_layer} with DROPOUT {config.DROPOUT}\n')

#     def forward(self, x):
#         base_model_logits = self.base_model(x)
#         if config.DROPOUT:
#             for i, dropout in enumerate(self.dropouts):
#                 if i == 0:
#                     logits = self.output_layer(dropout(base_model_out))
#                 else:
#                     logits += self.output_layer(dropout(base_model_out))
#             logits /= len(self.dropouts)
#             return logits
#         else:
#             return base_model_logits
        