import torch
import torch.nn as nn

import math

import config

class Loss(nn.modules.Module):
    '''
    logits is output of the last layer of model
    '''
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='none', pos_weight=None):
        super().__init__()
        self.weight = weight,
        self.size_average = size_average,
        self.reduce = reduce,
        self.pos_weight = pos_weight,
        self.reduction = reduction
     
class BCEWithLogitsLoss(Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='none', pos_weight=None):
        super().__init__(weight, size_average, reduce, reduction, pos_weight)

    def forward(self, logits, targets):
        return nn.BCEWithLogitsLoss(reduction=self.reduction)(logits, targets.view(-1,1))

class ArcLoss(Loss):
    '''
    W = Weight at last layer
    x = last layer feature
    Z = logits
    logits = logits = Z = W*x = |W||x|cos(theta)
    normalised weights W and normalised features x sent here, |W|=1 |x|=1
    thus logits become cos(theta)

    MAKE SURE model logits takes care of above before using this loss
    '''
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='none', pos_weight=None, feature_scale=30.0, margin=0.5):
        super().__init__(weight, size_average, reduce, reduction, pos_weight)
        self.feature_scale = feature_scale
        self.margin = margin
        self.margin_cos = math.cos(margin)
        self.margin_sin = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, logits, targets):
        '''
        logits = logits = cos(theta)
        margin added to theta cos(theta+margin)
        '''
        logits = logits.float()
        logits_to_sine = torch.sqrt(1 - torch.pow(logits, 2))
        logits_plus_margin = logits*self.margin_cos-logits_to_sine*self.margin_sin
        logits_plus_margin = torch.where(logits > self.th, logits_plus_margin, logits-self.mm)
        targets_onehot = torch.FloatTensor(targets.size(0), config.TARGET_SIZE+1).to(targets.device)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1,1).long(), 1)

        logits_plus_margin = (targets_onehot*logits_plus_margin)+((1-targets_onehot)*logits)

        logits_plus_margin *= self.feature_scale
        loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)(logits_plus_margin, targets.long())
        
        return loss



