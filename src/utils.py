import config
import torch.nn as nn
import math

class Loss(nn.modules.Module):
    '''
    logits is output of the last layer of model
    '''
    def __init__(self, logits, targets, reduction='none'):
        super().__init__()
        self.logits = logits
        self.targets = targets
        self.reduction = reduction
     
class BCEWithLogitsLoss(Loss):
    def __init__(self):
        super().__init__(logits, targets, reduction)
    def forward(self):
        return nn.BCEWithLogitsLoss()(self.logits, self.targets.view(-1,1), reduction=reduction)

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
    def __init__(self, logits, targets, feature_scale=30.0, margin=0.5):
        super().__init__(logits, targets, reduction)
        self.feature_scale = feature_scale
        self.margin = margin
        self.margin_cos = math.cos(margin)
        self.margin_sin = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self):
        '''
        logits = logits = cos(theta)
        margin added to theta cos(theta+margin)
        '''
        logits = logits.float()
        logits_to_sine = torch.sqrt(1 - torch.pow(logits, 2))
        logits_plus_margin = logits*self.margin_cos-logits_to_sine*self.margin_sin
        logits_plus_margin = torch.where(logits > self.th, logits_plus_margin, logits-self.mm)
        
        # labels2 = torch.zeros_like(logits)
        # labels.scatter_(1, labels.view(-1, 1).long(), 1)
        targets = self.targets.view(-1,1)
        logits_plus_margin = (targets*logits_plus_margin)+((1-targets)*logits)

        logits_plus_margin *= self.feature_scale

        loss = nn.BCEWithLogitsLoss()(logits_plus_margin, self.targets.view(-1,1), reduction = reduction)
        
        return loss



