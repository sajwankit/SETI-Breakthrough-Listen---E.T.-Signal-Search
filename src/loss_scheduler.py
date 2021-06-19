import config
import torch.nn as nn
import math

class Loss(nn.modules.Module):
    '''
    outputs means the output of the last layer of model
    '''
    def __init__(self, outputs, targets):
        super().__init__()
        self.outputs = outputs
        self.targets = targets
    
     
class BCEWithLogitsLoss(Loss):
    def forward(self):
        return nn.BCEWithLogitsLoss()(self.outputs, self.targets.view(-1,1))

class ArcLoss(Loss):
    '''
    W = Weight at last layer
    x = last layer feature
    Z = logits
    outputs = logits = Z = W*x = |W||x|cos(theta)
    normalised weights W and normalised features x sent here, |W|=1 |x|=1
    thus outputs become cos(theta)

    MAKE SURE model output takes care of above before using this loss
    '''
    def __init__(self, outputs, targets, feature_scale = 30.0, margin = 0.5, reduction = 'mean'):
        super().__init__(outputs, targets)
        self.feature_scale = feature_scale
        self.margin = margin
        self.reduction = reduction
        self.margin_cos = math.cos(margin)
        self.margin_sin = math.sin(margin)

    def forward(self):
        '''
        outputs = logits = cos(theta)
        margin added to theta cos(theta+margin)
        '''
        outputs = outputs.float()



