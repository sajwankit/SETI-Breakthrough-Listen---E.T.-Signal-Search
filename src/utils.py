import torch
import torch.nn as nn

import math

import config


'''
LOSSES
'''
class Loss(nn.modules.Module):
    '''
    logits is output of the last layer of model
    '''
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='none', pos_weight=None):
        super().__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.pos_weight = pos_weight
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
#         print(targets_onehot)

        logits_plus_margin = (targets_onehot*logits_plus_margin)+((1-targets_onehot)*logits)

        logits_plus_margin *= self.feature_scale
        loss = torch.nn.CrossEntropyLoss(reduction=self.reduction)(logits_plus_margin, targets.long())
        
        return loss
    

'''
OPTIMIZER AND SCHEDULER
'''
class OptSch:
    def __init__(self, sch=None, opt='Adam'):
        self.lr = config.INIT_LEARNING_RATE
        self.opt = opt
        self.sch = sch
        self.eta_min = config.ETA_MIN
        self.T_0 = config.T_0
        self.T_max=config.T_MAX
    
    def get_opt_sch(self, model):
        if self.opt=='Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)
            
        if self.sch=='CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                             T_0 = self.T_0,
                                                                             eta_min = self.eta_min,
                                                                             last_epoch = -1)
        elif self.sch=='ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=config.FACTOR, patience=config.PATIENCE,
                                                                verbose=True, eps=config.EPS)
        elif self.sch=='CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max = self.T_max,
                                                                   eta_min = self.eta_min,
                                                                   last_epoch = -1)
        elif self.sch==None:
            scheduler=None
        return optimizer, scheduler
