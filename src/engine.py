import torch
from torch._C import dtype
import torch.nn as nn
from tqdm import tqdm
import config
from tqdm import tqdm
import time
from sklearn import metrics
from torch.cuda import amp
import numpy as np
import utils

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
           
def get_loss(logits, targets, reduction='mean'):
    if config.NET == 'NetArcFace':
        loss = utils.ArcLoss(reduction=reduction, feature_scale=30, margin=0.2)(logits=logits, targets=targets, )
    else:
        loss = utils.BCEWithLogitsLoss(reduction=reduction)(logits, targets, )
    return loss

def loss_criterion(logits, targets):
    if config.OHEM_LOSS:
        batch_size = logits.size(0) 
        ohem_cls_loss = get_loss(logits, targets, reduction='none')
        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, 0, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*config.OHEM_RATE) )
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        cls_loss = ohem_cls_loss.sum() / keep_num
        return cls_loss
    else:
        return get_loss(logits, targets, reduction='mean')

def mixup(inputs, targets):
    lam = np.random.beta(config.MIXUP_APLHA, config.MIXUP_APLHA)
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size)
    mixed_inputs = np.sqrt(lam) * inputs + np.sqrt((1 - lam)) * inputs[index, :]
    targets1, targets2 = targets, targets[index]
    return mixed_inputs, targets1, targets2, lam    

def train(data_loader, model, optimizer, device, scaler = None):
    #this function does training for one epoch
#     print(f'ohem rate: {config.OHEM_RATE}')
    losses = AverageMeter()

    #putting model to train mode
    model.train()   

    st = time.time()

    final_targets = []
    final_outputs = []
    if config.NET == 'NetArcFace':
        final_output_confs = []
    final_ids = []

#    len_data_loader = len(data_loader)
    progressDisp_stepsize = 0.05
    progressDisp_step = 1
    #go over every batch of data in data_loader
    for batch_number, data in enumerate(data_loader):
        inputs = data['images']
        targets = data['targets']
        ids = data['ids']
        
#         print(f'{batch_number}, {np.unique(np.array(targets), return_counts=True)}, {torch.sum(ids)},{ids[0:4]}')

        #moving inputs and targets to device: cpu or cuda
        inputs = inputs.to(device, dtype = torch.float)
        targets = targets.to(device, dtype = torch.float)

        #zero grad the optimizer
        optimizer.zero_grad()
        
        if config.MIXED_PRECISION:
            ''' Mix Precision '''
            with amp.autocast():       
                if config.MIXUP:
                    #Forward Step
                    mixed_inputs, targets1, targets2, lam = mixup(inputs, targets)
                    logits = model(mixed_inputs)
                    #calculate loss
                    loss = np.sqrt(lam)*loss_criterion(logits, targets1)+np.sqrt(1 - lam)*loss_criterion(logits, targets2)
                else:
                    #Forward Step
                    logits = model(inputs)
                    #calculate loss
                    loss = loss_criterion(logits, targets)
            #backward step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.MIXUP:
                #Forward Step
                mixed_inputs, targets1, targets2, lam = mixup(inputs, targets)
                logits = model(mixed_inputs)
                #calculate loss
                loss = np.sqrt(lam)*loss_criterion(logits, targets1)+np.sqrt(1 - lam)*loss_criterion(logits, targets2)
            else:
                #Forward Step
                logits = model(inputs)
                #calculate loss
                loss = loss_criterion(logits, targets)
            loss.backward()
            optimizer.step()

        #update average loss, auc
        losses.update(loss.item(), config.BATCH_SIZE)

#        if batch_number == int(len_data_loader * progressDisp_stepsize) * progressDisp_step:
#            et = time.time()
#            print(f'batch: {batch_number} of {len_data_loader}, loss: {loss}. Time Elapsed: {(et-st)/60} minutes')
#            progressDisp_step = progressDisp_step*2

        
        if config.NET == 'NetArcFace':
            output_confs = logits.softmax(1)
            outputs = output_confs[:, 1]
        else:
            outputs = torch.sigmoid(logits)
        
        if config.NET == 'NetArcFace':
            final_output_confs.extend(output_confs.detach().cpu().numpy().tolist())
        final_outputs.extend(outputs.detach().cpu().numpy().tolist())
        final_targets.extend(targets.detach().cpu().numpy().tolist())
        final_ids.extend(ids)
        
    if config.NET == 'NetArcFace':
        return final_output_confs, final_outputs, final_targets, final_ids, losses.avg
    else:
        return final_outputs, final_targets, final_ids, losses.avg

def evaluate(data_loader, model, device):
    #this function does evaluation for one epoch

    losses = AverageMeter()


#    len_data_loader = len(data_loader)
    progressDisp_stepsize = 0.05
    progressDisp_step = 1

    #putting model to eval mode
    model.eval()
    
    final_targets = []
    final_outputs = []
    final_ids = []
    if config.NET == 'NetArcFace':
        final_output_confs = []

    #we use no_grad context:
    with torch.no_grad():

        st = time.time()
        for batch_number, data in enumerate(data_loader):
            inputs = data['images']
            targets = data['targets']
            ids = data['ids']
            
            inputs = inputs.to(device, dtype = torch.float)
            targets = targets.to(device, dtype = torch.float)

            #do forward step to generat prediction
            logits = model(inputs)

            loss = loss_criterion(logits, targets)
            losses.update(loss.item(), config.BATCH_SIZE)
            
            if config.NET == 'NetArcFace':
                output_confs = logits.softmax(1)
                outputs = output_confs[:, 1]
            else:
                outputs = torch.sigmoid(logits)
            
            if config.NET == 'NetArcFace':
                final_output_confs.extend(output_confs.detach().cpu().numpy().tolist())
            final_outputs.extend(outputs.detach().cpu().numpy().tolist())
            final_targets.extend(targets.detach().cpu().numpy().tolist())
            final_ids.extend(ids)

#            if batch_number == int(len_data_loader * progressDisp_stepsize) * progressDisp_step:
#                et = time.time()
#                print(f'batch: {batch_number} of {len_data_loader}, v_loss: {loss}. Time Elapsed: {(et-st)/60} minutes')
#                progressDisp_step = progressDisp_step*2
        
    if config.NET == 'NetArcFace':
        return final_output_confs, final_outputs, final_targets, final_ids, losses.avg
    else:
        return final_outputs, final_targets, final_ids, losses.avg


def predict(data_loader, model, device):
    #this function does evaluation for one epoch

#    len_data_loader = len(data_loader)
    progressDisp_stepsize = 0.05
    progressDisp_step = 1

    #putting model to eval mode
    model.eval()
#     print(model.training)
    final_outputs = []
    if config.NET == 'NetArcFace':
        final_output_confs = []
        
    #we use no_grad context:
    with torch.no_grad():

        st = time.time()
        for batch_number, data in enumerate(data_loader):
            inputs = data['images']
            inputs = inputs.to(device, dtype = torch.float)

            
            #do forward step to generate prediction
            logits = model(inputs)
            if config.NET == 'NetArcFace':
                output_confs = logits.softmax(1)
                outputs = output_confs[:, 1]
            else:
                outputs = torch.sigmoid(logits)
                
            if config.NET == 'NetArcFace':
                final_output_confs.extend(output_confs.detach().cpu().numpy().tolist())
            final_outputs.extend(outputs.detach().cpu().numpy().tolist())
            
#            if batch_number == int(len_data_loader * progressDisp_stepsize) * progressDisp_step:
#                et = time.time()
#                print(f'batch: {batch_number} of {len_data_loader}. Time Elapsed: {(et-st)/60} minutes')
#                progressDisp_step = progressDisp_step*2
    if config.NET == 'NetArcFace':
        return final_output_confs, final_outputs
    else:
        return final_outputs

    

