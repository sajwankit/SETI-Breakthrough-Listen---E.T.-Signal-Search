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

                
    def mixup(inputs, targets):
        lam = np.random.beta(config.MIXUP_APLHA, config.MIXUP_APLHA)
        batch_size = inputs.size()[0]
        index = torch.randperm(batch_size)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        targets1, targets2 = targets, targets[index]
        return mixed_inputs, targets1, targets2, lam

    def loss_criterion(outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

def train(data_loader, model, optimizer, device, scaler = None):
    #this function does training for one epoch

    losses = AverageMeter()

    #putting model to train mode
    model.train()   


    st = time.time()


    final_targets = []
    final_outputs = []
    final_ids = []

#    len_data_loader = len(data_loader)
    progressDisp_stepsize = 0.05
    progressDisp_step = 1
    #go over every batch of data in data_loader
    for batch_number, data in enumerate(data_loader):
        inputs = data['images']
        targets = data['targets']
        ids = data['ids']

        #moving inputs and targets to device: cpu or cuda
        inputs = inputs.to(device, dtype = torch.float)
        targets = targets.to(device, dtype = torch.float)

        #zero grad the optimizer
        optimizer.zero_grad()
        
        if config.MIXED_PRECISION:
            #mixed precision
            with amp.autocast():       
                if config.MIXUP:
                    #Forward Step
                    mixed_inputs, targets1, targets2, lam = mixup(inputs, targets)
                    outputs = model(mixed_inputs)
                    #calculate loss
                    loss = lam * loss_criterion(outputs, targets1) + (1 - lam) * loss_criterion(outputs, targets2)
                else:
                    #Forward Step
                    outputs = model(inputs)
                    #calculate loss
                    loss = loss_criterion(outputs, targets)
            #backward step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.MIXUP:
                    #Forward Step
                    mixed_inputs, targets1, targets2, lam = mixup(inputs, targets)
                    outputs = model(mixed_inputs)
                    #calculate loss
                    loss = lam * loss_criterion(outputs, targets1) + (1 - lam) * loss_criterion(outputs, targets2)
            else:
                #Forward Step
                outputs = model(inputs)
                #calculate loss
                loss = loss_criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # auc = metrics.roc_auc_score(targets.detach().cpu().numpy().tolist(), outputs.detach().cpu().numpy().tolist())
        # auc = metrics.roc_auc_score(targets, outputs)

        #update average loss, auc
        losses.update(loss.item(), config.BATCH_SIZE)
                
#
#        if batch_number == int(len_data_loader * progressDisp_stepsize) * progressDisp_step:
#            et = time.time()
#            print(f'batch: {batch_number} of {len_data_loader}, loss: {loss}. Time Elapsed: {(et-st)/60} minutes')
#            progressDisp_step = progressDisp_step*2

        final_targets.extend(targets.detach().cpu().numpy().tolist())
        final_outputs.extend(torch.sigmoid(outputs).detach().cpu().numpy().tolist())
        final_ids.extend(ids)
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
            outputs = model(inputs)

            loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))
            losses.update(loss.item(), config.BATCH_SIZE)

            targets = targets.detach().cpu().numpy().tolist()
            outputs = torch.sigmoid(outputs).detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(outputs)
            final_ids.extend(ids)
#
#            if batch_number == int(len_data_loader * progressDisp_stepsize) * progressDisp_step:
#                et = time.time()
#                print(f'batch: {batch_number} of {len_data_loader}, v_loss: {loss}. Time Elapsed: {(et-st)/60} minutes')
#                progressDisp_step = progressDisp_step*2

    return final_outputs, final_targets, final_ids, losses.avg


def predict(data_loader, model, device):
    #this function does evaluation for one epoch

#    len_data_loader = len(data_loader)
    progressDisp_stepsize = 0.05
    progressDisp_step = 1

    #putting model to eval mode
    model.eval()
    
    final_outputs = []

    #we use no_grad context:
    with torch.no_grad():

        st = time.time()
        for batch_number, data in enumerate(data_loader):
            inputs = data['images']
            inputs = inputs.to(device, dtype = torch.float)

            
            #do forward step to generate prediction
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs).detach().cpu().numpy().tolist()

            final_outputs.extend(outputs)

#            if batch_number == int(len_data_loader * progressDisp_stepsize) * progressDisp_step:
#                et = time.time()
#                print(f'batch: {batch_number} of {len_data_loader}. Time Elapsed: {(et-st)/60} minutes')
#                progressDisp_step = progressDisp_step*2

    return final_outputs

    

