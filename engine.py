import torch
from torch._C import dtype
import torch.nn as nn
from tqdm import tqdm
import config
from tqdm import tqdm
import time
from sklearn import metrics

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

def train(data_loader, model, optimizer, device):
    #this function does training for one epoch

    losses = AverageMeter()

    #putting model to train mode
    model.train()   


    st = time.time()


    final_targets = []
    final_outputs = []

    len_data_loader = len(data_loader)
    progressDisp_stepsize = 0.05
    progressDisp_step = 1
    #go over every batch of data in data_loader
    for batch_number, data in enumerate(data_loader):
        inputs = data['images']
        targets = data['targets']

        #moving inputs and targets to device: cpu or cuda
        inputs = inputs.to(device, dtype = torch.float)
        targets = targets.to(device, dtype = torch.float)

        #zero grad the optimizer
        optimizer.zero_grad()

        #Forward Step
        outputs = model(inputs)

        #calculate loss
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

        #backward step
        loss.backward()
        
        # auc = metrics.roc_auc_score(targets.detach().cpu().numpy().tolist(), outputs.detach().cpu().numpy().tolist())
        # auc = metrics.roc_auc_score(targets, outputs)

        #update average loss, auc
        losses.update(loss.item(), config.BATCH_SIZE)
        
        optimizer.step()

        if batch_number == int(len_data_loader * progressDisp_stepsize) * progressDisp_step:
            et = time.time()
            print('Batch_number: '+str(batch_number)+' of '+str(len_data_loader)+', loss: '+str(loss)+'. Time elapsed: '+str((et-st)//60)+' minutes')
            progressDisp_step = progressDisp_step*2

        final_targets.extend(targets.detach().cpu().numpy().tolist())
        final_outputs.extend(outputs.detach().cpu().numpy().tolist())
    return final_outputs, final_targets, losses.avg


def evaluate(data_loader, model, device):
    #this function does evaluation for one epoch

    losses = AverageMeter()

    #putting model to eval mode
    model.eval()

    final_targets = []
    final_outputs = []

    #we use no_grad context:
    with torch.no_grad():

        for data in tqdm(data_loader):
            inputs = data['images']
            targets = data['targets']
            inputs = inputs.to(device, dtype = torch.float)
            targets = targets.to(device, dtype = torch.float)

            #do forward step to generat prediction
            outputs = model(inputs)

            loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))
            losses.update(loss.item(), config.BATCH_SIZE)

            targets = targets.detach().cpu().numpy().tolist()
            outputs = outputs.detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(outputs)

    return final_outputs, final_targets, losses.avg


    

