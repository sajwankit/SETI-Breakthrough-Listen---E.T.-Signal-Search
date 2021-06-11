import pandas as pd
import numpy as np
import glob
import argparse
import time

import albumentations
import torch

from sklearn import metrics

import config
import dataset
import engine
import models
import validation_strategy as vs
import seedandlog

from torch.cuda import amp
torch.multiprocessing.set_sharing_strategy('file_system')
if __name__ == '__main__':
    seedandlog.seed_torch(seed=config.SEED)
    date_time = config.DATETIME
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type = int)
    args = parser.parse_args()

    data_path = config.DATA_PATH
    device = config.DEVICE
    epochs = config.EPOCHS
    bs = config.BATCH_SIZE
    lr = config.LEARNING_RATE
    target_size = config.TARGET_SIZE


    needle_target_encoding = {
                'brightpixel':0,
                'narrowband': 1,
                'narrowbanddrd': 2,
                'noise': 3, 
                'squarepulsednarrowand': 4,
                'squiggle': 5,
                'squigglesquarepulsednarrowband': 6
                }


    train_image_paths = glob.glob(f'{config.NEEDLE_PATH}*train/*.png')
    train_targets = [needle_target_encoding(x.split('/')[-2]) for p in train_image_paths]

    logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_bs{bs}_size{config.IMAGE_SIZE[0]}_mixup{config.MIXUP}_aug{config.AUG}_dt{date_time}')
    logger.info(f'fold,epoch,val_loss,val_auc,tr_auc, train_loss, time')

    model = models.Model(pretrained = True, training = True)
    model.to(device)

    train_dataset = dataset.SetiNeedleDataset(image_paths = train_image_paths,
                                                    targets = train_targets,
                                                    resize = None,
                                                    augmentations = True)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size = bs,
                                        shuffle = True,
                                        num_workers = 4,
                                        worker_init_fn = seedandlog.seed_torch(seed=config.SEED))


    valid_image_paths = glob.glob(f'{config.NEEDLE_PATH}*valid/*.png')
    valid_targets = [needle_target_encoding(x.split('/')[-2]) for p in valid_image_paths]

    valid_dataset = dataset.SetiNeedleDataset(image_paths = valid_image_paths,
                                                    targets = valid_targets,
                                                    resize = None,
                                                    augmentations = True)
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                        batch_size = bs,
                                        shuffle = True,
                                        num_workers = 4,
                                        worker_init_fn = seedandlog.seed_torch(seed=config.SEED))


    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
#    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        factor=config.FACTOR, patience=config.PATIENCE,
                                                        verbose=True, eps=config.EPS)
    #            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 7, eta_min = 1e-7, last_epoch = -1)

    if config.MIXED_PRECISION:
        #mixed precision training
        scaler = amp.GradScaler()
    else:
        scaler = None

    best_valid_loss = 999
    best_valid_roc_auc = -999
    for epoch in range(epochs):
        st = time.time()
        train_predictions, train_targets, train_ids, train_loss = engine.train(train_loader, model, optimizer, device, scaler)
        predictions, valid_targets, valid_ids, valid_loss = engine.evaluate(valid_loader, model, device)
        scheduler.step(valid_loss)
        train_roc_auc = metrics.roc_auc_score(train_targets, train_predictions)
        valid_roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        et = time.time()

        # train auc doesnot make sense when using mixup
        logger.info(f'{fold},{epoch},{valid_loss},{valid_roc_auc},{train_roc_auc},{train_loss}, {(et-st)/60}')
        
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({'model': model.state_dict(),
                        'valid_ids': valid_ids,
                        'predictions': predictions,
                        'valid_targets': valid_targets},
                        f'{config.MODEL_OUTPUT_PATH}loss_{config.MODEL_NAME}_fold{fold}_bs{bs}_size{config.IMAGE_SIZE[0]}_mixup{config.MIXUP}_aug{config.AUG}_dt{config.DATETIME}.pth')

        if valid_roc_auc >= best_valid_roc_auc:
            best_valid_roc_auc = valid_roc_auc
            torch.save({'model': model.state_dict(),
                        'valid_ids': valid_ids,
                        'predictions': predictions,
                        'valid_targets': valid_targets},
                        f'{config.MODEL_OUTPUT_PATH}auc_{config.MODEL_NAME}_fold{fold}_bs{bs}_size{config.IMAGE_SIZE[0]}_mixup{config.MIXUP}_aug{config.AUG}_dt{config.DATETIME}.pth')










