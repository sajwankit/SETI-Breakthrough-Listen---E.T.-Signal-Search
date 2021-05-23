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

if __name__ == '__main__':
    seedandlog.seed_torch(seed=config.SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument( '--nfolds', type = int)
    parser.add_argument('--fold', type = int)
    args = parser.parse_args()
    args.nfolds = 3
    args.fold = 1

    data_path = config.DATA_PATH
    device = config.DEVICE
    epochs = config.EPOCHS
    bs = config.BATCH_SIZE
    lr = config.LEARNING_RATE
    target_size = config.TARGET_SIZE

    df = pd.read_csv(data_path+'train_labels.csv')
    df = pd.concat([df.query('target == 1').sample(len(df.query('target==1'))//40), df.query('target == 0').sample(len(df.query('target == 0'))//400)]).sample(frac=1).reset_index(drop=True)
    images = list(glob.glob(data_path+'train/*'))
    targets = df.target.values

    model = models.Model(pretrained = True, target_size = target_size)
    model.to(device)

    
    skFoldData = vs.get_SKFold(ids = df.index.values,
                                targets = targets,
                                n_folds = args.nfolds,
                                seed = 2021,
                                shuffle = True)

    logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_bs_{bs}.pth')
    logger.info(f'fold,epoch,val_loss,val_auc,tr_auc, time')

    for fold, foldData in enumerate(skFoldData):
        trIDs = foldData['trIDs']
        vIDs = foldData['vIDs']

        train_images_path = []
        train_targets = []
        for id in trIDs:
            train_images_path.append(data_path+'train/'+str(df.loc[int(id),'id'])[0]+'/'+str(df.loc[int(id),'id'])+'.npy')
            train_targets.append(int(df.loc[int(id), 'target']))

        train_dataset = dataset.SetiDataset(image_paths = train_images_path,
                                                targets = train_targets,
                                                resize = None,
                                                augmentations = None)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = bs,
                                            shuffle = True,
                                            num_workers = 4,
                                            worker_init_fn = seedandlog.seed_torch(seed=config.SEED))

        valid_images_path = []
        valid_targets = []
        for id in vIDs:
            valid_images_path.append(data_path+'train/'+str(df.loc[int(id),'id'])[0]+'/'+str(df.loc[int(id),'id'])+'.npy')
            valid_targets.append(int(df.loc[int(id), 'target']))

        valid_dataset = dataset.SetiDataset(image_paths = valid_images_path,
                                            targets = valid_targets,
                                            resize = None,
                                            augmentations = None)
                                                
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                    batch_size = bs,
                                                    shuffle = True,
                                                    num_workers = 4,
                                                    worker_init_fn = seedandlog.seed_torch(seed=config.SEED))
        
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=config.FACTOR, patience=config.PATIENCE,
                                                            verbose=True, eps=config.EPS)

        # logger.info(f'***************************************************************************************************************************')
        # logger.info(f'fold: {fold}, device: {device}, batch_size: {bs}, model_name: {config.MODEL_NAME}, scheduler: ReduceLROnPlateau, lr: {lr}, seed: {config.SEED}')
        # logger.info(f'***************************************************************************************************************************')

        best_valid_loss = 999
        for epoch in range(epochs):
            st = time.time()
            train_predictions, train_targets, train_loss = engine.train(train_loader, model, optimizer, device)
            predictions, valid_targets, valid_loss = engine.evaluate(valid_loader, model, device)
            scheduler.step(valid_loss)
            train_roc_auc = metrics.roc_auc_score(train_targets, train_predictions)
            valid_roc_auc = metrics.roc_auc_score(valid_targets, predictions)
            et = time.time()
            logger.info(f'{fold},{epoch},{valid_loss},{valid_roc_auc},{train_roc_auc},{(et-st)/60}')
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                torch.save({'model': model.state_dict(), 
                            'predictions': predictions},
                            config.OUTPUT_PATH+f'{config.MODEL_NAME}_fold{fold}_best_loss.pth')















