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

    df = pd.read_csv(data_path+'train_labels.csv')
#    df = pd.concat([df.query('target == 1').sample(len(df.query('target==1'))//100), df.query('target == 0').sample(len(df.query('target == 0'))//1000)]).sample(frac=1).reset_index(drop=True)
    # images = list(glob.glob(data_path+'train/*'))
    targets = df.target.values



    
#    skFoldData = vs.get_SKFold(ids = df.index.values,
#                                targets = targets,
#                                n_folds = config.FOLDS,
#                                seed = config.SEED,
#                                shuffle = True)


    ######################################################################
    # stratify based on target and image group
    id_keys_map = {'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15}
    id_keys = []
    for key in df['id'].values.tolist():
        try:
            key = int(key[0])
        except:
            key = id_keys_map[key[0]]
        id_keys.append(key)
    target = df['target'].values.tolist()
    multi_targets = [ [id_keys[x], target[x]] for x in range(len(df))]

    mskFoldData = vs.get_MSKFold(ids = df.index.values,
                                multi_targets = np.array(multi_targets),
                                nfolds = config.FOLDS,
                                seed = config.SEED)
    ########################################################################

    logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_fold{args.fold}_bs{bs}_dt{date_time}')
    logger.info(f'fold,epoch,val_loss,val_auc,tr_auc, time')

    for fold, foldData in enumerate(mskFoldData):
        if fold == args.fold or args.fold is None:
        
            #for every fold model should start from zero training
            model = models.Model(pretrained = True, training = True)
            model.to(device)
    
            trIDs = foldData['trIDs']
            vIDs = foldData['vIDs']
    
            train_images_path = []
            train_targets = []
            for id in trIDs:
                # #for original images
                # train_images_path.append(data_path+'train/'+str(df.loc[int(id),'id'])[0]+'/'+str(df.loc[int(id),'id'])+'.npy')
                # train_targets.append(int(df.loc[int(id), 'target']))

                #for resized images
                filename = df.loc[int(id),'id']
                train_images_path.append(f'{config.RESIZED_IMAGE_PATH}train/{filename}.npy')
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
                # #for original images
                # valid_images_path.append(data_path+'train/'+str(df.loc[int(id),'id'])[0]+'/'+str(df.loc[int(id),'id'])+'.npy')
                # valid_targets.append(int(df.loc[int(id), 'target']))

                #for resized images
                filename = df.loc[int(id),'id']
                valid_images_path.append(f'{config.RESIZED_IMAGE_PATH}train/{filename}.npy')
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
#    
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=config.FACTOR, patience=config.PATIENCE,
                                                                verbose=True, eps=config.EPS)
#            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 7, eta_min = 1e-7, last_epoch = -1)

    
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
                                'predictions': predictions,
                                'valid_targets': valid_targets},
                                f'{config.OUTPUT_PATH}{config.MODEL_NAME}_fold{fold}_dt{date_time}.pth')














