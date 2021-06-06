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

    df = pd.read_csv(data_path+'train_labels.csv')
    if config.DEBUG:
        df = pd.concat([df.query('target == 1').sample(len(df.query('target==1'))//1000), df.query('target == 0').sample(len(df.query('target == 0'))//10000)]).sample(frac=1).reset_index(drop=True)
    
    images = list(glob.glob(data_path+'train/*'))
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

    logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_bs{bs}_size{config.IMAGE_SIZE[0]}_dt{date_time}')
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
                if config.ORIG_IMAGE:
#                     for original images
                    train_images_path.append(data_path+'train/'+str(df.loc[int(id),'id'])[0]+'/'+str(df.loc[int(id),'id'])+'.npy')
                    train_targets.append(int(df.loc[int(id), 'target']))
                else:
#                    for resized images
                    filename = df.loc[int(id),'id']
                    train_images_path.append(f'{config.RESIZED_IMAGE_PATH}train/{filename}.npy')
                    train_targets.append(int(df.loc[int(id), 'target']))
    
            train_dataset = dataset.SetiDataset(image_paths = train_images_path,
                                                    targets = train_targets,
                                                    ids = trIDs,
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
                if config.ORIG_IMAGE:
#                 for original images
                    valid_images_path.append(data_path+'train/'+str(df.loc[int(id),'id'])[0]+'/'+str(df.loc[int(id),'id'])+'.npy')
                    valid_targets.append(int(df.loc[int(id), 'target']))
                else:
#                     for resized images
                    filename = df.loc[int(id),'id']
                    valid_images_path.append(f'{config.RESIZED_IMAGE_PATH}train/{filename}.npy')
                    valid_targets.append(int(df.loc[int(id), 'target']))
    
            valid_dataset = dataset.SetiDataset(image_paths = valid_images_path,
                                                targets = valid_targets,
                                                ids = vIDs,
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
                train_roc_auc = metrics.roc_auc_score(round(train_targets), train_predictions)
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
                                f'loss_{config.MODEL_OUTPUT_PATH}{config.MODEL_NAME}_fold{fold}_bs{bs}_size{config.IMAGE_SIZE[0]}_mixup{config.MIXUP}_dt{config.DATETIME}.pth')

                if valid_roc_auc >= best_valid_roc_auc:
                    best_valid_roc_auc = valid_roc_auc
                    torch.save({'model': model.state_dict(),
                                'valid_ids': valid_ids,
                                'predictions': predictions,
                                'valid_targets': valid_targets},
                                f'auc_{config.MODEL_OUTPUT_PATH}{config.MODEL_NAME}_fold{fold}_bs{bs}_size{config.IMAGE_SIZE[0]}_mixup{config.MIXUP}_dt{config.DATETIME}.pth')














