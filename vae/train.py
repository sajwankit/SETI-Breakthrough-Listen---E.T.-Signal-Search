import pandas as pd
import numpy as np
import glob
import argparse
import time

import albumentations
import torch

from sklearn import metrics

import utils
import config
import dataset
import engine
import models
import validation_strategy as vs
import seedandlog
import sampler

import vae

import warnings
warnings.filterwarnings("ignore")

from torch.cuda import amp
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == '__main__':
    seedandlog.seed_torch(seed=config.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type = int)
    args = parser.parse_args()

    saved_model_name = config.SAVED_MODEL_NAME

    data_path = config.DATA_PATH
    device = config.DEVICE
    epochs = config.EPOCHS
    bs = config.BATCH_SIZE
    lr = config.INIT_LEARNING_RATE
    target_size = config.TARGET_SIZE

    df = pd.read_csv(data_path+'train_labels.csv')
    if config.DEBUG:
        ids = [x.split('/')[-1].split('.')[0] for x in glob.glob(f'{config.RESIZED_IMAGE_PATH}train/*.npy')][:320]
        df = df[df['id'].isin(ids)]
        df.reset_index(inplace = True, drop = True)
    images = list(glob.glob(data_path+'train/*'))
    targets = df.target.values

    '''
    stratify based on target and image group
    ''' 
    skFoldData = vs.get_SKFold(ids = df.index.values,
                               targets = targets,
                               n_folds = config.FOLDS,
                               seed = config.SEED,
                               shuffle = True)


    '''
    stratify based on target and image group
    ''' 
#     id_keys_map = {'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15}
#     id_keys = []
#     for key in df['id'].values.tolist():
#         try:
#             key = int(key[0])
#         except:
#             key = id_keys_map[key[0]]
#         id_keys.append(key)
#     target = df['target'].values.tolist()
#     multi_targets = [ [id_keys[x], target[x]] for x in range(len(df))]

#     mskFoldData = vs.get_MSKFold(ids = df.index.values,
#                                 multi_targets = np.array(multi_targets),
#                                 nfolds = config.FOLDS,
#                                 seed = config.SEED)

    logger = seedandlog.init_logger(log_name = f'{saved_model_name}')
    if config.NET == 'VAE':
        logger.info(f'fold,epoch,valid_loss,valid_recon_loss, valid_kld_loss, train_loss, train_recon_loss, train_kld_loss, time')
    else:
        logger.info(f'fold,epoch,val_loss,val_auc,tr_auc, train_loss, time')

    for fold, foldData in enumerate(skFoldData):
        if fold == args.fold or args.fold is None:
        
            #for every fold model should start from zero training
            if config.NET == 'VAE':
                model = vae.VariationalAutoencoder(pretrained=True, )
            else:
                model = models.get_model(pretrained=True, net_out_features=config.TARGET_SIZE)
            model.to(device)
            
            if config.LOAD_SAVED_MODEL:
                states = torch.load(f'{config.MODEL_OUTPUT_PATH}auc_fold{fold}_{saved_model_name}.pth')
                model.load_state_dict(states['model'])
                optimizer.load_state_dict(states['optimizer'])
                scheduler.load_state_dict(states['scheduler'])
                scaler.load_state_dict(states['scaler'])
                print('Saved Model LOADED!')
                start_epoch = states['epoch']
            else:
                start_epoch = 0
                
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

            if config.DEBUG:
                train_targets = (np.random.rand(240) > 0.9).astype(int)

            train_dataset = dataset.SetiDataset(image_paths = train_images_path,
                                                    targets = train_targets,
                                                    ids = trIDs,
                                                    resize = None,
                                                    augmentations = True)
            
#             train_loader = torch.utils.data.DataLoader(train_dataset, pin_memory = True,
#                                                         batch_sampler = sampler.StratifiedSampler( ids = trIDs,
#                                                                                             targets = train_targets,
#                                                                                             batch_size = config.BATCH_SIZE),                              
#                                                 num_workers = 8,
#                                                 worker_init_fn = seedandlog.seed_torch(seed=config.SEED)
#                                                 )


            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size = bs,
                                                shuffle = True,
                                                num_workers = 4,
                                                worker_init_fn = seedandlog.seed_torch(seed=config.SEED),
                                                      pin_memory = True)
    
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

            if config.DEBUG:
                valid_targets = (np.random.rand(80) > 0.9).astype(int)

            valid_dataset = dataset.SetiDataset(image_paths = valid_images_path,
                                                targets = valid_targets,
                                                ids = vIDs,
                                                resize = None,
                                                augmentations = False)
                                                    
            valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                        batch_size = bs,
                                                        shuffle = True,
                                                        num_workers = 4,
                                                        worker_init_fn = seedandlog.seed_torch(seed=config.SEED),
                                                      pin_memory = True)

#             valid_loader = torch.utils.data.DataLoader(valid_dataset, pin_memory = True,
#                                                         batch_sampler = sampler.StratifiedSampler( ids = vIDs,
#                                                                                             targets = valid_targets,
#                                                                                             batch_size = config.BATCH_SIZE),                              
#                                                 num_workers = 8,
#                                                 worker_init_fn = seedandlog.seed_torch(seed=config.SEED)
#                                                 )

            optimizer, scheduler = utils.OptSch(opt=config.OPTIMIZER, sch=config.SCHEDULER).get_opt_sch(model)

            if config.MIXED_PRECISION:
                #mixed precision training
                scaler = amp.GradScaler()
            else:
                scaler = None

            best_valid_loss = np.inf
            best_valid_roc_auc = -np.inf

            '''
            Training ON
            '''
            for epoch in range(start_epoch, epochs):
                
                if config.OHEM_LOSS:
                    if epoch >= (config.EPOCHS -12):
                        initial_rate = 1
                        final_rate = 0.4
                        config.OHEM_RATE = initial_rate + (epoch - (config.EPOCHS-12))*(final_rate - initial_rate)/( (config.EPOCHS-1) - (config.EPOCHS-12) )
                        print(f'applying ohem with rate {config.OHEM_RATE}')
                    else:
                        config.OHEM_RATE = 1
                
                st = time.time()
                if config.NET == 'NetArcFace':
                    train_prediction_confs, train_predictions, train_targets, train_ids, train_loss = engine.train(train_loader, model, optimizer, device, scaler)
                    prediction_confs, predictions, valid_targets, valid_ids, valid_loss = engine.evaluate(valid_loader, model, device)
#                     train_roc_auc = metrics.roc_auc_score(np.array(train_targets), np.array(train_predictions)[:,1])
#                     valid_roc_auc = metrics.roc_auc_score(np.array(valid_targets), np.array(predictions)[:,1])
                else:
                    if config.NET == 'VAE':
                        train_loss, train_recon_loss, train_kld_loss = engine.train(train_loader, model, optimizer, device, scaler)
                        valid_loss, valid_recon_loss, valid_kld_loss = engine.evaluate(valid_loader, model, device)
                    else:
                        train_predictions, train_targets, train_ids, train_loss = engine.train(train_loader, model, optimizer, device, scaler)
                        predictions, valid_targets, valid_ids, valid_loss = engine.evaluate(valid_loader, model, device)
                
                if config.NET != 'VAE':
                    train_roc_auc = metrics.roc_auc_score(np.array(train_targets), np.array(train_predictions))
                    valid_roc_auc = metrics.roc_auc_score(np.array(valid_targets), np.array(predictions))
                    
                if config.SCHEDULER == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss)
                else:
                    scheduler.step()

                et = time.time()


                if config.NET == 'VAE':
                    logger.info(f'{fold},{epoch},{valid_loss},{valid_recon_loss},{valid_kld_loss},{train_loss},{train_recon_loss},{train_kld_loss}, {(et-st)/60}')
                else:
                    # train auc doesnot make sense when using mixup
                    logger.info(f'{fold},{epoch},{valid_loss},{valid_roc_auc},{train_roc_auc},{train_loss}, {(et-st)/60}')
                
                if valid_loss <= best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save({'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'scaler': scaler.state_dict(),
                                'epoch': epoch,
                                'valid_ids': valid_ids if config.NET != 'VAE' else 0,
                                'predictions': predictions if config.NET != 'VAE' else 0,
                                'prediction_confs': prediction_confs if config.NET == 'NetArcFace' else 1,
                                'valid_targets': valid_targets if config.NET != 'VAE' else 0},
                                f'{config.MODEL_OUTPUT_PATH}loss_fold{fold}_{saved_model_name}.pth')

                if valid_roc_auc >= best_valid_roc_auc:
                    best_valid_roc_auc = valid_roc_auc
                    torch.save({'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'scaler': scaler.state_dict(),
                                'epoch': epoch,
                                'valid_ids': valid_ids if config.NET != 'VAE' else 0,
                                'predictions': predictions if config.NET != 'VAE' else 0,
                                'prediction_confs': prediction_confs if config.NET == 'NetArcFace' and config.NET != 'VAE' else 1,
                                'valid_targets': valid_targets if config.NET != 'VAE' else 0},
                                f'{config.MODEL_OUTPUT_PATH}auc_fold{fold}_{saved_model_name}.pth')














