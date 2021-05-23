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

    data_path = config.DATA_PATH
    device = config.DEVICE
    bs = config.BATCH_SIZE
    target_size = config.TARGET_SIZE
    MODEL_DIR = config.OUTPUT_PATH

    model = models.Model(pretrained = False, target_size = target_size)
    model.to(device)
    states = [torch.load(MODEL_DIR+f'{config.MODEL_NAME}_fold{fold}_best_loss.pth') for fold in range(4)]

    def get_oof_df(state):
        df = pd.DataFrame({'predictions': None, 'targets': None})
        df['predictions'] = state['predictions'].values
        df['targets'] = state['valid_targets'].values
        return df

    oof_df = None
    for fold in range(4):
        _oof_df = get_oof_df(states[fold])
        oof_df = pd.concat([oof_df, _oof_df])

    oof_auc = metrics.roc_auc_score(oof_df['targets'].values, oof_df['predictions'].values)

    logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_bs_{bs}.pth')
    logger.info(f'Final OOF ROC AUC SCORE: {oof_auc}')


    def get_test_file_path(id):
            return data_path+'train/'+str(inference_df.loc[int(id),'id'])[0]+'/'+str(inference_df.loc[int(id),'id'])+'.npy'
    inference_df = pd.read_csv(data_path+'sample_submission.csv')
    inference_df['image_path'] = inference_df['id'].apply(get_test_file_path)

    
    test_dataset = dataset.SetiDataset(image_paths = inference_df['image_path'].values.tolist())
    test_loader = torch.utils.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                                         shuffle=False, 
                                        num_workers=4, pin_memory=True)

    
    predictions_all_folds = []
    mean_predictions = np.array([0]*len(inference_df))
    for fold in range(4):
        model.load_state_dict(states[fold]['model'])                                    
        predictions = engine.predict(model, test_loader, device)
        mean_predictions = mean_predictions + np.array(predictions)
        # predictions_all_folds.append(predictions)
    mean_predictions = mean_predictions/4

    inference_df['target'] = mean_predictions
    inference_df[['id', 'target']].to_csv(config.OUTPUT_PATH+'submission.csv', index=False)
    print(inference_df[['id', 'target']].head())
    