import pandas as pd
import numpy as np
import torch
from sklearn import metrics

import config
import dataset
import engine
import models
import validation_strategy as vs
import seedandlog
import os

from tqdm import tqdm

if __name__ == '__main__':
    seedandlog.seed_torch(seed=config.SEED)

    data_path = config.DATA_PATH
    device = config.DEVICE
    bs = config.BATCH_SIZE
    target_size = config.TARGET_SIZE
    
    saved_model_name = config.SAVED_MODEL_NAME
    
    model = models.Model(pretrained = False)
    model.to(device)
    states = [torch.load(f'{config.MODEL_OUTPUT_PATH}{config.MODEL_LOAD_FOR_INFER}_fold{fold}_{saved_model_name}.pth') for fold in range(config.FOLDS)]

    def get_oof_df(state):
        df = pd.DataFrame({'predictions': np.array([]), 'targets': np.array([])})
        df['predictions'] = np.array(state['predictions']).reshape(-1)
        df['targets'] = np.array(state['valid_targets']).reshape(-1)
        df['target_ids'] = np.array(state['valid_ids']).reshape(-1)
        return df

    oof_df = None
    for fold in range(config.FOLDS):
        _oof_df = get_oof_df(states[fold])
        _oof_df.to_csv(f'{config.LOG_DIR}{config.MODEL_LOAD_FOR_INFER}_oof_df_fold{fold}_{saved_model_name}.csv', index = False)
        oof_df = pd.concat([oof_df, _oof_df])

    oof_df.to_csv(f'{config.LOG_DIR}{config.MODEL_LOAD_FOR_INFER}_oof_df_{saved_model_name}.csv', index = False)
    oof_auc = metrics.roc_auc_score(oof_df['targets'].values, oof_df['predictions'].values)
    
    # logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_bs{bs}_size{config.IMAGE_SIZE[0]}_dt{config.DATETIME}')
    # logger.info(f'Final OOF ROC AUC SCORE: {oof_auc}')
    print(f'Final OOF ROC AUC SCORE: {oof_auc}')
    if config.INFER == True:
        def get_test_file_path(image_id):
            if config.ORIG_IMAGE:
                return f"{data_path}test/{image_id[0]}/{image_id}.npy"
            else:
                return f"{config.RESIZED_IMAGE_PATH}test/{image_id}.npy"
            
        if config.DEBUG:
            inference_df = pd.read_csv(data_path+'sample_submission.csv')[:10]
        else:
            inference_df = pd.read_csv(data_path+'sample_submission.csv')
            
        inference_df['image_path'] = inference_df['id'].apply(get_test_file_path)


        test_dataset = dataset.SetiDataset(image_paths = inference_df['image_path'].values.tolist(), ids = inference_df.index.values.tolist())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                                             shuffle=False, 
                                            num_workers=4)


        predictions_all_folds = []
        mean_predictions = np.array([0]*len(inference_df))
        print('Inference ON ! \n')
        for fold in tqdm(range(config.FOLDS)):
            model.load_state_dict(states[fold]['model'])                                    
            predictions = engine.predict(test_loader, model, device)
            mean_predictions = mean_predictions + np.array(predictions)
            # predictions_all_folds.append(predictions)
        mean_predictions = mean_predictions/config.FOLDS

        inference_df['target'] = mean_predictions
        inference_df[['id', 'target']].to_csv(f'{config.LOG_DIR}submission_cv{oof_auc}_{config.MODEL_LOAD_FOR_INFER}_{saved_model_name}', index=False)
        print(inference_df[['id', 'target']].head())
    
