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


if __name__ == '__main__':
    seedandlog.seed_torch(seed=config.SEED)

    data_path = config.DATA_PATH
    device = config.DEVICE
    bs = config.BATCH_SIZE
    target_size = config.TARGET_SIZE

    model = models.Model(pretrained = False, target_size = target_size)
    model.to(device)
    states = [torch.load(config.OUTPUT_PATH+f'{config.MODEL_NAME}_fold{fold}_best_loss.pth') for fold in range(4)]

    print(states[0]['predictions'])
    def get_oof_df(state):
        df = pd.DataFrame({'predictions': np.array([]), 'targets': np.array([])})
        df['predictions'] = np.array(state['predictions']).reshape(-1)
        df['targets'] = np.array(state['valid_targets']).reshape(-1)
        return df

    oof_df = None
    for fold in range(4):
        _oof_df = get_oof_df(states[fold])
        oof_df = pd.concat([oof_df, _oof_df])

    oof_auc = metrics.roc_auc_score(oof_df['targets'].values, oof_df['predictions'].values)

    logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_bs_{bs}.pth')
    logger.info(f'Final OOF ROC AUC SCORE: {oof_auc}')


    def get_test_file_path(image_id):
            return f"{data_path}test/{image_id[0]}/{image_id}.npy"

    inference_df = pd.read_csv(data_path+'sample_submission.csv')[:10]
    inference_df['image_path'] = inference_df['id'].apply(get_test_file_path)

    
    test_dataset = dataset.SetiDataset(image_paths = inference_df['image_path'].values.tolist())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                                         shuffle=False, 
                                        num_workers=4, pin_memory=True)

    
    predictions_all_folds = []
    mean_predictions = np.array([0]*len(inference_df))
    for fold in range(4):
        model.load_state_dict(states[fold]['model'])                                    
        predictions = engine.predict(test_loader, model, device)
        mean_predictions = mean_predictions + np.array(predictions)
        # predictions_all_folds.append(predictions)
    mean_predictions = mean_predictions/4

    inference_df['target'] = mean_predictions
    inference_df[['id', 'target']].to_csv(config.OUTPUT_PATH+'submission.csv', index=False)
    print(inference_df[['id', 'target']].head())
    