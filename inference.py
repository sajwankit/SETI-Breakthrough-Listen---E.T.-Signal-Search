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

    inference_images = list(glob.glob(data_path+'test/*'))

    model = models.Model(pretrained = True, target_size = target_size)
    model.to(device)

    logger = seedandlog.init_logger(log_name = f'{config.MODEL_NAME}_bs_{bs}.pth')
    logger.info(f'fold,epoch,val_loss,val_auc,tr_auc, time')

    model = models.Model(pretrained = False, target_size = target_size)
    model.to(device)
    MODEL_DIR = config.OUTPUT_PATH
    states = [torch.load(MODEL_DIR+f'{config.MODEL_NAME}_fold{fold}_best_loss.pth') for fold in range(4)]
    test_dataset = dataset.SetiDataset(image_paths = inference_images)
    test_loader = torch.utils.DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                                         shuffle=False, 
                                        num_workers=4, pin_memory=True)
    predictions_all_folds = []
    for fold in range(4):
        model.load_state_dict(states[fold]['model'])                                    
        predictions = engine.predict(model, test_loader, device)
        predictions_all_folds.append(predictions)

    