import pandas as pd
import numpy as np
import glob
import argparse

import albumentations
import torch
import torchsample

from sklearn import metrics

import config
import dataset
import engine
import models
import validation_strategy as vs

from sklearn.model_selection import StratifiedKFold

class StratifiedSampler(torchsample.Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--nfolds', type = int)
    parser.add_argument('--fold', type = int)
    args = parser.parse_args()
    args.nfolds = 5
    args.fold = 1

    data_path = config.DATA_PATH
    device = config.DEVICE
    epochs = config.EPOCHS
    bs = config.BATCH_SIZE
    lr = config.LEARNING_RATE
    target_size = config.TARGET_SIZE

    df = pd.read_csv(data_path+'train_labels.csv')
    df = pd.concat([df.query('target == 1'), df.query('target == 0').sample(len(df.query('target == 0'))//7)]).sample(frac=1).reset_index(drop=True)
    images = list(glob.glob(data_path+'train/*'))
    targets = df.target.values

    model = models.Model(pretrained = True, target_size = target_size)
    model.to(device)

    
    skFoldData = vs.get_SKFold(ids = df.index.values,
                                targets = targets,
                                n_folds = args.nfolds,
                                seed = 2021,
                                shuffle = True)


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

        # train_loader = torch.utils.data.DataLoader(train_dataset,
        #                                                 sampler=StratifiedSampler(class_vector = torch.from_numpy(np.array(train_targets)),
        #                                                                                  batch_size = bs),
        #                                                 batch_size = bs
        #                                                 # batch_sampler = StratifiedBatchSampler(y = np.array(train_targets), batch_size = bs),
        #                                             )
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = bs,
                                            shuffle = True,
                                            num_workers = 4)

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
                                                    num_workers = 4)
        
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=config.FACTOR, patience=config.PATIENCE,
                                                                 verbose=True, eps=config.EPS)

        for epoch in range(epochs):
            train_predictions, train_targets, train_loss = engine.train(train_loader, model, optimizer, device)
            predictions, valid_targets, valid_loss = engine.evaluate(valid_loader, model, device)
            scheduler.step(valid_loss)
            train_roc_auc = metrics.roc_auc_score(train_targets, train_predictions)
            valid_roc_auc = metrics.roc_auc_score(valid_targets, predictions)
            print(f"Epoch = {epoch}, Valid Loss = {valid_loss}, Valid ROC AUC = {valid_roc_auc}, Train ROC AUC = {train_roc_auc}")















