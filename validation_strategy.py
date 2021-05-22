from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import numpy as np

# create kfolds, return: list of tuples: (fold_number, training_indexes on that fold_number, validation indexes on that fold_number)
# if fold_number specified, return: list with single tuple: (fold_number, training_indexes on that fold_number, validation indexes on that fold_number) 
def get_Kfold(len_samples, n_folds, seed = 2021, shuffle = True):
    kFoldsData = {}
    kf = KFold(n_splits = n_folds,shuffle = shuffle,random_state = seed)
    for fold,(idxT,idxV) in enumerate(kf.split(np.arange(len_samples))):
        kFoldsData[fold] = {'trIDs':idxT, 'vIDs':idxV}
    return kFoldsData

def get_SKFold(ids, targets, n_folds, seed = 2021, shuffle = True):
    skFoldsData = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold,(idxT,idxV) in enumerate(skf.split(ids, targets)):
        skFoldsData.append({'trIDs':idxT, 'vIDs':idxV})
    return skFoldsData


# train['fold'] = train['fold'].astype(int)
# display(train.groupby(['fold', 'target']).size())


# x = get_Kfold(len_samples = 50165, n_folds = 5, seed = 2021, shuffle = True)
# print(x[2]['trIDs'][0])