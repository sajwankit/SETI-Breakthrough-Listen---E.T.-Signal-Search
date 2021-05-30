from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import numpy as np
import glob
import pandas as pd
import config
from skmultilearn.model_selection import IterativeStratification

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

def get_MSKFold(ids, multi_targets, nfolds, seed = 2021):
    mskf = IterativeStratification(n_splits=nfolds, order=1)
    msKFoldsData = []
    for fold, (idxT, idxV) in enumerate(mskf.split(np.array(ids), np.array(multi_targets))):
        msKFoldsData.append({'trIDs':idxT, 'vIDs':idxV})
    return msKFoldsData

# msKFoldsData = get_customSKFold(ids=[], multi_targets=[], nfolds=6, seed = 2021, shuffle = True)

# trl = pd.read_csv(f'{config.DATA_PATH}train_labels.csv')
# for i in range(4):
#     tri = msKFoldsData[i]['trIDs']
#     vi = msKFoldsData[i]['vIDs']
#     tr = trl.loc[trl.index.isin(tri)]
#     v = trl.loc[trl.index.isin(vi)]
#     # train['fold'] = train['fold'].astype(int)
#     # display(train.groupby(['fold', 'target']).size())
#     tr['g'] = tr['id'].apply(lambda x: x[0])
#     v['g'] = v['id'].apply(lambda x: x[0])

#     print(tr.g.value_counts())
#     print(v.g.value_counts())

#     print(tr.target.value_counts())
#     print(v.target.value_counts())
# # x = get_Kfold(len_samples = 50165, n_folds = 5, seed = 2021, shuffle = True)
# # print(x[2]['trIDs'][0])