from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import numpy as np
import glob
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import config

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

def get_customSKFold(ids, multi_targets, nfolds, seed = 2021, shuffle = True):
    tr = pd.read_csv(f'{config.DATA_PATH}train_labels.csv')
    ids = [x for x in range(len(tr))]
    id_keys_map = {'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15}
    id_keys = []
    for key in tr['id'].values.tolist():
        try:
            key = int(key[0])
        except:
            key = id_keys_map[key[0]]
        id_keys.append(key)
    target = tr['target'].values.tolist()

    multi_targets = [ [id_keys[x], target[x]] for x in range(len(tr))]
    mskf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    msKFoldsData = []
    for fold, (idxT, idxV) in enumerate(mskf.split(ids, multi_targets)):
        msKFoldsData.append({'trIDs':idxT, 'vIDs':idxV})
    return msKFoldsData

msKFoldsData = get_customSKFold(ids=[], multi_targets=[], nfolds=6, seed = 2021, shuffle = True)

trl = pd.read_csv(f'{config.DATA_PATH}train_labels.csv')
tri = msKFoldsData[0]['trIDs']
vi = msKFoldsData[0]['vIDs']
tr = trl.loc[~trl.index.isin(tri)]
v = trl.loc[~trl.index.isin(vi)]
# train['fold'] = train['fold'].astype(int)
# display(train.groupby(['fold', 'target']).size())
tr['g'] = tr['id'].apply(lambda x: x[0])
v['g'] = v['id'].apply(lambda x: x[0])

print(tr.g.value_counts())
print(v.g.value_counts())

print(tr.target.value_counts())
print(v.target.value_counts())
# x = get_Kfold(len_samples = 50165, n_folds = 5, seed = 2021, shuffle = True)
# print(x[2]['trIDs'][0])