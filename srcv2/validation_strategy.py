from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import numpy as np
import glob
import pandas as pd
import config
# from skmultilearn.model_selection import IterativeStratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# create kfolds, return: list of tuples: (fold_number, training_indexes on that fold_number, validation indexes on that fold_number)
# if fold_number specified, return: list with single tuple: (fold_number, training_indexes on that fold_number, validation indexes on that fold_number) 
def get_Kfold(len_samples, n_folds, seed = 2021, shuffle = True):
    kFoldsData = {}
    kf = KFold(n_splits = n_folds,shuffle = shuffle,random_state = seed)
    for fold,(idxT,idxV) in enumerate(kf.split(np.arange(len_samples))):
        kFoldsData[fold] = {'trIDs':X[idxT], 'vIDs':X[idxV]}
    return kFoldsData

def get_SKFold(X, labels, n_folds, seed = 2021, shuffle = True):
    skFoldsData = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold,(idxT,idxV) in enumerate(skf.split(X, labels)):
        skFoldsData.append({'trIDs': X[idxT], 'vIDs': X[idxV]})
    return skFoldsData

def get_MSKFold(X, multi_labels, nfolds, seed = 2021):
    # mskf = IterativeStratification(n_splits=nfolds, order=1)
    mskf = MultilabelStratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)
    msKFoldsData = []
    for fold, (idxT, idxV) in enumerate(mskf.split(np.array(X), np.array(multi_labels))):
        msKFoldsData.append({'trIDs':X[idxT], 'vIDs':X[idxV]})
    return msKFoldsData

############TRYING THE VALIDATION STRATEGY######################################
# trl = pd.read_csv(f'{config.DATA_PATH}train_labels.csv')
# id_keys_map = {'a':10, 'b':11, 'c':12, 'd':13, 'e':14, 'f':15}
# id_keys = []
# for key in trl['id'].values.tolist():
#     try:
#         key = int(key[0])
#     except:
#         key = id_keys_map[key[0]]
#     id_keys.append(key)
# target = trl['target'].values.tolist()
# multi_labels = [ [id_keys[x], target[x]] for x in range(len(trl))]


# mskFoldData = get_MSKFold(ids = trl.index.values,
#                                 multi_labels = np.array(multi_labels),
#                                 nfolds = config.FOLDS,
#                                 seed = config.SEED)

# for i in range(4):
#     tri = mskFoldData[i]['trIDs']
#     vi = mskFoldData[i]['vIDs']
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