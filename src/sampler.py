import torch as th
import math
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import config

class Sampler:
    def __init__(self, data_source):
        pass
    def __iter__(self):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    '''
    takes in original indices belonging to a fold and corresponding target
    Returns batches of original indices of image samples
    noteL target input has only role to create batches.
    target corresponding to batched indices will be target[batch indices]
    '''  
    def __init__(self, X, labels, batch_size, oversample_rate=0):
        self.labels = np.array(labels)
        self.X = np.array(X)
        self.batch_size = batch_size
        self.X_batches = []
        self.oversample_rate = oversample_rate
        if self.oversample_rate > 0:
            self.oversample(self.X, self.labels)
        
    def oversample(self, X, labels):
        target_pos_indices = np.argwhere(self.labels).reshape(-1,)
        X_pos = self.X[target_pos_indices]
        self.X = np.append(self.X, np.tile(self.X[target_pos_indices], self.oversample_rate))
#         print(self.X)
        self.labels = np.append(self.labels, np.tile(self.labels[target_pos_indices], self.oversample_rate))
#         print(self.labels)s
          
    def make_batches(self, X, labels, seed_per_epoch):
        unique, counts = np.unique(labels, return_counts=True)
        if (counts[0] < 2 or counts[1] <2) or len(X) <= self.batch_size:
            self.X_batches.append(X)
        else:

            s = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state=seed_per_epoch)
            left_batch_indices, right_batch_indices = [x for x in s.split(X, labels)][0]
            self.make_batches(X[left_batch_indices], labels[left_batch_indices], seed_per_epoch=seed_per_epoch)
            self.make_batches(X[right_batch_indices], labels[right_batch_indices], seed_per_epoch=seed_per_epoch)

    def gen_sample_array(self):
            
        self.X_batches = []
        seed_per_epoch = np.random.randint(10,20000)
        self.make_batches(self.X, self.labels, seed_per_epoch=seed_per_epoch)
        for X_batch in self.X_batches:
#             print(X_batch)
            yield list(X_batch)

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.labels)

    
'''
SANITY CHECK

import os
os.chdir('/home/asajw/SETI/src')
import sampler
import numpy as np
X = np.arange(32*7)
labels = (np.random.rand(32*7) > 0.5).astype(int)
a = sampler.StratifiedSampler(trIDs, targets[trIDs], 32, oversample_rate=3)
test_X = np.array([])
for i, X_batch in enumerate(a.X_batches):
    target_value, counts = np.unique(targets[a.X_batches[i]], return_counts=True)
    print(f'batch:{i}, batch_size {len(X_batch)}, target {target_value[0]} counts: {counts[0]}, target {target_value[1]} counts: {counts[1]}, {np.sum(a.X[np.searchsorted(a.X, a.X_batches[i])])},{a.X[np.searchsorted(a.X, a.X_batches[i])][0:4]}')
    test_X = np.concatenate([test_X, a.X[np.searchsorted(a.X, a.X_batches[0])]])
print(np.all(np.sort(test_X) == np.sort(a.X)))

'''