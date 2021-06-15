import torch as th
import math
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import config

class Sampler():

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class StratifiedSampler(Sampler):
    #treat ids as X, targets as Y. dont confuse ids with indices
    def __init__(self, ids, targets, batch_size):
        self.targets = np.array(targets)
        self.ids = np.array(ids)
        self.batch_size = batch_size
        self.ids_batches = []

    def make_batches(self, ids, targets):
        unique, counts = np.unique(targets, return_counts=True)
        if (counts[0] < 2 or counts[1] <2) or len(ids) <= self.batch_size:
            self.ids_batches.append(ids)
        else:
            s = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state=config.SEED)
            left_batch_indices, right_batch_indices = [x for x in s.split(ids, targets)][0]
            self.make_batches(ids[left_batch_indices], targets[left_batch_indices])
            self.make_batches(ids[right_batch_indices], targets[right_batch_indices])

    def gen_sample_array(self):
        self.ids_batches = [] #to make sure batches  are created only one time
        self.make_batches(self.ids, self.targets)

        # #sanity check
        # test_ids = np.array([])
        # for i, ids_batch in enumerate(self.ids_batches):
        #     ids_batch_indices = np.searchsorted(self.ids, ids_batch)
        #     print(f'{i}, {np.unique(np.array(self.targets[ids_batch_indices]), return_counts=True)}, {np.sum(self.ids[ids_batch_indices])},{self.ids[ids_batch_indices][0:4]}')
        #     test_ids = np.concatenate([test_ids, self.ids[ids_batch_indices]])
        # print(np.all(np.sort(test_ids) == np.sort(self.ids)))

        for ids_batch in self.ids_batches:
            ids_batch_indices = np.searchsorted(self.ids, ids_batch)
            yield list(ids_batch_indices)

    def __iter__(self):
        return iter(self.gen_sample_array())
        # return iter(self.batches_ids)

    def __len__(self):
        return len(self.targets)


# ids = np.arange(32*7)
# targets = (np.random.rand(32*7) > 0.5).astype(int)
# a = StratifiedSampler(ids, targets, 32)
# print(next(iter(a)))
# for x in iter(a):
#     print(x)


# i = -1
# mini_batches = []
# def make_batches(ids, targets):
#     unique, counts = np.unique(targets, return_counts=True)

#     if (counts[0] < 2 or counts[1] <2) or len(ids) <= batch_size:
#         mini_batches.append(ids)
#     else:
#         s = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5)
#         left_batch_ids, right_batch_ids = [x for x in s.split(ids, targets)][0]
#         make_batches(left_batch_ids, targets[left_batch_ids])
#         make_batches(right_batch_ids, targets[right_batch_ids])
# ids = np.arange(32*7)
# targets = (np.random.rand(32*7) > 0.5).astype(int)
# # a = iter(StratifiedSampler(ids, target, 2))
# make_batches(ids, targets)


# def sample_generator(ids, targets):
#     while len(ids)
#     # if len(ids) <= 2:
#     #     yield np.array(ids)
#     # else:
#     #     s = StratifiedShuffleSplit(n_splits = 1, test_size=0.5)
#     #     left_batch_ids, right_batch_ids = [x for x in s.split(ids, targets)][0]
#     #     return np.array(left_batch_ids)
#     #     # self.gen_sample_array(ids[left_batch_ids], targets[left_batch_ids])
#     #     # self.gen_sample_array(ids[right_batch_ids], targets[right_batch_ids])
#     #     # yield 1
#     #     # if len(left_batch_ids) <= self.batch_size:
#     #     #     pass
#     #     #     # yield np.array(left_batch_ids)
#     #     # if len(right_batch_ids) <= self.batch_size:
#     #     #     pass
#     #     #     # yield np.array(right_batch_ids)
#     # for i in range(len(ids)):
#     i = 0
#     while i < len(ids):
        
#         yield ids[i], targets[i]
#         i = i+1

# ids = np.arange(6)
# targets = np.array([0,1,0,1,0,1])

# sample_generator_object = sample_generator(ids, targets)
# print(next(sample_generator_object))
# print(next(sample_generator_object))
# print(next(sample_generator_object))
# print(next(sample_generator_object))
# # print((sample_generator_object))
# # print((sample_generator_object))
# # print((sample_generator_object))
# # for a, b in sample_generator_object:
# #     print(a, b)


# # a = iter(StratifiedSampler(ids, target, 2))
# # print(len(a))
# # a = (StratifiedSampler(np.arange(6), np.array([0, 1, 1,1,0,0]), 2).gen_sample_array(np.arange(6), np.array([0, 1, 1,1,0,0]))

# # iter -- return a fresh batch