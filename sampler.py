import torch as th
import math
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class Sampler():
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, ids, targets, batch_size):
        """
        Arguments
        ---------
        targets : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(len(targets) / batch_size)
        self.targets = np.array(targets)
        self.ids = np.array(ids)
        self.batch_size = batch_size

    def gen_sample_array(self, ids, targets):
        if len(ids) <= self.batch_size:
            # yield np.array(ids)
            pass
        else:
            s = StratifiedShuffleSplit(n_splits = 1, test_size=0.5)
            left_batch_ids, right_batch_ids = [x for x in s.split(ids, targets)][0]
            return np.array(left_batch_ids)
            # self.gen_sample_array(ids[left_batch_ids], targets[left_batch_ids])
            # self.gen_sample_array(ids[right_batch_ids], targets[right_batch_ids])
            # yield 1
            # if len(left_batch_ids) <= self.batch_size:
            #     pass
            #     # yield np.array(left_batch_ids)
            # if len(right_batch_ids) <= self.batch_size:
            #     pass
            #     # yield np.array(right_batch_ids)

    def __iter__(self):
        return self.gen_sample_array(self.ids, self.targets)

    def __len__(self):
        return len(self.targets)

i = -1

def make_b
def sample_generator(ids, targets):
    while len(ids)
    # if len(ids) <= 2:
    #     yield np.array(ids)
    # else:
    #     s = StratifiedShuffleSplit(n_splits = 1, test_size=0.5)
    #     left_batch_ids, right_batch_ids = [x for x in s.split(ids, targets)][0]
    #     return np.array(left_batch_ids)
    #     # self.gen_sample_array(ids[left_batch_ids], targets[left_batch_ids])
    #     # self.gen_sample_array(ids[right_batch_ids], targets[right_batch_ids])
    #     # yield 1
    #     # if len(left_batch_ids) <= self.batch_size:
    #     #     pass
    #     #     # yield np.array(left_batch_ids)
    #     # if len(right_batch_ids) <= self.batch_size:
    #     #     pass
    #     #     # yield np.array(right_batch_ids)
    # for i in range(len(ids)):
    i = 0
    while i < len(ids):
        
        yield ids[i], targets[i]
        i = i+1

ids = np.arange(6)
targets = np.array([0,1,0,1,0,1])

sample_generator_object = sample_generator(ids, targets)
print(next(sample_generator_object))
print(next(sample_generator_object))
print(next(sample_generator_object))
print(next(sample_generator_object))
# print((sample_generator_object))
# print((sample_generator_object))
# print((sample_generator_object))
# for a, b in sample_generator_object:
#     print(a, b)


# a = StratifiedSampler(ids, target, 2).gen_sample_array(ids, target)
# print(len(a))
# a = (StratifiedSampler(np.arange(6), np.array([0, 1, 1,1,0,0]), 2).gen_sample_array(np.arange(6), np.array([0, 1, 1,1,0,0]))

# iter -- return a fresh batch