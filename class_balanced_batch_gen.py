import random
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
'''
from class_balanced_batch_gen import ClassBalancedBatchGenerator
from tensorflow.keras.datasets import cifar100
import pickle
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
gen = ClassBalancedBatchGenerator(X_train, y_train, 1000)
gen.get_rearranged_batch_indices(X_train, y_train)
'''


class ClassBalancedBatchGenerator(Sequence):
  def __init__(self, X_train, y_train, batch_size):
    self.X_train = X_train
    self.y_train = y_train
    self.batch_size = batch_size
    assert self.batch_size % 100 == 0
    self.generator = self.train_generator()

  def __len__(self):
    return int(self.X_train.shape[0] / self.batch_size)

  @property
  def shape(self):
    return self.X_train.shape

  def __getitem__(self, item):
    return self.__next__()

  def __next__(self):
    return next(self.generator)

  def get_rearranged_batch_indices(self):
    num_instance_for_each_class = int(len(self.X_train) / 100)
    indices_for_each_class = [
        np.where(self.y_train[:, c] == 1)[0] for c in range(100)
    ]
    for c in range(100):
      random.shuffle(indices_for_each_class[c])
    return np.vstack(indices_for_each_class).T.flatten()

  def batch_indice_generator(self):
    train_data_num = len(self.X_train)
    batch_count = int(train_data_num / self.batch_size)
    while True:
      new_indices = self.get_rearranged_batch_indices()
      for current_batch_id in range(batch_count):
        yield new_indices[current_batch_id *
                          self.batch_size:(current_batch_id + 1) *
                          self.batch_size]

  def train_generator(self):
    batch_indices_gen = self.batch_indice_generator()
    while True:
      batch_indices = next(batch_indices_gen)
      yield self.X_train[batch_indices], self.y_train[batch_indices]
