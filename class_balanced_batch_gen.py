import random
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence


class ClassBalancedBatchGenerator(Sequence):
  def __init__(self, X_train, y_train, batch_size):
    self.X_train = X_train
    self.y_train = y_train
    self.batch_size = batch_size
    self.generator = self.train_generator(batch_size=batch_size)

  def __len__(self):
    return int(self.X_train.shape[0] / self.batch_size)

  @property
  def shape(self):
    return self.X_train.shape

  def next(self):
    return self.__next__()

  def get_rearranged_batch_indices(self, X_train, y_train):
    num_instance_for_each_class = int(len(X_train) / 100)
    indices_for_each_class = [
        np.where(y_train == c)[0] for c in range(100)
    ]
    for c in range(100):
      random.shuffle(indices_for_each_class[c])
    new_indices = []
    for i in range(num_instance_for_each_class):
      for c in range(100):
        new_indices.append(indices_for_each_class[c][i])
    return new_indices
    # X_train = X_train[new_indices]
    # y_train = y_train[new_indices]

  def batch_indice_generator(self, X_train, y_train, batch_size=1000):
    assert batch_size % 100 == 0
    train_data_num = len(X_train)
    batch_count = int(train_data_num / batch_size)
    while True:
      new_indices = self.get_rearranged_batch_indices(X_train, y_train)
      for current_batch_id in range(batch_count):
        yield new_indices[current_batch_id *
                          batch_size:(current_batch_id + 1) *
                          batch_size]

  def train_generator(self, batch_size=1000):
    batch_indices_gen = self.batch_indice_generator(
        self.X_train, self.y_train, batch_size=self.batch_size)
    while True:
      batch_indices = next(batch_indices_gen)
      yield self.X_train[batch_indices], self.y_train[batch_indices]

  def __next__(self):
    return next(self.generator)
