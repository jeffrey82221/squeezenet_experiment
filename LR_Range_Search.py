import sys
import gc
from numba import cuda
import pickle
import tensorflow as tf
from SqueezeNet import squeeze_net
from sklearn.preprocessing import OneHotEncoder
from clr import LRFinder
# Reference: https://github.com/titu1994/keras-one-cycle


# def LRSearch(squeeze_scale_exp, small_filter_rate):
scale = 10**float(sys.argv[1])  # float(sys.argv[1])
small_filter_rate = float(sys.argv[2])  # float(sys.argv[2])
batch_size = 2048
minimum_lr = 1e-8
maximum_lr = 1e8
f = open('data.p', 'rb')
(X_train, y_train), (X_test,
                     y_test) = pickle.load(f)  # cifar100.load_data()
num_samples = len(X_train)

lr_callback = LRFinder(
    num_samples,
    batch_size,
    minimum_lr,
    maximum_lr,
    # validation_data=(X_val, Y_val),
    lr_scale='exp',
    save_dir='lr_log')
op = tf.keras.optimizers.SGD(momentum=0.95)  # , decay=1e-6, momentum=0.9)
model = squeeze_net(small_filter_rate=small_filter_rate,
                    squeeze_scale=scale,
                    verbose=False)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(loss=loss, optimizer=op, metrics=['acc'])
oh = OneHotEncoder(sparse=False)
oh.fit(y_train)
history = model.fit(X_train / 255.,
                    oh.transform(y_train),
                    epochs=1,
                    batch_size=batch_size,
                    callbacks=[lr_callback])

# Test:
#from LR_Range_Search import LRSearch
import numpy as np
for i in np.load('lr_log/lrs.npy'):
  print(i)
for i in np.load('lr_log/losses.npy'):
  print(i)
