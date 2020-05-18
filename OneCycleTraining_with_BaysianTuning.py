import sys
import gc
from numba import cuda
import pickle
import tensorflow as tf
from SqueezeNet import squeeze_net
from sklearn.preprocessing import OneHotEncoder
#from clr import LRFinder
from clr import OneCycleLR

# Reference: https://github.com/titu1994/keras-one-cycle


# def LRSearch(squeeze_scale_exp, small_filter_rate):
scale = 10**float(sys.argv[1])  # float(sys.argv[1])
small_filter_rate = float(sys.argv[2])  # float(sys.argv[2])
max_lr = float(sys.argv[3])
num_epoch = int(sys.argv[4])
batch_size = 2048
minimum_lr = 1e-8
maximum_lr = 1e8
f = open('data.p', 'rb')
(X_train, y_train), (X_test,
                     y_test) = pickle.load(f)  # cifar100.load_data()
num_samples = len(X_train)


op = tf.keras.optimizers.Nadam()  # .SGD(momentum=0.95)  # , decay=1e-6, momentum=0.9)
model = squeeze_net(small_filter_rate=small_filter_rate,
                    squeeze_scale=scale,
                    verbose=False)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
lr_manager = OneCycleLR(max_lr, maximum_momentum=None, minimum_momentum=None)
model.compile(loss=loss, optimizer=op, metrics=['acc'])
oh = OneHotEncoder(sparse=False)
oh.fit(y_train)
history = model.fit(X_train / 255.,
                    oh.transform(y_train),
                    epochs=num_epoch,
                    batch_size=batch_size,
                    callbacks=[lr_manager])
# TODO:
# 1. modulize the training process : input arguments and output val acc
# 2. incorporating GPU cleaning scheme
# 3. incorporating the baysian optimization scheme for tuning num_epoch and max_lr
