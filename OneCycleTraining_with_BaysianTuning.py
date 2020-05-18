import sys
import gc
from numba import cuda
import pickle
import tensorflow as tf
from SqueezeNet import squeeze_net
from sklearn.preprocessing import OneHotEncoder
#from clr import LRFinder
from clr import OneCycleLR
import gc
from numba import cuda

# Reference: https://github.com/titu1994/keras-one-cycle


def OneCycleTrain(squeeze_scale_exp, small_filter_rate, max_lr_exp, max_momentum, num_epoch):
    # def LRSearch(squeeze_scale_exp, small_filter_rate):
    scale = 10**squeeze_scale_exp  # float(sys.argv[1])  # float(sys.argv[1])
    small_filter_rate = small_filter_rate  # float(sys.argv[2])  # float(sys.argv[2])
    max_lr = 10**max_lr_exp  # float(sys.argv[3])
    max_momentum = max_momentum
    num_epoch = num_epoch  # int(sys.argv[4])
    batch_size = 2048
    #minimum_lr = 1e-8
    #maximum_lr = 1e8
    f = open('data.p', 'rb')
    (X_train, y_train), (X_test,
                         y_test) = pickle.load(f)  # cifar100.load_data()
    num_samples = len(X_train)

    op = tf.keras.optimizers.SGD(momentum=max_momentum - 0.05)  # , decay=1e-6, momentum=0.9)
    model = squeeze_net(small_filter_rate=small_filter_rate,
                        squeeze_scale=scale,
                        verbose=False)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    lr_manager = OneCycleLR(max_lr, maximum_momentum=max_momentum, minimum_momentum=max_momentum - 0.1)
    model.compile(loss=loss, optimizer=op, metrics=['acc'])
    oh = OneHotEncoder(sparse=False)
    oh.fit(y_train)
    history = model.fit(X_train / 255.,
                        oh.transform(y_train),
                        epochs=num_epoch,
                        batch_size=batch_size,
                        validation_data=(X_test / 255.,
                                         oh.transform(y_test)),
                        callbacks=[lr_manager], shuffle=True)
    final_val_acc = history.history['val_acc'][-1]
    del model
    gc.collect()
    tf.keras.backend.clear_session()
    cuda.select_device(0)
    cuda.close()
    return final_val_acc

    # TODO:
    # 1. modulize the training process : input arguments and output val acc
    # 2. incorporating GPU cleaning scheme
    # 3. incorporating the baysian optimization scheme for tuning num_epoch and max_lr
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
pbounds = {
    'squeeze_scale_exp': (1.5, 1.5),  # 2
    'small_filter_rate': (0.5, 0.5),  # 10
    'max_lr_exp': (-6, -1),            # 6
    'max_momentum': (0.8, 0.99),
    'num_epoch': (3, 100)
}
optimizer = BayesianOptimization(
    f=OneCycleTrain,
    pbounds=pbounds,
    random_state=1,
)
try:
    load_logs(optimizer, logs=["./one_cycle_baysian_logs.json"])
except:
    print('no one_cycle_baysian_logs.json')
logger = JSONLogger(path="./one_cycle_baysian_logs.json")

optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

# run the optimization
optimizer.maximize(
    init_points=300,  # determine according to the boundary of each parameter
    n_iter=8,      # also determine by the boundary of each parameter
)

# access history and result
for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))

print("Final Max:", optimizer.max)
