import sys
import gc
from numba import cuda
import pickle
import tensorflow as tf
from SqueezeNet import squeeze_net
from sklearn.preprocessing import OneHotEncoder


def SqueezeNetFunction(squeeze_scale_exp, small_filter_rate, lr_exp):
  batch_size = 2048
  f = open('data.p', 'rb')
  (X_train, y_train), (X_test, y_test) = pickle.load(f)  # cifar100.load_data()
  oh = OneHotEncoder(sparse=False)
  oh.fit(y_train)
  acc_dict = dict()
  # repeat = 30
  # sf_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  scale = 10**squeeze_scale_exp  # float(sys.argv[1])
  small_filter_rate = small_filter_rate  # float(sys.argv[2])
  lr = 10**lr_exp  # float(sys.argv[3])
  op = tf.keras.optimizers.Nadam(
      lr=lr)  # , decay=1e-6, momentum=0.9)
  model = squeeze_net(small_filter_rate=small_filter_rate, squeeze_scale=scale, verbose=False)
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  model.compile(loss=loss,
                optimizer=op,
                metrics=['acc'])
  stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                                patience=5,
                                                verbose=1)
  history = model.fit(X_train / 255.,
                      oh.transform(y_train),
                      epochs=10000,
                      batch_size=batch_size,
                      validation_data=(X_test / 255.,
                                       oh.transform(y_test)),
                      verbose=2,
                      callbacks=[stop_early], shuffle=True)
  # make sure the input of each pixel is between 0 and 1
  # get final val acc
  final_val_acc = history.history['val_acc'][-1]
  #print("ANS:", scale, small_filter_rate, final_val_acc, model.count_params())
  del model
  gc.collect()
  tf.keras.backend.clear_session()
  cuda.select_device(0)
  cuda.close()
  return final_val_acc
