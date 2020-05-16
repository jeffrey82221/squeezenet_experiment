import tensorflow as tf
import sys


def fire(x, squeeze, expand_s, expand_l, name):
  y = tf.keras.layers.Conv2D(filters=squeeze,
                             kernel_size=1,
                             padding='valid',
                             activation='relu',
                             name=name + "_1X1_1")(x)
  # reducing the channel size to the next two layers
  if expand_s == 0:
    y = tf.keras.layers.Conv2D(filters=expand_l,
                               kernel_size=3,
                               padding='same',
                               activation='relu',
                               name=name + "_3X3_2")(y)
    # reducing parameter size by replacing some of the 3X3 conv to 1X1 conv
    return y
  elif expand_l == 0:
    y = tf.keras.layers.Conv2D(filters=expand_s,
                               kernel_size=1,
                               padding='valid',
                               activation='relu',
                               name=name + "_1X1_2")(y)
    # reducing parameter size by replacing some of the 3X3 conv to 1X1 conv
    return y
  else:
    y1 = tf.keras.layers.Conv2D(filters=expand_s,
                                kernel_size=1,
                                padding='valid',
                                activation='relu',
                                name=name + "_1X1_2")(y)
    y3 = tf.keras.layers.Conv2D(filters=expand_l,
                                kernel_size=3,
                                padding='same',
                                activation='relu',
                                name=name + "_3X3_2")(y)
    # reducing parameter size by replacing some of the 3X3 conv to 1X1 conv
    return tf.keras.layers.concatenate([y1, y3])


# this is to make it behave similarly to other Keras layers
def fire_module(squeeze, expand_s, expand_l, squeeze_scale, name='fire'):
  return lambda x: fire(x, int(squeeze * squeeze_scale), expand_s, expand_l, name)


def squeeze_net(IMAGE_SIZE=[32, 32],
                CLASS_NUM=100,
                small_filter_rate=0.5,
                squeeze_scale=1.0,
                verbose=True):
  # Network according to Table I in the original paper:
  #IMAGE_SIZE = [32, 32]#list(train_pipe.target_size)#
  # CLASS_NUM = 100#len(train_pipe.class_indices)
  x = tf.keras.layers.Input(shape=[*IMAGE_SIZE, 3], name='input_image')
  # input is 192x192 pixels RGB (3 channels)
  y = tf.keras.layers.Conv2D(kernel_size=3,
                             filters=64,
                             strides=(2, 2),
                             padding='valid',
                             activation='relu',
                             name='conv1')(x)
  y = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2,
                                   name='maxpool1')(y)

  def small_filter_count(K):
    total_filter_count = K * 2
    return int(small_filter_rate * total_filter_count)

  def large_filter_count(K):
    total_filter_count = K * 2
    return total_filter_count - int(small_filter_rate * total_filter_count)

  y = fire_module(16,
                  small_filter_count(64),
                  large_filter_count(64),
                  squeeze_scale,
                  name='fire2')(y)
  y = fire_module(16,
                  small_filter_count(64),
                  large_filter_count(64),
                  squeeze_scale,
                  name='fire3')(y)

  y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
  y = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2,
                                   name='maxpool3')(y)
  y = fire_module(32,
                  small_filter_count(128),
                  large_filter_count(128),
                  squeeze_scale,
                  name='fire4')(y)
  y = fire_module(32,
                  small_filter_count(128),
                  large_filter_count(128),
                  squeeze_scale,
                  name='fire5')(y)
  y = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(y)
  y = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2,
                                   name='maxpool5')(y)
  y = fire_module(48,
                  small_filter_count(192),
                  large_filter_count(192),
                  squeeze_scale,
                  name='fire6')(y)
  y = fire_module(48,
                  small_filter_count(192),
                  large_filter_count(192),
                  squeeze_scale,
                  name='fire7')(y)
  y = fire_module(64,
                  small_filter_count(256),
                  large_filter_count(256),
                  squeeze_scale,
                  name='fire8')(y)
  y = fire_module(64,
                  small_filter_count(256),
                  large_filter_count(256),
                  squeeze_scale,
                  name='fire9')(y)
  #
  y = tf.keras.layers.Dropout(0.5, name='drop9')(y)
  y = tf.keras.layers.Conv2D(kernel_size=1,
                             filters=CLASS_NUM,
                             strides=1,
                             padding='same',
                             activation='relu',
                             name='conv10')(y)
  y = tf.keras.layers.GlobalAveragePooling2D(name='avgpool10')(y)
  y = tf.keras.layers.Activation('softmax', name='loss')(y)
  model = tf.keras.Model(x, y, name='squeezenet_v1.1')
  if verbose:
    model.summary()
  return model


#from tensorflow.keras.datasets import cifar100
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import pickle
f = open('data.p', 'rb')
(X_train, y_train), (X_test, y_test) = pickle.load(f)  # cifar100.load_data()
oh = OneHotEncoder(sparse=False)
oh.fit(y_train)
acc_dict = dict()
#repeat = 30
#sf_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
scale = float(sys.argv[1])
small_filter_rate = float(sys.argv[2])

lr = float(sys.argv[3])
op = tf.keras.optimizers.Nadam(
    lr=lr)  # , decay=1e-6, momentum=0.9)
model = squeeze_net(small_filter_rate=small_filter_rate, squeeze_scale=scale, verbose=False)
model.compile(loss='categorical_crossentropy',
              optimizer=op,
              metrics=['acc'])
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                              patience=5,
                                              verbose=1)
history = model.fit(X_train / 255.,
                    oh.transform(y_train),
                    epochs=10000,
                    batch_size=2048,
                    validation_data=(X_test / 255.,
                                     oh.transform(y_test)),
                    verbose=2,
                    callbacks=[stop_early], shuffle=True)
# make sure the input of each pixel is between 0 and 1
# get final val acc
final_val_acc = history.history['val_acc'][-1]
print("ANS:", scale, small_filter_rate, final_val_acc, model.count_params())
