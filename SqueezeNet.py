import tensorflow as tf


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
#y = tf.keras.layers.Activation('sigmoid', name='loss')(y)
  model = tf.keras.Model(x, y, name='squeezenet_v1.1')
  if verbose:
    model.summary()
  return model
