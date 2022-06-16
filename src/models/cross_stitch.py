import tensorflow as tf
import numpy as np


def apply_cross_stitch(input1, input2):
    input1_reshaped = tf.keras.layers.Flatten()(input1)
    input2_reshaped = tf.keras.layers.Flatten()(input2)
    input = tf.concat((input1_reshaped, input2_reshaped), axis=1)

    # initialize with identity matrix
    cross_stitch = tf.Variable(initial_value=tf.keras.initializers.Identity()(shape=(input.shape[1], input.shape[1])),
                               dtype='float32',
                               name="cross_stitch",
                               )
    output = tf.matmul(input, cross_stitch)

    print(input1.shape, input2.shape)
    # need to call .value to convert Dimension objects to normal value
    input1_shape = list(-1 if s is None else s for s in input1.shape)
    input2_shape = list(-1 if s is None else s for s in input2.shape)
    output1 = tf.reshape(output[:, :input1_reshaped.shape[1]], shape=input1_shape)
    output2 = tf.reshape(output[:, input1_reshaped.shape[1]:], shape=input2_shape)
    return output1, output2


def get_cross_stitch_network(n_output_1, n_output_2, dropout_prob, cross_stitch_enabled):
    # (?, 28, 28, 1) -> (?, 28, 28, 32)
    inputs = tf.keras.Input(shape=(28, 28, 1))
    conv1_1 = tf.keras.layers.Conv2D(32, 3, name='conv1_1', activation='relu')(inputs)
    conv1_2 = tf.keras.layers.Conv2D(32, 3, name='conv1_2', activation='relu')(inputs)

    # (?, 28, 28, 32) -> (?, 14, 14, 32)
    pool1_1 = tf.keras.layers.MaxPool2D(2, 2, name="pool_1_1")(conv1_1)
    pool1_2 = tf.keras.layers.MaxPool2D(2, 2, name="pool_1_2")(conv1_2)

    if cross_stitch_enabled:
        stitch_pool1_1, stitch_pool1_2 = apply_cross_stitch(pool1_1, pool1_2)
    else:
        stitch_pool1_1, stitch_pool1_2 = pool1_1, pool1_2

    # (?, 14, 14, 32) -> (?, 14, 14, 64)
    conv2_1 = tf.keras.layers.Conv2D(64, 3, name='conv2_1', activation='relu')(stitch_pool1_1)
    conv2_2 = tf.keras.layers.Conv2D(64, 3, name='conv2_2', activation='relu')(stitch_pool1_2)

    # (?, 14, 14, 64) -> (?, 7, 7, 64)
    pool2_1 = tf.keras.layers.MaxPool2D(2, 2, name="pool_2_1")(conv2_1)
    pool2_2 = tf.keras.layers.MaxPool2D(2, 2, name="pool_2_2")(conv2_2)

    if cross_stitch_enabled:
        stitch_pool2_1, stitch_pool2_2 = apply_cross_stitch(pool2_1, pool2_2)
    else:
        stitch_pool2_1, stitch_pool2_2 = pool2_1, pool2_2

    # (?, 7, 7, 64) -> (?, 3136) -> -> (?, 1024)
    flatten_1 = tf.keras.layers.Flatten()(stitch_pool2_1)
    fc_3_1 = tf.keras.layers.Dense(1024)(flatten_1)

    flatten_2 = tf.keras.layers.Flatten()(stitch_pool2_2)
    fc_3_2 = tf.keras.layers.Dense(1024)(flatten_2)

    if cross_stitch_enabled:
        stitch_fc_3_1, stitch_fc_3_2 = apply_cross_stitch(fc_3_1, fc_3_2)
    else:
        stitch_fc_3_1, stitch_fc_3_2 = fc_3_1, fc_3_2

    dropout_1 = tf.keras.layers.Dropout(dropout_prob)(stitch_fc_3_1)
    dropout_2 = tf.keras.layers.Dropout(dropout_prob)(stitch_fc_3_2)

    output_1 = tf.keras.layers.Dense(n_output_1, name='output_1')(dropout_1)
    output_2 = tf.keras.layers.Dense(n_output_2, name='output_2')(dropout_2)

    model = tf.keras.Model(inputs, [output_1, output_2])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss={'output_1': tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        'output_2': tf.keras.losses.CategoricalCrossentropy(from_logits=True)},
                  metrics=['accuracy'])

    return model
