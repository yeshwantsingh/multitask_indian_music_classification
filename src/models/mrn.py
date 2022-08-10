# Deep relationship network

import tensorflow as tf


def task_layer(i, inputs, num_task):
    z = tf.keras.layers.Dense(128, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(inputs)
    z = tf.keras.layers.Dense(64, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(z)
    z = tf.keras.layers.Dense(32, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(z)
    z = tf.keras.layers.Dense(num_task, activation='softmax', name='output'+str(i+1),
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(z)
    return z


def get_mrn_model(input_shape, outputs):
    inputs = tf.keras.Input(shape=input_shape)

    z = tf.expand_dims(inputs, -1)

    for filter_size in [64, 128, 256, 512, 512]:
        z = tf.keras.layers.Conv2D(filter_size, 3, padding="same", activation="relu")(z)
        z = tf.keras.layers.MaxPool2D(2, 2)(z)

    z = tf.keras.layers.Flatten()(z)

    tasks = [task_layer(i, z, outputs[i]) for i in range(len(outputs))]

    return tf.keras.models.Model(inputs=inputs, outputs=tasks)
