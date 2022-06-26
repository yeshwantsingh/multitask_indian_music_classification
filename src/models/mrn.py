# Deep relationship network

import tensorflow as tf


def task_layer(inputs, num_tasks):
    z = tf.keras.layers.Dense(num_tasks, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(inputs)
    z = tf.keras.layers.Dense(num_tasks, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(z)
    z = tf.keras.layers.Dense(num_tasks, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(z)
    z = tf.keras.layers.Dense(num_tasks, activation='softmax',
                              kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                              bias_initializer=tf.keras.initializers.Zeros())(z)
    return z


def get_mrn_model(input_shape, outputs):
    outputs = [17, 16, 1, 1]
    inputs = tf.keras.Input(shape=input_shape)
    z = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)

    for filter_size in [64, 128, 256, 512, 512]:
        z = tf.keras.layers.Conv2D(filter_size, 3, padding="same", activation="relu")(z)
        z = tf.keras.layers.MaxPool2D(2, 2)(z)

    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dense(17, activation='softmax', dtype='float32')(z)

    task1 = task_layer(z, outputs[0])
    task2 = task_layer(z, outputs[1])
    task3 = task_layer(z, outputs[2])
    task4 = task_layer(z, outputs[3])

    output1 = tf.keras.layers.Dense(outputs[0], name="output1")(task1)
    output2 = tf.keras.layers.Dense(outputs[1], name="output2")(task2)
    output3 = tf.keras.layers.Dense(outputs[2], name="output3")(task3)
    output4 = tf.keras.layers.Dense(outputs[3], name="output4")(task4)

    return tf.keras.models.Model(inputs=inputs, outputs=[output1, output2, output3, output4])
