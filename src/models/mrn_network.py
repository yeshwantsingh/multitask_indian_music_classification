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


def get_model(shape):
    inputs = tf.keras.Input(shape=shape)
    z = tf.keras.layers.Reshape((shape[0], shape[1], 1))(inputs)

    for filter_size in [64, 128, 256, 512, 512]:
        z = tf.keras.layers.Conv2D(filter_size, 3, padding="same", activation="relu")(z)
        z = tf.keras.layers.MaxPool2D(2, 2)(z)

    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dense(17, activation='softmax', dtype='float32')(z)

    output1 = task_layer(z, 17)
    output2 = task_layer(z, 5)

    return tf.keras.Model(inputs=inputs, outputs=[output1, output2])


if __name__ == '__main__':
    model = get_model((224, 224))
    model.summary()
    tf.keras.utils.plot_model(model, 'model.png', dpi=600)
