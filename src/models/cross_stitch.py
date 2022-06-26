import tensorflow as tf


def apply_cross_stitch(input1, input2):
    input1_reshaped = tf.keras.layers.Flatten()(input1)
    input2_reshaped = tf.keras.layers.Flatten()(input2)
    input = tf.concat((input1_reshaped, input2_reshaped), axis=1)

    # initialize with identity matrix
    cross_stitch = tf.Variable(initial_value=tf.keras.initializers.Identity()(shape=(input.shape[1], input.shape[1])))
    output = tf.matmul(input, cross_stitch)

    # need to call .value to convert Dimension objects to normal value
    input1_shape = list(-1 if s is None else s for s in input1.shape)
    input2_shape = list(-1 if s is None else s for s in input2.shape)
    output1 = tf.reshape(output[:, :input1_reshaped.shape[1]], shape=input1_shape)
    output2 = tf.reshape(output[:, input1_reshaped.shape[1]:], shape=input2_shape)
    return output1, output2


def get_cross_stitch_network(input_shape, outputs):
    dropout_prob = 0.2
    cross_stitch_enabled = True

    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    x = tf.keras.layers.Resizing(28, 28)(x)

    conv1_1 = tf.keras.layers.Conv2D(32, 3, name='conv1_1', activation='relu')(x)
    conv1_2 = tf.keras.layers.Conv2D(32, 3, name='conv1_2', activation='relu')(x)

    pool1_1 = tf.keras.layers.MaxPool2D(2, 2, name="pool_1_1")(conv1_1)
    pool1_2 = tf.keras.layers.MaxPool2D(2, 2, name="pool_1_2")(conv1_2)

    if cross_stitch_enabled:
        stitch_pool1_1, stitch_pool1_2 = apply_cross_stitch(pool1_1, pool1_2)
    else:
        stitch_pool1_1, stitch_pool1_2 = pool1_1, pool1_2

    conv2_1 = tf.keras.layers.Conv2D(64, 3, name='conv2_1', activation='relu')(stitch_pool1_1)
    conv2_2 = tf.keras.layers.Conv2D(64, 3, name='conv2_2', activation='relu')(stitch_pool1_2)

    pool2_1 = tf.keras.layers.MaxPool2D(2, 2, name="pool_2_1")(conv2_1)
    pool2_2 = tf.keras.layers.MaxPool2D(2, 2, name="pool_2_2")(conv2_2)

    if cross_stitch_enabled:
        stitch_pool2_1, stitch_pool2_2 = apply_cross_stitch(pool2_1, pool2_2)
    else:
        stitch_pool2_1, stitch_pool2_2 = pool2_1, pool2_2

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

    output_1 = tf.keras.layers.Dense(17, name='output1')(dropout_1)
    output_2 = tf.keras.layers.Dense(68, name='output2')(dropout_2)

    return tf.keras.Model(inputs, [output_1, output_2])


def apply_cross_stitch_scaled(inputs):
    input_reshaped = [tf.keras.layers.Flatten()(input) for input in inputs]
    input_concat = tf.concat(input_reshaped, axis=1)

    # initialize with identity matrix
    cross_stitch = tf.Variable(initial_value=tf.keras.initializers.Identity()(shape=(input_concat.shape[1], input_concat.shape[1])),
                               dtype='float32',
                               )
    output = tf.matmul(input_concat, cross_stitch)

    # need to call .value to convert Dimension objects to normal value
    input_shape = [list(-1 if s is None else s for s in input.shape) for input in inputs]
    output1 = tf.reshape(output[:, :input1_reshaped.shape[1]], shape=input1_shape)
    output2 = tf.reshape(output[:, input1_reshaped.shape[1]:], shape=input2_shape)
    return output1, output2


def get_cross_stitch_network_scaled(input_shape, outputs):
    dropout_prob = 0.2
    cross_stitch_enabled = True

    inputs = tf.keras.layers.Input(shape=input_shape)

    conv1 = [tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs) for _ in range(len(outputs))]
    pool1 = [tf.keras.layers.MaxPool2D(2)(conv) for conv in conv1]

    if cross_stitch_enabled:
        stitch_pool1 = apply_cross_stitch(pool1)
    else:
        stitch_pool1 = pool1

    conv2 = [tf.keras.layers.Conv2D(64, 3, activation='relu')(stitch_pool) for stitch_pool in stitch_pool1]

    pool2 = [tf.keras.layers.MaxPool2D(2)(conv) for conv in conv2]

    if cross_stitch_enabled:
        stitch_pool2 = apply_cross_stitch(pool2)
    else:
        stitch_pool2 = pool2

    flatten = [tf.keras.layers.Flatten()(stitch_pool) for stitch_pool in stitch_pool2]
    fc_3 = [tf.keras.layers.Dense(1024)(_flatten) for _flatten in flatten]

    if cross_stitch_enabled:
        stitch_fc_3 = apply_cross_stitch(fc_3)
    else:
        stitch_fc_3 = fc_3

    dropout = [tf.keras.layers.Dropout(dropout_prob)(stitch_fc) for stitch_fc in stitch_fc_3]

    outputs = [tf.keras.layers.Dense(outputs[i])(dropout[i]) for i in range(len(outputs))]

    return tf.keras.Model(inputs, outputs)


def get_cross_stitch_network_basic(input_shape, outputs):
    inputs = tf.keras.layers.Input(shape=input_shape)

    def conv_block(inputs, kernels):
        x = tf.keras.layers.Conv1D(kernels, 3, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(2)(x)
        return x

    task1 = conv_block(inputs, 32)
    task2 = conv_block(inputs, 32)

    concat = tf.keras.layers.Concatenate()([task1, task2])

    task1 = conv_block(concat, 64)
    task2 = conv_block(concat, 64)

    concat = tf.keras.layers.Concatenate()([task1, task2])

    task1 = tf.keras.layers.Flatten()(concat)
    task2 = tf.keras.layers.Flatten()(concat)

    task1 = tf.keras.layers.Dense(1024)(task1)
    task2 = tf.keras.layers.Dense(1024)(task2)

    task1 = tf.keras.layers.Dropout(0.1)(task1)
    task2 = tf.keras.layers.Dropout(0.1)(task2)

    output1 = tf.keras.layers.Dense(num_langs, name="output1")(task1)
    output2 = tf.keras.layers.Dense(num_artists, name="output2")(task2)

    return tf.keras.models.Model(inputs=inputs, outputs=[output1, output2])

