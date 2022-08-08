import tensorflow as tf

tf.keras.backend.set_floatx('float16')


class CrossStitch(tf.keras.layers.Layer):
    def __init__(self, shape):
        super(CrossStitch, self).__init__()
        self.shape = shape
        self.idx_mat = tf.Variable(initial_value=tf.keras.initializers.Identity()(shape=(shape, shape)), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.idx_mat)


def apply_cross_stitch(inputs):
    input_reshaped = [tf.keras.layers.Flatten()(input) for input in inputs]
    input_concat = tf.concat(input_reshaped, axis=1)

    # # initialize with identity matrix
    cross_stitch = tf.Variable(
        initial_value=tf.keras.initializers.Identity()(shape=(input_concat.shape[1], input_concat.shape[1])))
    output = tf.matmul(input_concat, cross_stitch)

    # output = CrossStitch(input_concat.shape[1])(input_concat)

    # need to call .value to convert Dimension objects to normal value
    input_shape = [list(-1 if s is None else s for s in input.shape) for input in inputs]
    outputs = [
        tf.reshape(output[:, i * input_reshaped[0].shape[1]:(i + 1) * input_reshaped[0].shape[1]], shape=input_shape[i])
        for i
        in range(len(inputs))]
    return outputs


def get_cross_stitch_network(input_shape, outputs):
    dropout_prob = 0.2
    cross_stitch_enabled = True

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    x = tf.keras.layers.Resizing(28, 28)(x)

    conv1 = [tf.keras.layers.Conv2D(32, 3, activation='relu')(x) for _ in range(len(outputs))]
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

    outputs = [tf.keras.layers.Dense(outputs[i], name='output' + str(i + 1))(dropout[i]) for i in range(len(outputs))]

    return tf.keras.Model(inputs, outputs)
