import tensorflow as tf


def getTCN(inputs):
    x1 = tf.keras.layers.Conv1D(16, 4, padding="same", activation='relu')(inputs)
    x2 = tf.keras.layers.Conv1D(16, 4, padding="same", activation='relu', dilation_rate=2)(inputs)
    x2 = tf.keras.layers.ELU(alpha=0.8)(x2)
    x2 = tf.keras.layers.SpatialDropout1D(rate=0.3)(x2)
    x2 = tf.keras.layers.Conv1D(16, 4, padding="same", activation='relu')(x2)
    out = tf.keras.layers.Add()([x1, x2])
    out = tf.keras.layers.ReLU()(out)

    return out


def aggregation_block(inputs):
    x = tf.keras.layers.Add()(inputs)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(rate=0.4)(x)
    x = tf.keras.layers.Dense(128)(x)
    return x


def get_madmom_model(input_shape, outputs):
    inputs = tf.keras.Input(shape=input_shape)

    tcn = []
    for _ in range(len(outputs)):
        x = inputs
        if len(tcn) == 0:
            x = getTCN(inputs)
        else:
            x = getTCN(x)
        tcn.append(x)

    for i in range(1, len(outputs)+1):
        if i == 1:
            x = tf.keras.layers.Flatten()(tcn[0])
            x = tf.keras.layers.Dropout(rate=0.4)(x)
            x = tf.keras.layers.Dense(128)(x)
        else:
            x = aggregation_block(tcn[0:i])
        outputs[i-1] = tf.keras.layers.Dense(outputs[i-1], name="output"+str(i))(x)

    return tf.keras.Model(inputs, outputs)

