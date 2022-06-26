import tensorflow as tf


def getTCN(input_layer):
    x1 = tf.keras.layers.Conv1D(16, 4, padding="same", activation='relu')(input_layer)
    x2 = tf.keras.layers.Conv1D(16, 4, padding="same", activation='relu', dilation_rate=2)(input_layer)
    x2 = tf.keras.layers.ELU(alpha=0.8)(x2)
    x2 = tf.keras.layers.SpatialDropout1D(rate=0.3)(x2)
    x2 = tf.keras.layers.Conv1D(16, 4, padding="same", activation='relu')(x2)
    out = tf.keras.layers.Add()([x1, x2])
    out = tf.keras.layers.ReLU()(out)

    return out


def get_madmom_model(input_shape, outputs):
    outputs = [17, 16, 1, 1]
    inputs = tf.keras.Input(shape=input_shape)

    tcn1 = getTCN(inputs)
    tcn2 = getTCN(tcn1)
    tcn3 = getTCN(tcn2)
    tcn4 = getTCN(tcn3)

    x1 = tf.keras.layers.Flatten()(tcn1)
    x1 = tf.keras.layers.Dropout(rate=0.4)(x1)
    x1 = tf.keras.layers.Dense(128)(x1)
    output1 = tf.keras.layers.Dense(outputs[0], name="output1")(x1)

    x2 = tf.keras.layers.Add()([tcn1, tcn2])
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Dropout(rate=0.4)(x2)
    x2 = tf.keras.layers.Dense(128)(x2)

    output2 = tf.keras.layers.Dense(outputs[1], name="output2")(x2)

    x3 = tf.keras.layers.Add()([tcn1, tcn2, tcn3])
    x3 = tf.keras.layers.GlobalAveragePooling1D()(x3)
    x3 = tf.keras.layers.Flatten()(x3)
    x3 = tf.keras.layers.Dropout(rate=0.4)(x3)
    x3 = tf.keras.layers.Dense(128)(x3)

    output3 = tf.keras.layers.Dense(outputs[2], name="output3")(x3)

    x4 = tf.keras.layers.Add()([tcn1, tcn2, tcn3, tcn4])
    x4 = tf.keras.layers.GlobalAveragePooling1D()(x4)
    x4 = tf.keras.layers.Flatten()(x4)
    x4 = tf.keras.layers.Dropout(rate=0.4)(x4)
    x4 = tf.keras.layers.Dense(128)(x4)

    output4 = tf.keras.layers.Dense(outputs[3], name="output4")(x4)

    return tf.keras.Model(inputs=inputs, outputs=[output1, output2, output3, output4])
