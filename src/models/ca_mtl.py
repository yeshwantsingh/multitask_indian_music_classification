import tensorflow as tf


def attention_block(inputs):
    x = tf.keras.layers.Conv1D(16, 4, padding="same", activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(4, 2, padding="same")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv1D(32, 4, padding="same", activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(4, 2, padding="same")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    out = tf.keras.layers.AdditiveAttention()([x, x])

    out = tf.keras.layers.Concatenate()(
        [x, out])

    return out


def get_ca_mtl_model(input_shape, output_nodes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    x1 = attention_block(x)
    x2 = attention_block(x)
    x3 = attention_block(x)

    # x4 = tf.keras.layers.Attention()([x1, x2])
    # x5 = tf.keras.layers.Attention()([x1, x2])
    # x6 = tf.keras.layers.Attention()([x2, x3])

    x = tf.keras.layers.Concatenate()([x1, x2, x3])
    # x = tf.keras.layers.Attention()([x, x])

    x = tf.keras.layers.Conv1D(32, 2, padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(16, 2, padding='same', activation='relu')(x)

    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.GlobalMaxPool1D()(x)
    # x = tf.keras.layers.Flatten()(x)

    #
    #
    # x = tf.keras.layers.Conv1D(32, 10, padding="same", activation='relu')(x)
    # x = tf.keras.layers.MaxPooling1D(10, 5, padding="same")(x)
    # x = tf.keras.layers.Dropout(0.1)(x)
    #
    # x = tf.keras.layers.Conv1D(128, 10, padding="same", activation='relu')(x)
    # x = tf.keras.layers.MaxPooling1D(10, 5, padding="same")(x)
    #
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    # encoder = layers.Dense(12)(x)
    # x = layers.Dense(12, name="output")(encoder)

    outputs = [tf.keras.layers.Dense(out, name="output"+str(i+1))(x) for i, out in enumerate(output_nodes)]

    return tf.keras.Model(inputs=inputs, outputs=outputs)