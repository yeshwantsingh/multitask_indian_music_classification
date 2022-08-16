import tensorflow as tf


def get_multi_f0_model(input_shape, outputs):
    # outputs = [17, 16, 1, 1]
    inputs = tf.keras.Input(shape=input_shape)

    reshaped_inputs = tf.expand_dims(inputs, -1)

    y0 = tf.keras.layers.BatchNormalization()(reshaped_inputs)

    y1_pitch = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(y0)
    y1a_pitch = tf.keras.layers.BatchNormalization()(y1_pitch)

    y2_pitch = tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu')(y1a_pitch)
    y2a_pitch = tf.keras.layers.BatchNormalization()(y2_pitch)

    y3_pitch = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(y2a_pitch)
    y3a_pitch = tf.keras.layers.BatchNormalization()(y3_pitch)

    y4_pitch = tf.keras.layers.Conv2D(8, (5, 3), padding='same', activation='relu')(y3a_pitch)
    y4a_pitch = tf.keras.layers.BatchNormalization()(y4_pitch)

    y_multif0 = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu', name='y_multif0')(y4a_pitch)
    y_multif0 = tf.keras.layers.MaxPool2D(2)(y_multif0)

    output1 = tf.keras.layers.Flatten()(y_multif0)
    outputs[0] = tf.keras.layers.Dense(outputs[0], name="output1")(output1)

    y1_timbre = tf.keras.layers.Conv2D(16, 1, padding='same', activation='relu')(y0)
    y1a_timbre = tf.keras.layers.BatchNormalization()(y1_timbre)
    y1a_timbre = tf.keras.layers.MaxPool2D(2)(y1a_timbre)

    y2_timbre = tf.keras.layers.Conv2D(15, 5, padding='same', activation='relu')(y1a_timbre)
    y2a_timbre = tf.keras.layers.BatchNormalization()(y2_timbre)

    y_concat = tf.keras.layers.Concatenate()([y_multif0, y1a_timbre, y2a_timbre])

    for i in range(1, len(outputs)):
        x = tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu')(y_concat)
        x = tf.keras.layers.Conv2D(1, (18, 1), padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        outputs[i] = tf.keras.layers.Dense(outputs[i], name='output' + str(i + 1))(x)

    return tf.keras.Model(inputs, outputs)
