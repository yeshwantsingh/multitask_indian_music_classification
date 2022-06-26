import tensorflow as tf


def get_multi_f0_model(input_shape, outputs):
    outputs = [17, 16, 1, 1]
    inputs = tf.keras.Input(shape=input_shape)

    reshaped_inputs = tf.expand_dims(inputs, -1)

    y0 = tf.keras.layers.BatchNormalization()(reshaped_inputs)

    y1_pitch = tf.keras.layers.Conv2D(128, 5, padding='same', activation='relu')(y0)
    y1a_pitch = tf.keras.layers.BatchNormalization()(y1_pitch)

    y2_pitch = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(y1a_pitch)
    y2a_pitch = tf.keras.layers.BatchNormalization()(y2_pitch)

    y3_pitch = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(y2a_pitch)
    y3a_pitch = tf.keras.layers.BatchNormalization()(y3_pitch)

    y4_pitch = tf.keras.layers.Conv2D(8, (70, 3), padding='same', activation='relu')(y3a_pitch)
    y4a_pitch = tf.keras.layers.BatchNormalization()(y4_pitch)

    y_multif0 = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu', name='y_multif0')(y4a_pitch)
    y_multif0 = tf.keras.layers.MaxPool2D(2)(y_multif0)

    output1 = tf.keras.layers.Flatten()(y_multif0)
    output1 = tf.keras.layers.Dense(outputs[0], name="output1")(output1)

    y1_timbre = tf.keras.layers.Conv2D(256, 1, padding='same', activation='relu')(y0)
    y1a_timbre = tf.keras.layers.BatchNormalization()(y1_timbre)
    y1a_timbre = tf.keras.layers.MaxPool2D(2)(y1a_timbre)

    y2_timbre = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(y1a_timbre)
    y2a_timbre = tf.keras.layers.BatchNormalization()(y2_timbre)

    y_concat = tf.keras.layers.Concatenate()([y_multif0, y1a_timbre, y2a_timbre])

    # Melody filters
    y_mel_feat = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(y_concat)
    y_mel_feat2 = tf.keras.layers.Conv2D(1, (180, 1), padding='same', activation='relu')(y_mel_feat)

    # Bass filters
    y_bass_feat = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(y_concat)
    y_bass_feat2 = tf.keras.layers.Conv2D(1, (180, 1), padding='same', activation='relu')(y_bass_feat)

    # piano filters
    y_piano_feat = tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu')(y_concat)
    y_piano_feat2 = tf.keras.layers.Conv2D(1, (180, 1), padding='same', activation='relu')(y_piano_feat)

    # Melody presqueeze
    y_melody = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu', name='y_melody')(y_mel_feat2)
    y_melody = tf.keras.layers.MaxPool2D(2)(y_melody)
    y_melody = tf.keras.layers.Flatten()(y_melody)

    output2 = tf.keras.layers.Dense(outputs[1], name='output2')(y_melody)

    # Bass presqueeze
    y_bass = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu', name='y_bass')(y_bass_feat2)
    y_bass = tf.keras.layers.MaxPool2D(2)(y_bass)
    y_bass = tf.keras.layers.Flatten()(y_bass)
    output3 = tf.keras.layers.Dense(outputs[2], name="output3")(y_bass)

    # Piano presqueeze
    y_piano = tf.keras.layers.Conv2D(1, 1, padding='same', activation='relu', name='y_piano')(y_piano_feat2)
    y_piano = tf.keras.layers.Flatten()(y_piano)
    output4 = tf.keras.layers.Dense(outputs[3], name="output4")(y_piano)

    return tf.keras.Model(inputs=inputs, outputs=[output1, output2, output3, output4])