import tensorflow as tf


def get_model():
    input_shape = (None, None, 5)
    inputs = tf.keras.Input(shape=input_shape)

    y0 = tf.keras.layers.BatchNormalization()(inputs)

    y1_pitch = tf.keras.layers.Conv2D(
        128, (5, 5), padding='same', activation='relu', name='pitch_layer1')(y0)
    y1a_pitch = tf.keras.layers.BatchNormalization()(y1_pitch)
    y2_pitch = tf.keras.layers.Conv2D(
        32, (5, 5), padding='same', activation='relu', name='pitch_layer2')(y1a_pitch)
    y2a_pitch = tf.keras.layers.BatchNormalization()(y2_pitch)
    y3_pitch = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='smoothy2')(y2a_pitch)
    y3a_pitch = tf.keras.layers.BatchNormalization()(y3_pitch)
    y4_pitch = tf.keras.layers.Conv2D(8, (70, 3), padding='same', activation='relu', name='distribute')(y3a_pitch)
    y4a_pitch = tf.keras.layers.BatchNormalization()(y4_pitch)

    y1_timbre = tf.keras.layers.Conv2D(
        512, (1, 1), padding='same', activation='relu', name='timbre_layer1')(y0)
    y1a_timbre = tf.keras.layers.BatchNormalization()(y1_timbre)
    # y2_timbre = Conv2D(
    #     32, (5, 5), padding='same', activation='relu', name='timbre_layer2')(y1a_timbre)
    # y2a_timbre = BatchNormalization()(y2_timbre)

    y_multif0 = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='multif0_presqueeze')(y4a_pitch)
    multif0 = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3), name='multif0')(y_multif0)

    y_concat = tf.keras.layers.Concatenate(name='timbre_and_pitch')([y_multif0, y1a_timbre])

    y_mel_feat = tf.keras.layers.Conv2D(
        32, (5, 5), padding='same', activation='relu', name='melody_filters')(y_concat)
    y_mel_feat2 = tf.keras.layers.Conv2D(
        1, (360, 1), padding='same', activation='relu', name='melody_filters2')(y_mel_feat)
    y_bass_feat = tf.keras.layers.Conv2D(
        32, (5, 5), padding='same', activation='relu', name='bass_filters')(y_concat)
    y_bass_feat2 = tf.keras.layers.Conv2D(
        1, (360, 1), padding='same', activation='relu', name='bass_filters2')(y_bass_feat)
    y_piano_feat = tf.keras.layers.Conv2D(
        32, (5, 5), padding='same', activation='relu', name='piano_filters')(y_concat)
    y_piano_feat2 = tf.keras.layers.Conv2D(
        1, (360, 1), padding='same', activation='relu', name='piano_filters2')(y_piano_feat)

    y_melody = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='melody_presqueeze')(y_mel_feat2)
    melody = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3), name='melody')(y_melody)

    y_bass = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='bass_presqueeze')(y_bass_feat2)
    bass = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3), name='bass')(y_bass)

    y_piano = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same', activation='sigmoid', name='piano_presqueeze')(y_piano_feat2)
    piano = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=3), name='piano')(y_piano)

    model =  tf.keras.Model(inputs=inputs, outputs=[multif0, melody, bass, piano])
    model.compile(optimizer='adam', loss='mse')

    return model

if __name__ == '__main__':
    model = get_model()
    model.summary()
