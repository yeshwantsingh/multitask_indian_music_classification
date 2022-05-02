from tensorflow.keras import layers, Model


def get_model(shape, output_shapes):
    inputs = layers.Input(shape=shape)

    x = layers.Conv1D(16, 4, padding="same", activation='relu')(inputs)
    x = layers.MaxPooling1D(4, 2, padding="same")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv1D(32, 4, padding="same", activation='relu')(x)
    x = layers.MaxPooling1D(4, 2, padding="same")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv1D(32, 10, padding="same", activation='relu')(x)
    x = layers.MaxPooling1D(10, 5, padding="same")(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Conv1D(128, 10, padding="same", activation='relu')(x)
    x = layers.MaxPooling1D(10, 5, padding="same")(x)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)

    # encoder = layers.Dense(12)(x)
    # x = layers.Dense(12, name="output")(encoder)
    # attention = layers.Attention()([encoder, x])

    label1pred = layers.Dense(output_shapes[0], name="output1")(x)
    label2pred = layers.Dense(output_shapes[1], name="output2")(x)
    label3pred = layers.Dense(output_shapes[2], name="output3")(x)
    label4pred = layers.Dense(output_shapes[3], name="output4")(x)
    label5pred = layers.Dense(output_shapes[4], name="output5")(x)

    return Model(inputs=inputs, outputs=[label1pred, label2pred, label3pred, label4pred, label5pred])


if __name__ == '__main__':
    model = get_model((128, 130), [17, 64, 2, 340, 5])
    print(model.summary())
