import os
import tensorflow as tf

from data.load_data import get_dataset
from data.split_data import train_val_split
from models import baseline, ca_mtl, cross_stitch, madmom, mrn, mtl_f0


def setup_gpu_state():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.mixed_precision.set_global_policy('mixed_float16')


def plot_model_diagram(model, path):
    root_dir = '/'.join(path.split('/')[:-1])
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    tf.keras.utils.plot_model(model, path, dpi=600, show_shapes=True)


def get_callbacks(model_save_path, tensorboard_logs_path, monitor):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor,
            verbose=2,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch',
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.1,
            patience=10,
            verbose=0,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_logs_path,
            histogram_freq=1,
            write_images=True,
            update_freq='epoch',
        )

    ]


def get_model_func(model_name):
    model_dict = {
        'baseline': baseline.get_baseline_model,
        'ca_mtl': ca_mtl.get_ca_mtl_model,
        'cross_stitch': cross_stitch.get_cross_stitch_network,
        'madmom': madmom.get_madmom_model,
        'mrn': mrn.get_mrn_model,
        'mtl_f0': mtl_f0.get_multi_f0_model
    }

    return model_dict[model_name]


def get_regional_one_labels(mel, y1, y2, y3, y4, y5):
    return mel, y1


def get_regional_all_labels(mel, y1, y2, y3, y4, y5):
    return mel, (y1, y2, y3, y4, y5)


def get_train_info(dataset_name, model_name, task):
    model_save_path = '/media/B/multitask_indian_music_classification/saved models/' + dataset_name + '/' + model_name + '/' + model_name + '_' + task + '.h5'
    tensorboard_logs_path = '/media/B/multitask_indian_music_classification/reports/visualization/' + dataset_name + '/' + model_name + '/' + model_name + '_' + task + '/logs'
    model_plot_path = '/media/B/multitask_indian_music_classification/reports/figures/' + dataset_name + '/' + model_name + '/' + model_name + '_' + task + '.pdf'

    return model_save_path, tensorboard_logs_path, model_plot_path


def prepare_regional_ds(model_name, path, dataset_dict, train_percentage, val_percentage, batch_size):
    dataset = get_dataset(path, dataset_dict)
    (X_train, y1_train, y2_train, y3_train, y4_train, y5_train), (
        X_test, y1_test, y2_test, y3_test, y4_test, y5_test) = train_val_split(dataset, train_percentage,
                                                                               val_percentage)

    # train_ds = tf.data.Dataset.from_tensor_slices((X_train, y1_train, y2_train, y3_train, y4_train))
    # val_ds = tf.data.Dataset.from_tensor_slices((X_val, y1_val, y2_val, y3_val, y4_val))
    # test_ds = tf.data.Dataset.from_tensor_slices((X_test, y1_test, y2_test, y3_test, y4_test, y5_test))

    # label_func = get_regional_one_labels # if model_name == 'ca_mtl' else get_regional_two_labels
    # train_ds = (train_ds
    #             .map(label_func, num_parallel_calls=tf.data.AUTOTUNE)
    #             .batch(batch_size)
    #             .prefetch(tf.data.AUTOTUNE)
    #             
    #             )
    # val_ds = (val_ds
    #           .map(label_func, num_parallel_calls=tf.data.AUTOTUNE)
    #           .batch(batch_size)
    #           .prefetch(tf.data.AUTOTUNE)
    #           .cache()
    #           )

    # test_ds = (test_ds
    #           .map(label_func, num_parallel_calls=tf.data.AUTOTUNE)
    #           .batch(batch_size)
    #           .prefetch(tf.data.AUTOTUNE)
    #           .cache()
    #           )
    # y_train = np.dstack((y1_train, y2_train, y3_train, y4_train))
    # y_val = np.dstack((y1_val, y2_val, y3_val, y4_val))
    # y_test = np.dstack((y1_test, y2_test, y3_test, y4_test))
    return (X_train, y5_train), (X_test, y5_test)


def _compile_model(model, dataset_name, model_name):
    if dataset_name == 'regional':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss={
                          'output1': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          # 'output2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          # 'output3': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          # 'output4': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          # 'output5': tf.keras.losses.MeanSquaredError(),
                      },
                      metrics=['accuracy'])
    elif dataset_name == 'folk':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss={
                          'output1': tf.keras.losses.MeanSquaredError(),
                          # 'output2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          # 'output3': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          # 'output4': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          # 'output5': tf.keras.losses.MeanSquaredError(),
                      },
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss={
                          'output1': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          # 'output2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          # 'output3': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          # 'output4': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          # 'output5': tf.keras.losses.MeanSquaredError(),
                      },
                      metrics=['accuracy'])


def compile_train_for_regional(model, dataset_name, model_name, train_ds, val_split,
                               batch, model_save_path, tensorboard_logs_path, epochs):
    _compile_model(model, dataset_name, model_name)
    return model.fit(x=train_ds[0], y=train_ds[1],
                     validation_split=val_split,
                     batch_size=batch,
                     epochs=epochs,
                     callbacks=get_callbacks(model_save_path, tensorboard_logs_path, monitor='val_accuracy'))
