import os

import tensorflow as tf
from sklearn.model_selection import train_test_split

from data.make_dataset import make_dataset_ds
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
        # tf.keras.callbacks.ModelCheckpoint(
        #     filepath=model_save_path,
        #     monitor=monitor,
        #     verbose=2,
        #     save_best_only=True,
        #     save_weights_only=False,
        #     mode='auto',
        #     save_freq='epoch',
        # ),

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


def get_train_info(dataset_name, model_name, task, base_path):
    model_save_path = base_path + 'saved models/' + dataset_name + '/' + model_name + '/' + model_name + '_' + task + '/{epoch:02d}.h5'
    tensorboard_logs_path = base_path + 'reports/visualization/' + dataset_name + '/' + model_name + '/' + model_name + '_' + task + '/logs'
    model_plot_path = base_path + 'reports/figures/' + dataset_name + '/' + model_name + '/' + model_name + '_' + task + '.pdf'

    return model_save_path, tensorboard_logs_path, model_plot_path


def prepare_dataset(path, dataset_name, val_percentage, batch_size):
    if dataset_name in ['carnatic', 'hindustani']:
        pattern = '/*/*.wav'
    else:
        pattern = '/*/*/*.wav'

    filenames = tf.io.gfile.glob(path + pattern)
    labels = [song.split('/')[7].split('_')[-1] for song in filenames]
    train_files, val_files, _, _ = train_test_split(filenames, labels, test_size=val_percentage, random_state=42)

    train_ds = make_dataset_ds(filenames, batch_size)
    val_ds = make_dataset_ds(val_files, batch_size)
    return train_ds, val_ds


def _compile_model(model, dataset_name, model_name):
    if dataset_name == 'regional':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss={
                          'output1': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output3': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          'output4': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          'output5': tf.keras.losses.MeanSquaredError(),
                      },
                      metrics=['accuracy'])
    elif dataset_name == 'folk':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss={
                          'output1': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output3': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          'output4': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                          'output5': tf.keras.losses.MeanSquaredError(),
                      },
                      metrics=['accuracy'])
    elif dataset_name == 'hindustani':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss={
                          'output1': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output3': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output4': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output5': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output6': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output7': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output8': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      },
                      metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss={
                          'output1': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output3': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output4': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output5': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          'output6': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      },
                      metrics=['accuracy'])


def compile_train_model(model, dataset_name, model_name, train_ds, val_ds,
                               model_save_path, tensorboard_logs_path, epochs):
    _compile_model(model, dataset_name, model_name)
    return model.fit(train_ds,
                     validation_data=val_ds,
                     epochs=epochs,
                     callbacks=get_callbacks(model_save_path, tensorboard_logs_path, monitor='val_output1_accuracy'))
