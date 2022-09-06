import os
import warnings

import numpy as np
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.simplefilter('ignore')

from train_utils import prepare_dataset, setup_gpu_state, get_train_info, get_model_func, compile_train_model

from data.make_dataset import get_dataset_info

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

setup_gpu_state()
snapshots_path = '/media/A/CarnaticSnapshots/1'


def get_label(x, y):
    return x, y[0]


def main(dataset_nick_name, model_name, task, epochs, val_split, batch_size, base_path):
    input_shape = (259, 256)
    model_save_path, tensorboard_logs_path, model_plot_path = get_train_info(dataset_nick_name, model_name, task,
                                                                             base_path)

    dataset_path, outputs = get_dataset_info(base_path, dataset_nick_name)
    train_ds, val_ds = prepare_dataset(dataset_path, dataset_nick_name, val_split, batch_size)
    #
    tf.data.experimental.save(train_ds, snapshots_path + '/Train')
    tf.data.experimental.save(val_ds, snapshots_path + '/Val')

    # train_ds = tf.data.experimental.load(snapshots_path + '/Train')
    # val_ds = tf.data.experimental.load(snapshots_path + '/Val')

    # train_ds = train_ds.map(map_func=get_label, num_parallel_calls=tf.data.AUTOTUNE)
    # val_ds = val_ds.map(map_func=get_label, num_parallel_calls=tf.data.AUTOTUNE)

    # ds = val_ds.filter(lambda x, y: tf.reduce_any(tf.math.is_nan(x)))
    # print(len(list(ds.as_numpy_iterator())))

    # model = get_model_func(model_name)
    # model = model(input_shape, outputs[:1])
    # #
    # # # model.summary()
    # #
    # # # plot_model_diagram(model, model_plot_path)
    # # #
    # compile_train_model(model, dataset_nick_name, model_name,
    #                     train_ds, val_ds, model_save_path,
    #                     tensorboard_logs_path, epochs)


if __name__ == '__main__':
    dataset = 'carnatic'
    model_name = 'baseline'
    task = 'task1'
    base_path = '/media/B/multitask_indian_music_classification/'
    epochs = 100
    batch_size = 256
    val_split = .2
    main(dataset, model_name, task, epochs, val_split, batch_size, base_path)
