import os
import warnings

import numpy as np
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

warnings.simplefilter('ignore')

from train_utils import prepare_dataset, \
    plot_model_diagram, compile_train_model, \
    setup_gpu_state, get_train_info, get_model_func

from data.make_dataset import get_dataset_info

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

setup_gpu_state()


def main(dataset_nick_name, model_name, task, epochs, val_split, batch_size):
    # input_shape = (128, 130)
    input_shape = (259, 256)
    model_save_path, tensorboard_logs_path, model_plot_path = get_train_info(dataset_nick_name, model_name, task)

    dataset_path, outputs = get_dataset_info(dataset_nick_name)
    train_ds, val_ds = prepare_dataset(dataset_path, dataset_nick_name, val_split, batch_size)
    model = get_model_func(model_name)
    model = model(input_shape, outputs[0:1])

    plot_model_diagram(model, model_plot_path)

    compile_train_model(model, dataset_nick_name, model_name,
                               train_ds, val_ds, model_save_path,
                               tensorboard_logs_path, epochs)
    # model = tf.keras.models.load_model(model_save_path)
    # model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
    #           callbacks=get_callbacks(model_save_path, tensorboard_logs_path, 'val_accuracy'))
    # acc = model.evaluate(test_ds[0], test_ds[1])
    # print(acc)


if __name__ == '__main__':
    dataset = 'folk'
    model_name = 'baseline'
    task = 'genre'
    epochs = 100
    batch_size = 256
    val_split = .2
    main(dataset, model_name, task, epochs, val_split, batch_size)
