import os
import numpy as np
import tensorflow as tf
import warnings

warnings.simplefilter('ignore')

from train_utils import prepare_regional_ds, \
    plot_model_diagram, compile_train_for_regional, \
    setup_gpu_state, get_train_info, get_model_func, get_callbacks

from features.feature_utils import get_dataset_info

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

setup_gpu_state()


def main(dataset_nick_name, model_name, task, epochs, train_split, val_split, batch_size):
    input_shape = (128, 130)

    model_save_path, tensorboard_logs_path, model_plot_path = get_train_info(dataset_nick_name, model_name, task)

    path, _, _, dataset_dict, outputs = get_dataset_info(dataset_nick_name)
    (X_train, y_train), (X_test, y_test) = prepare_regional_ds(model_name, path, dataset_dict, train_split, val_split,
                                                               batch_size)
    model = get_model_func(model_name)
    model = model(input_shape, outputs[4:5])

    plot_model_diagram(model, model_plot_path)

    train_ds = (X_train, y_train)
    # test_ds = (X_test, y_test)

    compile_train_for_regional(model, dataset_nick_name, model_name,
                               train_ds, val_split / 100, batch_size, model_save_path,
                               tensorboard_logs_path, epochs)
    # model = tf.keras.models.load_model(model_save_path)
    # model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
    #           callbacks=get_callbacks(model_save_path, tensorboard_logs_path, 'val_accuracy'))
    # acc = model.evaluate(test_ds[0], test_ds[1])
    # print(acc)


if __name__ == '__main__':
    dataset = 'folk'
    model_name = 'baseline'
    task = 'num_artists'
    epochs = 100
    batch_size = 128
    train_split, val_split = 80, 20
    main(dataset, model_name, task, epochs, train_split, val_split, batch_size)
