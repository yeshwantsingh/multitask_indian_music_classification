import pickle
import os
import numpy as np


def get_dataset(path, dataset_dict):
    """
    Load the regional music dataset from the provided dataset path.

    Args:
        path(str): Dataset path of the regional dataset.

    Returns:
        dataset(dict): Dictionary containing the load regional dataset.
    """

    for language in os.listdir(path):
        for artist in os.listdir(os.path.join(path, language)):
            with open(os.path.join(path, language, artist), 'rb') as fp:
                data = pickle.load(fp)
                for key in dataset_dict.keys():
                    dataset_dict[key] = [*dataset_dict[key], *data[key]]

    for key in dataset_dict.keys():
        if key == 'mel_spec':
            dataset_dict[key] = np.array(dataset_dict[key], dtype='float32')
        else:
            dataset_dict[key] = np.array(dataset_dict[key], dtype='int')

    return dataset_dict
