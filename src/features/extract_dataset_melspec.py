import warnings
import pandas as pd

from feature_utils import process_genres, get_dataset_info

warnings.simplefilter('ignore')


def main(dataset_nickname):
    path, dataset, metadata, key_dictionary, outputs = get_dataset_info(dataset_nickname)
    process_genres(3, 1.5, dataset, key_dictionary, metadata)


if __name__ == '__main__':
    dataset = 'semi-classical'
    main(dataset)
