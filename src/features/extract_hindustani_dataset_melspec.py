import warnings
import pandas as pd
from src.features.utils import process_genres

warnings.simplefilter('ignore')

dataset = 'Indian Hindustani Classical Music'
metadata = pd.read_excel('../../data/raw/Indian Hindustani Classical Music/hindustani_raags_metadata.xlsx')


if __name__ == '__main__':
    key_dictionary = {
        'mel_spec': [],
        'genre': [],
        'swaras': [],
        'jati': [],
        'vadi': [],
        'samvadi': [],
        'aaroh': [],
        'avroh': [],
        'genre_id': [],
        'swaras_set_id': [],
        'jati_id': [],
        'thaat_id': [],
        'vadi_id': [],
        'samvadi_id': [],
        'aaroh_set_id': [],
        'avroh_set_id': [],
    }

    process_genres(3, 0.5, dataset, key_dictionary, metadata)
