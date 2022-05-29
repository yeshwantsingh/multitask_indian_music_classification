import warnings
import pandas as pd
from src.features.utils import process_genres

warnings.simplefilter('ignore')

dataset = 'Indian Carnatic Classical Music'
metadata = pd.read_excel('../../data/raw/Indian Carnatic Classical Music/carnatic_raags_metadata.xlsx')


if __name__ == '__main__':
    key_dictionary = {
        'mel_spec': [],
        'genre': [],
        'swaras': [],
        'melakarta': [],
        'aaroh': [],
        'avroh': [],
        'janak/janya': [],
        'genre_id': [],
        'swaras_set_id': [],
        'melakarta_id': [],
        'aaroh_set_id': [],
        'avroh_set_id': [],
        'janak_id': [],
    }

    process_genres(3, 0.5, dataset, key_dictionary, metadata)
