import pandas as pd
from utils import process_genres

import warnings
warnings.simplefilter('ignore')

dataset = 'Indian Folk Music'
features_path = '../../data/processed/mel_spec'

metadata = pd.read_excel('../../data/raw/' + dataset + '/' + 'Indian Folk Music.xlsx', index_col='index')


if __name__ == '__main__':
    key_dictionary = {
        'mel_spec': [],
        'genre': [],
        'state': [],
        'artist': [],
        'gender': [],
        'song': [],
        'source': [],
        'no_of_artists': [],
        'genre_id': [],
        'state_id': [],
        'artist_id': [],
        'gender_id': [],
    }
    process_genres(3, 0.5, dataset, key_dictionary, metadata)
