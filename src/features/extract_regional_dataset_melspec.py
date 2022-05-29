import warnings
import pandas as pd

from src.features.utils import process_genres

warnings.simplefilter('ignore')

dataset = 'Indian Regional Music'
metadata = pd.read_excel('../../data/raw/Indian Regional Music/regional_metadata.xlsx')


if __name__ == '__main__':
    key_dictionary = {
        'mel_spec': [],
        'language': [],
        'genre': [],
        'artist': [],
        'gender': [],
        'song': [],
        'genre_id': [],
        'artist_id': [],
        'gender_id': [],
        'song_id': [],
        'local_song_id': [],
        'no_of_artists': [],
    }

    process_genres(3, 0.5, dataset, key_dictionary, metadata)