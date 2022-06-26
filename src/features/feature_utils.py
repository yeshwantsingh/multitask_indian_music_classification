import os
import pickle
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from tqdm import tqdm

datasets_path = '/media/B/multitask_indian_music_classification/data/raw/'
features_path = '/media/B/multitask_indian_music_classification/data/processed/mel_spec'


def get_dataset_info(dataset_nick_name):
    path, dataset, metadata, key_dictionary, outputs = None, None, None, None, None
    if dataset_nick_name == 'carnatic':
        dataset = 'Indian Carnatic Classical Music'
        path = features_path + '/' + dataset
        metadata = pd.read_excel(datasets_path + dataset + '/' + 'carnatic_raags_metadata.xlsx')
        key_dictionary = {
            'mel_spec': [],
            'genre_id': [],
            'swaras_set_id': [],
            'melakarta_id': [],
            'aaroh_set_id': [],
            'avroh_set_id': [],
            'janak_id': [],
        }
        outputs = [
            len(np.unique(key_dictionary['genre_id'])),
            len(np.unique(key_dictionary['swaras_set_id'])),
            len(np.unique(key_dictionary['melakarta_id'])),
            len(np.unique(key_dictionary['aaroh_set_id'])),
            len(np.unique(key_dictionary['avroh_set_id'])),
            len(np.unique(key_dictionary['janak_id']))
        ]

    elif dataset_nick_name == 'hindustani':
        dataset = 'Indian Hindustani Classical Music'
        path = features_path + '/' + dataset
        metadata = pd.read_excel(datasets_path + dataset + '/' + 'hindustani_raags_metadata.xlsx')
        key_dictionary = {
            'mel_spec': [],
            'genre_id': [],
            'swaras_set_id': [],
            'jati_id': [],
            'thaat_id': [],
            'vadi_id': [],
            'samvadi_id': [],
            'aaroh_set_id': [],
            'avroh_set_id': [],
        }
        outputs = [
            len(np.unique(key_dictionary['genre_id'])),
            len(np.unique(key_dictionary['swaras_set_id'])),
            len(np.unique(key_dictionary['melakarta_id'])),
            len(np.unique(key_dictionary['aaroh_set_id'])),
            len(np.unique(key_dictionary['avroh_set_id'])),
            len(np.unique(key_dictionary['janak_id']))
        ]

    elif dataset_nick_name == 'regional':
        dataset = 'Indian Regional Music'
        path = features_path + '/' + dataset
        metadata = pd.read_excel(datasets_path + dataset + '/' + 'Indian Regional Music.xlsx', index_col='s_no')
        key_dictionary = {
            's_no': [],
            'mel_spec': [],
            'location_id': [],
            'artist_id': [],
            'gender_id': [],
            'veteran': [],
            'no_of_artists': [],
        }
        outputs = [17, 68, 1, 1, 1]

    elif dataset_nick_name == 'folk':
        dataset = 'Indian Folk Music'
        path = features_path + '/' + dataset
        metadata = pd.read_excel(datasets_path + dataset + '/' + 'Indian Folk Music.xlsx', index_col='s_no')
        key_dictionary = {
            's_no': [],
            'mel_spec': [],
            'genre_id': [],
            'state_id': [],
            'artist_id': [],
            'gender_id': [],
            'no_of_artists': [],
        }
        outputs = [15, 12, 126, 1, 1]

    elif dataset_nick_name == 'semi-classical':
        dataset = 'Indian Semi Classical Music'
        path = features_path + '/' + dataset
        metadata = pd.read_excel(datasets_path + dataset + '/' + 'Indian Semi Classical Music.xlsx',
                                 index_col='s_no')
        key_dictionary = {
            's_no': [],
            'mel_spec': [],
            'genre_id': [],
            'artist_id': [],
            'gender_id': [],
            'no_of_artists': [],
        }
        outputs = [9, 49, 1, 1]
    return path, dataset, metadata, key_dictionary, outputs


def process_song(song, sr, frame, slide):
    def extract_mel(x, sr=22050):
        return librosa.power_to_db((librosa.feature.melspectrogram(y=x, sr=sr)), ref=np.max)

    slices = [song[int(i * sr): int((i + frame) * sr)] for i in np.arange(0, int(len(song) // sr), slide)[:-5]]

    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_mel, slices, [sr] * len(slices))
    return result


def process_songs(artist, dataset, frame, slide, genre, key_dictionary, metadata):
    for song in tqdm(os.listdir(os.path.join(datasets_path, dataset, genre, artist))):
        x, sr = librosa.load(os.path.join(datasets_path, dataset, genre, artist, song), sr=22050)
        mel_spec = process_song(x, sr, frame, slide)
        if mel_spec:
            index = song.split('.')[0]
            for mel in mel_spec:
                key_dictionary['mel_spec'].append(mel)
                key_dictionary['s_no'].append(index)
                for key in key_dictionary.keys():
                    if key not in ['mel_spec', 's_no']:
                        key_dictionary[key].append(str(metadata[metadata.index == int(index)][key].values[0]))


def process_genre(genre, frame, slide, key_dictionary, dataset, metadata):
    if dataset not in ['Indian Carnatic Classical Music', 'Indian Hindustani Classical Music']:
        for artist in tqdm(os.listdir(os.path.join(datasets_path, dataset, genre))):
            process_songs(artist, dataset, frame, slide, genre, key_dictionary, metadata)
    else:
        process_songs(None, dataset, frame, slide, genre, key_dictionary, metadata)

    for key in key_dictionary.keys():
        if key == 'mel_spec':
            key_dictionary[key] = np.array(key_dictionary[key], dtype='float')
        else:
            key_dictionary[key] = np.array(key_dictionary[key])

    return key_dictionary


def process_genres(frame, slide, dataset, key_dictionary, metadata):
    path = os.path.join(datasets_path, dataset)
    genres = os.listdir(path)
    genres = [folder for folder in genres if os.path.isdir(os.path.join(datasets_path, dataset, folder))]

    for genre in tqdm(genres):
        for key in key_dictionary.keys():
            key_dictionary[key] = []
        data = process_genre(genre, frame, slide, key_dictionary, dataset, metadata)
        dst_path = os.path.join(features_path, dataset, genre)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path, exist_ok=True)

        with open(os.path.join(dst_path, genre + '.pickle'), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
