import os
import pickle
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

datasets_path = '../../data/raw/'
features_path = '../../data/processed/mel_spec'


def append(data, index, key, metadata):
    data[key].append(str(metadata[metadata.index == index][key].values[0]))


def convert(data, key, type=None):
    if type == 'S':
        data[key] = np.array(data[key]).astype('S')
    else:
        data[key] = np.array(data[key])


def extract_mel(x, sr=22050):
    return librosa.power_to_db((librosa.feature.melspectrogram(y=x, sr=sr)), ref=np.max)


def process_song(song, sr, frame, slide):
    slices = [song[int(i * sr): int((i + frame) * sr)] for i in np.arange(0, int(len(song) // sr), slide)[:-5]]

    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_mel, slices, [sr]*len(slices))
    return result


def process_genre(genre, frame, slide, key_dictionary, dataset, metadata):
    def _process_songs(_artist, _dataset, _frame, _genre, _key_dictionary, _metadata, _slide):
        for song in tqdm(os.listdir(os.path.join(datasets_path, _dataset, _genre, _artist))):
            x, sr = librosa.load(os.path.join(datasets_path, _dataset, _genre, _artist, song), sr=22050)
            mel_spec = process_song(x, sr, _frame, _slide)
            if mel_spec:
                for mel in mel_spec:
                    _key_dictionary['mel_spec'].append(mel)
                    _key_dictionary['genre'].append(str(_genre))
                    for key in _key_dictionary.keys():
                        if key not in ['mel_spec', 'genre']:
                            append(_key_dictionary, int(song.split('.')[0]), key, _metadata)
    if dataset not in ['Indian Carnatic Classical Music', 'Indian Hindustani Classical Music']:
        for artist in tqdm(os.listdir(os.path.join(datasets_path, dataset, genre))):
            _process_songs(artist, dataset, frame, genre, key_dictionary, metadata, slide)
    else:
        _process_songs(None, dataset, frame, genre, key_dictionary, metadata, slide)

    for key in key_dictionary.keys():
        if '-' in key:
            convert(key_dictionary, key)
        else:
            convert(key_dictionary, key, 'S')

    return key_dictionary


def process_genres(frame, slide, dataset, key_dictionary, metadata):
    dataset_path = os.path.join(datasets_path, dataset)
    genres = os.listdir(dataset_path)
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
