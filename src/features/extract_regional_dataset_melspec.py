import os
import numpy as np
import librosa
import pandas as pd
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.simplefilter('ignore')

datasets_path = '../../data/raw/'
dataset = 'regional music dataset'
features_path = '../../data/processed/mel_spec'


metadata = pd.read_excel('../../data/raw/regional music dataset/regional_metadata.xlsx')


def extract_mel(x, sr=22050):
    return librosa.power_to_db((librosa.feature.melspectrogram(y=x, sr=sr)), ref=np.max)


def process_song(song, sr, frame, slide):
    slices = [song[int(i * sr): int((i + frame) * sr)] for i in np.arange(0, int(len(song) // sr), slide)[:-5]]

    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_mel, slices, [sr]*len(slices))
    return result


def process_genre(artist, language, frame, slide):
    data = {
        'mel_spec': [],
        'language': [],
        'location': [],
        'artist': [],
        'gender': [],
        'song': [],
        'location_id': [],
        'artist_id': [],
        'gender_id': [],
        'song_id': [],
        'local_song_id': [],
        'no_of_artists':[],
    }

    for j, song in enumerate(tqdm([song for song in os.listdir(os.path.join(datasets_path, dataset, language, artist)) if song.endswith('.wav')])):
        song, sr = librosa.load(os.path.join(datasets_path, dataset, language, artist, song), sr=22050)
        mel_spec = process_song(song, sr, frame, slide)
        if mel_spec:
            for mel in mel_spec:
                data['mel_spec'].append(mel)
                data['language'].append(str(language))
                data['location'].append(str(metadata[metadata.language == language][metadata.artist == artist]['location'].values[0]))
                data['artist'].append(str(artist))
                data['gender'].append(str(metadata[metadata.language == language][metadata.artist == artist]['gender'].values[0]))
                data['location_id'].append(metadata[metadata.language == language][metadata.artist == artist]['location_id'].values[0])
                data['artist_id'].append(metadata[metadata.language == language][metadata.artist == artist]['artist_id'].values[0])
                data['gender_id'].append(metadata[metadata.language == language][metadata.artist == artist]['gender_id'].values[0])
                data['song_id'].append(metadata[metadata.language == language][metadata.artist == artist][metadata.local_song_index == (j + 1)]['s_no'].values[0])
                data['song'].append(metadata[metadata.language == language][metadata.artist == artist][metadata.local_song_index == (j + 1)]['song_name'].values[0])
                data['local_song_id'].append(j + 1)
                data['no_of_artists'].append(metadata[metadata.language == language][metadata.artist == artist][metadata.local_song_index == (j + 1)]['no_of_artists'].values[0])


    data['mel_spec'] = np.array(data['mel_spec'])
    data['language'] = np.array(data['language']).astype('S')
    data['location'] = np.array(data['location']).astype('S')
    data['artist'] = np.array(data['artist']).astype('S')
    data['gender'] = np.array(data['gender']).astype('S')
    data['location_id'] = np.array(data['location_id'])
    data['artist_id'] = np.array(data['artist_id'])
    data['gender_id'] = np.array(data['gender_id'])
    data['song_id'] = np.array(data['song_id'])
    data['local_song_id'] = np.array(data['local_song_id'])
    data['no_of_artists'] = np.array(data['no_of_artists'])

    return data


def process_genres(dataset, frame, slide):
    dataset_path = os.path.join(datasets_path, dataset)
    languages = os.listdir(dataset_path)
    languages = [folder for folder in languages if os.path.isdir(os.path.join(datasets_path, dataset, folder))]

    for language in tqdm(languages):
        for artist in tqdm(os.listdir(os.path.join(dataset_path, language))):
            data = process_genre(artist, language, frame, slide)
            dst_path = os.path.join(features_path, language)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path, exist_ok=True)

            with open(os.path.join(features_path, language, artist + '.pickle'), 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    process_genres(dataset, 3, 0.5)
