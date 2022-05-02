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
dataset = 'carnatic'
features_path = '../../data/processed/mel_spec'


metadata = pd.read_excel('../../data/raw/carnatic/carnatic_raags_metadata.xlsx')


def extract_mel(x, sr=22050):
    return librosa.power_to_db((librosa.feature.melspectrogram(y=x, sr=sr)), ref=np.max)


def process_song(song, sr, frame, slide):
    slices = [song[int(i * sr): int((i + frame) * sr)] for i in np.arange(0, int(len(song) // sr), slide)[:-5]]

    with ThreadPoolExecutor() as ex:
        result = ex.map(extract_mel, slices, [sr]*len(slices))
    return result


def process_genre(raag, frame, slide):
    data = {
        'mel_spec': [],
        'raag': [],
        'swaras': [],
        'melakarta': [],
        'aaroh': [],
        'avroh': [],
        'janak/janya': [],
        'raag_id': [],
        'swaras_set_id': [],
        'melakarta_id': [],
        'aaroh_set_id': [],
        'avroh_set_id': [],
        'janak': [],
    }

    for j, song in enumerate(tqdm(os.listdir(os.path.join(datasets_path, dataset, raag)))):
        x, sr = librosa.load(os.path.join(datasets_path, dataset, raag, song ), sr=22050)
        mel_spec = process_song(x, sr, frame, slide)
        if mel_spec:
            for mel in mel_spec:
                data['mel_spec'].append(mel)
                data['raag'].append(str(raag))
                data['swaras'].append(str(metadata[metadata.raag == raag]['swaras'].values[0]))
                data['melakarta'].append(str(metadata[metadata.raag == raag]['melakarta'].values[0]))
                data['aaroh'].append(str(metadata[metadata.raag == raag]['aaroh'].values[0]))
                data['avroh'].append(str(metadata[metadata.raag == raag]['avroh'].values[0]))
                data['janak/janya'].append(str(metadata[metadata.raag == raag]['janak/janya'].values[0]))
                data['raag_id'].append(str(metadata[metadata.raag == raag]['raag_id'].values[0]))
                data['swaras_set_id'].append(str(metadata[metadata.raag == raag]['swaras_set_id'].values[0]))
                data['melakarta_id'].append(str(metadata[metadata.raag == raag]['melakarta_id'].values[0]))
                data['aaroh_set_id'].append(str(metadata[metadata.raag == raag]['aaroh_set_id'].values[0]))
                data['avroh_set_id'].append(str(metadata[metadata.raag == raag]['avroh_set_id'].values[0]))
                data['janak'].append(str(metadata[metadata.raag == raag]['janak'].values[0]))

    data['mel_spec'] = np.array(data['mel_spec'])
    data['raag'] = np.array(data['raag']).astype('S')
    data['swaras'] = np.array(data['swaras']).astype('S')
    data['melakarta'] = np.array(data['melakarta']).astype('S')
    data['aaroh'] = np.array(data['aaroh']).astype('S')
    data['avroh'] = np.array(data['avroh']).astype('S')
    data['janak/janya'] = np.array(data['janak/janya']).astype('S')
    data['raag_id'] = np.array(data['raag_id'])
    data['swaras_set_id'] = np.array(data['swaras_set_id'])
    data['melakarta_id'] = np.array(data['melakarta_id'])
    data['aaroh_set_id'] = np.array(data['aaroh_set_id'])
    data['avroh_set_id'] = np.array(data['avroh_set_id'])
    data['janak'] = np.array(data['janak'])

    return data


def process_genres(frame, slide):
    dataset_path = os.path.join(datasets_path, dataset)
    raags = os.listdir(dataset_path)
    raags = [folder for folder in raags if os.path.isdir(os.path.join(datasets_path, dataset, folder))]
    for raag in tqdm(raags):
        data = process_genre(raag, frame, slide)
        dst_path = os.path.join(features_path, raag)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path, exist_ok=True)

        with open(os.path.join(features_path, dataset, raag + '.pickle'), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    process_genres(3, 0.5)
