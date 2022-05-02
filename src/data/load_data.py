import pickle
import os


def get_regional_data(path):
    """
    Load the regional music dataset from the provided dataset path.

    Args:
        path(str): Dataset path of the regional dataset.

    Returns:
        dataset(dict): Dictionary containing the load regional dataset.
    """

    dataset = {
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
    }

    for language in os.listdir(path):
        for artist in os.listdir(os.path.join(path, language)):
            with open(os.path.join(path, language, artist), 'rb') as fp:
                data = pickle.load(fp)
                dataset['mel_spec'] = [*dataset['mel_spec'], *data['mel_spec']]
                dataset['language'] = [*dataset['language'], *data['language']]
                dataset['location'] = [*dataset['location'], *data['location']]
                dataset['artist'] = [*dataset['artist'], *data['artist']]
                dataset['gender'] = [*dataset['gender'], *data['gender']]
                dataset['song'] = [*dataset['song'], *data['song']]
                dataset['location_id'] = [*dataset['location_id'], *data['location_id']]
                dataset['artist_id'] = [*dataset['artist_id'], *data['artist_id']]
                dataset['gender_id'] = [*dataset['gender_id'], *data['gender_id']]
                dataset['song_id'] = [*dataset['song_id'], *data['song_id']]
                dataset['local_song_id'] = [*dataset['local_song_id'], *data['local_song_id']]
    
    return dataset
