import numpy as np


def train_val_split(dataset, train_size_percentage, val_size_percentage):
    """This function splits the dataset into the training, validation, and testing subset splits in the provided
    percentage portions.

    Args:
        dataset(dict): Dataset dictionary object.
        train_size_percentage(int): Percentage of the training subset split.
        val_size_percentage(int): Percentage of the validation subset split.

    Returns:
        tuple1(tuple): Training split data with corresponding to multitask labels.
        tuple2(tuple): Validation split data with corresponding to multitask labels.
        tuple3(tuple): Testing split data with corresponding to multitask labels.
    """

    # Shuffle dataset
    indices = np.random.permutation(len(dataset['language']))
    dataset['mel_spec'] = np.array(dataset['mel_spec'])[indices]
    dataset['language'] = np.array(dataset['language'])[indices]
    dataset['artist'] = np.array(dataset['artist'])[indices]
    dataset['gender'] = np.array(dataset['gender'])[indices]
    dataset['song'] = np.array(dataset['song'])[indices]
    dataset['location_id'] = np.array(dataset['location_id'])[indices]
    dataset['artist_id'] = np.array(dataset['artist_id'])[indices]
    dataset['gender_id'] = np.array(dataset['gender_id'])[indices]
    dataset['song_id'] = np.array(dataset['song_id'])[indices]
    dataset['local_song_id'] = np.array(dataset['local_song_id'])[indices]

    # Split dataset according to the training, validation, and test split percentages.
    dataset_size = len(dataset['song_id'])
    train_size = (dataset_size * train_size_percentage) // 100
    val_size = (dataset_size * val_size_percentage) // 100

    x_train, y1_train, y2_train, y3_train, y4_train, y5_train = dataset['mel_spec'][:train_size],\
                                                                dataset['location_id'][:train_size],\
                                                                dataset['artist_id'][:train_size],\
                                                                dataset['gender_id'][:train_size],\
                                                                dataset['song_id'][:train_size],\
                                                                dataset['local_song_id'][:train_size]

    x_val, y1_val, y2_val, y3_val, y4_val, y5_val = dataset['mel_spec'][train_size:train_size+val_size],\
                                                    dataset['location_id'][train_size:train_size+val_size],\
                                                    dataset['artist_id'][train_size:train_size+val_size],\
                                                    dataset['gender_id'][train_size:train_size+val_size],\
                                                    dataset['song_id'][train_size:train_size+val_size],\
                                                    dataset['local_song_id'][train_size:train_size+val_size]

    x_test, y1_test, y2_test, y3_test, y4_test, y5_test = dataset['mel_spec'][train_size+val_size:],\
                                                          dataset['location_id'][train_size+val_size:],\
                                                          dataset['artist_id'][train_size+val_size:],\
                                                          dataset['gender_id'][train_size+val_size:],\
                                                          dataset['song_id'][train_size+val_size:],\
                                                          dataset['local_song_id'][train_size+val_size:]

    tuple1 = (x_train, y1_train, y2_train, y3_train, y4_train, y5_train)
    tuple2 = (x_val, y1_val, y2_val, y3_val, y4_val, y5_val)
    tuple3 = (x_test, y1_test, y2_test, y3_test, y4_test, y5_test)

    return tuple1, tuple2, tuple3
