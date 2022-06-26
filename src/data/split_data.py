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
    indices = np.random.permutation(len(dataset['genre_id']))
    for key in dataset.keys():
        dataset[key] = dataset[key][indices]

    # Split dataset according to the training, validation, and test split percentages.
    dataset_size = len(indices)
    train_size = int((dataset_size * train_size_percentage) // 100)
    val_size = int((dataset_size * val_size_percentage) // 100)

    x_train, y1_train, y2_train, y3_train, y4_train, y5_train = np.abs(dataset['mel_spec'][:train_size]),\
                                                                dataset['genre_id'][:train_size],\
                                                                dataset['state_id'][:train_size],\
                                                                dataset['artist_id'][:train_size],\
                                                                dataset['gender_id'][:train_size],\
                                                                dataset['no_of_artists'][:train_size]

    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean)/std

    x_val, y1_val, y2_val, y3_val, y4_val, y5_val = np.abs(dataset['mel_spec'][train_size:train_size+val_size]),\
                                                    dataset['genre_id'][train_size:train_size+val_size],\
                                                    dataset['state_id'][train_size:train_size+val_size],\
                                                    dataset['artist_id'][train_size:train_size+val_size],\
                                                    dataset['gender_id'][train_size:train_size+val_size],\
                                                    dataset['no_of_artists'][train_size:train_size+val_size]

    x_val = (x_val - mean) / std

    # x_test, y1_test, y2_test, y3_test, y4_test = np.abs(dataset['mel_spec'][train_size+val_size:]),\
    #                                                       dataset['genre_id'][train_size+val_size:],\
    #                                                       dataset['artist_id'][train_size+val_size:],\
    #                                                       dataset['gender_id'][train_size+val_size:],\
    #                                                       dataset['no_of_artists'][train_size+val_size:]
    #
    # x_test = (x_test - mean) / std

    tuple1 = (x_train, y1_train, y2_train, y3_train, y4_train, y5_train)
    tuple2 = (x_val, y1_val, y2_val, y3_val, y4_val, y5_val)
    # tuple3 = (x_test, y1_test, y2_test, y3_test, y4_test)

    return tuple1, tuple2
