import os

import tensorflow as tf
import tensorflow_io as tfio

AUTOTUNE = tf.data.AUTOTUNE


def get_dataset_info(base_path, dataset_nick_name):
    path, dataset, outputs = None, None, None
    if dataset_nick_name == 'carnatic':
        dataset = 'Indian Carnatic Classical Music'
        path = base_path + 'data/raw/' + dataset
        outputs = [40, 21, 12, 33, 29, 1]

    elif dataset_nick_name == 'hindustani':
        dataset = 'Indian Hindustani Classical Music'
        path = base_path + 'data/raw/' + dataset
        outputs = [30, 23, 7, 10, 7, 6, 30, 30]

    elif dataset_nick_name == 'regional':
        dataset = 'Indian Regional Music'
        path = base_path + 'data/raw/' + dataset
        outputs = [17, 68, 1, 1, 1]

    elif dataset_nick_name == 'folk':
        dataset = 'Indian Folk Music'
        path = base_path + 'data/raw/' + dataset
        outputs = [15, 12, 126, 1, 1]

    elif dataset_nick_name == 'semi-classical':
        dataset = 'Indian Semi Classical Music'
        path = base_path + 'data/raw/' + dataset
        outputs = [9, 49, 1, 1]
    return path, outputs


def get_waveform(file_path):
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = tf.audio.decode_wav(contents=audio_binary,
                                       desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    frames = tf.signal.frame(waveform, sr * 3, (sr * 3) // 2, pad_end=True)
    return frames


def get_carnatic_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)

    label1 = int(tf.strings.split(input=parts[7], sep='_')[1])  # Raag_id
    label2 = int(tf.strings.split(input=parts[8], sep='_')[1])  # Swaras_set_id
    label3 = int(tf.strings.split(input=parts[8], sep='_')[2])  # Melakarta_id
    label4 = int(tf.strings.split(input=parts[8], sep='_')[3])  # aaroh_set_id
    label5 = int(tf.strings.split(input=parts[8], sep='_')[3])  # avroh_set_id
    label6 = int(tf.strings.split(input=parts[8], sep='_')[3])  # janak_id

    return label1  # , label2, label3, label4, label5


def get_hindustani_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)

    label1 = int(tf.strings.split(input=parts[-2], sep='_')[1])  # Raag_id
    label2 = int(tf.strings.split(input=parts[-2], sep='_')[2])  # Swaras_set_id
    label3 = int(tf.strings.split(input=parts[-2], sep='_')[3])  # jati_id
    label4 = int(tf.strings.split(input=parts[-2], sep='_')[4])  # thaat_id
    label5 = int(tf.strings.split(input=parts[-2], sep='_')[5])  # vadi_id
    label6 = int(tf.strings.split(input=parts[-2], sep='_')[6])  # samvadi_id
    label7 = int(tf.strings.split(input=parts[-2], sep='_')[7])  # aaroh_set_id
    label8 = int(tf.strings.split(input=parts[-2], sep='_')[8])  # avroh_set_id

    return label1, label2, label3, label4, label5, label6, label7, label8


def get_folk_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    label1 = int(tf.strings.split(input=parts[7], sep='_')[1])  # Genre_id
    label2 = int(tf.strings.split(input=parts[8], sep='_')[1])  # State_id
    label3 = int(tf.strings.split(input=parts[8], sep='_')[2])  # Artist_id
    label4 = int(tf.strings.split(input=parts[8], sep='_')[3])  # Gender_id
    label5 = int(tf.strings.split(input=tf.strings.split(input=parts[9],
                                                         sep='_')[1], sep='.')[0])  # No_of_artists
    return label1  # , label2, label3, label4, label5


def get_regional_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    label1 = int(tf.strings.split(input=parts[7], sep='_')[1])  # Location_id
    label2 = int(tf.strings.split(input=parts[8], sep='_')[1])  # Artist_id
    label3 = int(tf.strings.split(input=parts[8], sep='_')[2])  # Gender_id
    label4 = int(tf.strings.split(input=parts[8], sep='_')[3])  # Veteran
    label5 = int(tf.strings.split(input=tf.strings.split(input=parts[9],
                                                         sep='_')[1], sep='.')[0])  # No_of_artists
    return label1  # , label2, label3, label4, label5


def get_semi_classical_label(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)

    label1 = int(tf.strings.split(input=parts[7], sep='_')[1])  # Genre_id
    label2 = int(tf.strings.split(input=parts[8], sep='_')[1])  # Artist_id
    label3 = int(tf.strings.split(input=parts[8], sep='_')[2])  # Gender_id
    label4 = int(tf.strings.split(input=tf.strings.split(input=parts[9],
                                                         sep='_')[1], sep='.')[0])  # No_of_artists
    return label1  # , label2, label3, label4


def get_waveform_and_label(file_path):
    label = get_hindustani_label(file_path)
    waveform = get_waveform(file_path)
    # labels = tf.repeat(label, repeats=tf.shape(waveform)[0])
    labels = tf.tile(tf.expand_dims(label, axis=0), [tf.shape(waveform)[0], 1])
    return waveform, labels


def get_spectrogram(waveform):
    return tfio.audio.spectrogram(
        waveform, nfft=2048 * 4, window=2048 * 4, stride=512)


def get_mel_spec(spec):
    mel_spectrogram = tfio.audio.melscale(
        spec, rate=44100, mels=256, fmin=0, fmax=8000)

    # Convert to db scale mel-spectrogram
    return tfio.audio.dbscale(
        mel_spectrogram, top_db=80)


def get_labels(y):
    return y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]


def make_dataset_ds(filenames, batch_size):
    files_ds = tf.data.Dataset.from_tensor_slices(filenames)

    ds = (files_ds
          .map(map_func=get_waveform_and_label,
               num_parallel_calls=AUTOTUNE)
          .flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
          .map(map_func=lambda x, y: (get_spectrogram(x), y), num_parallel_calls=AUTOTUNE)
          .map(map_func=lambda x, y: (get_mel_spec(x), y), num_parallel_calls=AUTOTUNE)
          .map(map_func=lambda x, y: (x, get_labels(y)), num_parallel_calls=AUTOTUNE)
          .batch(batch_size)
          .prefetch(AUTOTUNE)
          )
    return ds
