from keras.utils import np_utils
import numpy as np
import os
import re

from data_config import *

# Constants for the Digits dataset
min_frames = 6
max_frames = 58
min_spec_len = 19
total = 750

people = ['Andreea', 'Andrej', 'Catalina', 'Costin', 'Dani', 'Edgar', 'Hugo', 'Ioana', 'Laura',
          'Lucia', 'Milos', 'Petar', 'Razvan', 'Sisi', 'Tudor']
img_path = BASE_DIR + 'Digits/Video frames/'
mfcc_path = BASE_DIR + 'Digits/MFCCs/'
spec_path = BASE_DIR + 'Digits/Spectrograms/'


def list_files(path):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files


def get_digits_mfccs():

    imgs = list(sorted(list_files(img_path)))
    audios = list(sorted(list_files(mfcc_path)))

    # Containers for the data
    X_image = np.empty(shape=(total, 90, 120, min_frames), dtype=np.float32)
    X_mfcc = np.empty(shape=(total, 26 * min_frames), dtype=np.float32)
    Y = np.empty(shape=(total,), dtype=np.int)
    P = np.empty(shape=(total,), dtype=np.dtype('a16'))

    count = 0
    # Process each pair of video and audio files
    for i,a in zip(imgs,audios):
        if i.endswith('Store') or a.endswith('Store'):
            continue

        person = ""
        for p in people:
            if p in i:
                person = p
                break
        P[count] = person

        # Label of class == letter
        class_label = ord(i[0]) - ord('0')
        Y[count] = class_label

        # Load MFCCs
        mfccs = np.fromfile(mfcc_path + a)
        mfccs = np.reshape(mfccs,
                           newshape=(mfccs.shape[0] // 26, 26))

        # Load video frames
        frames = np.fromfile(img_path + i)
        frames = np.reshape(frames,
                            newshape=(frames.shape[0] // 90 // 120, 90, 120))

        i_window = frames.shape[0] - min_frames + 1
        a_window = mfccs.shape[0] - min_frames + 1
        for j in range(0, min_frames):
            X_image[count, :, :, j] = frames[j:j + i_window, :, :].mean(0)
            X_mfcc[count, 26*j:26*(j+1)] = mfccs[j:j + a_window, :].mean(0)

        count += 1

    assert len(X_image) == len(X_mfcc) and len(X_mfcc) == len(Y) and \
           len(Y) == len(P)

    Y = np_utils.to_categorical(Y, 10)
    return (X_image, X_mfcc, P, Y)


def get_digits_mfccs_LSTM(with_num_frames=False):
    # returns the data from the digits dataset in the form of a tuple
    # (X_image, X_mfcc, P, Y) with variable number of frames/MFCC sets, padding the rest of the
    # array up to the maximum number of frames

    imgs = list(sorted(list_files(img_path)))
    audios = list(sorted(list_files(mfcc_path)))

    # Containers for the data
    X_image = np.zeros(shape=(total, max_frames, 90, 120, 1), dtype=np.float32)
    X_mfcc = np.zeros(shape=(total, max_frames, 26, ), dtype=np.float32)
    Y = np.empty(shape=(total,), dtype=np.uint32)
    P = np.empty(shape=(total,), dtype=np.dtype('a16'))
    F = np.empty(shape=(total,), dtype=np.int)

    count = 0
    # Process each pair of video and audio files
    for i,a in zip(imgs,audios):
        if i.endswith('Store'):
            continue

        person = ""
        for p in people:
            if p in i:
                person = p
                break
        P[count] = person

        # Label of class == letter
        class_label = ord(i[0]) - ord('0')
        Y[count] = class_label

        # Load MFCCs
        mfccs = np.fromfile(mfcc_path + a)
        mfccs = np.reshape(mfccs,
                           newshape=(mfccs.shape[0] // 26, 26))
        # Load video frames
        frames = np.fromfile(img_path + i)
        frames = np.reshape(frames,
                            newshape=(frames.shape[0] // 90 // 120, 90, 120))

        num_frames = min(mfccs.shape[0], frames.shape[0])
        mfcc_window = mfccs.shape[0] - num_frames + 1
        image_window = frames.shape[0] - num_frames + 1

        for i in range(0, num_frames):
            X_image[count, i, :, :, 0] = frames[i:i+image_window, :, :].mean(0)
            X_mfcc[count, i, :] = mfccs[i:i+mfcc_window, :].mean(0)

        F[count] = num_frames

        count += 1

    assert len(X_image) == len(X_mfcc) and len(X_mfcc) == len(Y) and \
           len(Y) == len(P) and len(P) == len(F)

    Y = np_utils.to_categorical(Y, 10)
    if with_num_frames:
        return (X_image, X_mfcc, P, Y, F)

    return (X_image, X_mfcc, P, Y)


def get_digits_spectrograms():

    imgs = list(sorted(list_files(img_path)))
    audios = list(sorted(list_files(mfcc_path)))

    # Containers for the data
    X_image = np.empty(shape=(total, 90, 120, min_frames), dtype=np.float32)
    X_spec = np.empty(shape=(total, 128, min_spec_len), dtype=np.float32)
    Y = np.empty(shape=(total,), dtype=np.int)
    P = np.empty(shape=(total,), dtype=np.dtype('a16'))

    count = 0
    # Process each pair of video and spectrogram files
    for i,s in zip(imgs,specgrams):
        if i.endswith('Store'):
            continue

        person = ""
        for p in people:
            if p in i:
                person = p
                break
        P[count] = person

        # Label of class == letter
        class_label = ord(i[0]) - ord('0')
        Y[count] = class_label

        # Load MFCCs
        specgram = np.fromfile(spec_path + s)
        specgram = np.reshape(specgram,
                              newshape=(128, specgram.shape[0] // 128))
        # Load video frames
        frames = np.fromfile(img_path + i)
        frames = np.reshape(frames,
                            newshape=(frames.shape[0] // 90 // 120, 90, 120))

        i_window = frames.shape[0] - min_frames + 1
        s_window = specgram.shape[1] - min_spec_len + 1
        for j in range(0, min_frames):
            X_image[count, :, :, j] = frames[j:j + i_window, :, :].mean(0)
        for j in range(0, min_spec_len):
            X_spec[count, :, j] = specgram[:, j:j + s_window].mean(1)

        count += 1

    assert len(X_image) == len(X_spec) and len(X_spec) == len(Y) and \
           len(Y) == len(P)

    Y = np_utils.to_categorical(Y, 10)
    return (X_image, X_spec, P, Y)
