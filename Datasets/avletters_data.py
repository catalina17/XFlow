from keras.utils import np_utils
import numpy as np
import os
import re
import scipy.io

from data_config import *

img_path = BASE_DIR + 'avletters/Lips/'
audio_path = BASE_DIR + 'avletters/Audio/textmfcc/'


def list_files(path):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files


def get_avletters():
    # returns the data from the AVletters dataset in the form of a tuple
    # (X_image, X_audio, Y)

    imgs = list(sorted(list_files(img_path)))
    audios = list(sorted(list_files(audio_path)))

    min_n_frames = 11
    total = 780
    # Containers for the data
    X_image = np.empty(shape=(total, 80, 60, min_n_frames), dtype=np.float32)
    X_audio = np.empty(shape=(total, 26 * min_n_frames), dtype=np.float32)
    Y = np.empty(shape=(total,), dtype=np.int)
    P = np.empty(shape=(total,), dtype=np.dtype('a16'))

    count = 0
    min_lines = 100000
    people = ['Anya', 'Bill', 'Faye', 'John', 'Kate', 'Nicola', 'Stephen',
              'Steve', 'Verity', 'Yi']

    # Process each pair of video and audio files
    for i,a in sorted(zip(imgs,audios)):
        if i.endswith('Store') or a.endswith('Store'):
            continue

        person = ""
        for p in people:
            if p in i:
                person = p
                break

        # Label of class == letter
        class_label = ord(i[0]) - ord('A')

        # Load video data
        mat = scipy.io.loadmat(img_path + i)
        n_frames = int(mat['siz'][0,2])
        frames = mat['vid'].reshape(80, 60, n_frames)
        frames = frames[:, :, :-1]

        # Load audio data/MFCCs
        f = open(audio_path + a, "rb")
        lines = f.readlines()
        f.close()
        lines = lines[1:-1]
        min_lines = min(min_lines, len(lines))
        n_frames = len(lines) // 6
        line_idx = 1

        # Average each window of size (n_frames - 12 + 1), such that final
        # example has depth 12
        window_size = n_frames - min_n_frames + 1
        #assert window_size >= 1

        # Get MFCC sets
        mfccs = np.zeros(shape=(n_frames, 26), dtype=np.float32)
        for j in range(0, n_frames):
            coefs = []

            # Parse 3 lines corresponding to one MFCC set
            for l in range(0, 3):
                try:
                    line = lines[line_idx]
                    line_idx += 1
                    line = line[8:]
                    coefs += re.findall("[-]?\d+.\d+", line.decode(errors='ignore'))
                except IndexError:
                    print(line_idx, len(lines), n_frames)
            assert len(coefs) == 26, 'len is %d (frame %d out of %d)' % (len(coefs), j, n_frames)

            mfccs[j, :] = coefs[:]
            # Skip every 2nd frame because there are 2x MFCC coefs
            # than video frames.
            line_idx += 3

        # Average over window_size to get depth 12 for image data and
        # 26*12 MFCCs
        for j in range(0, min_n_frames):
            X_image[count, :, :, j] = frames[:, :, j:j + window_size].mean(2)
            X_audio[count, 26*j:26*(j+1)] = mfccs[j:j + window_size, :].mean(0)

        if "1" in i: person += "1"
        if "2" in i: person += "2"
        if "3" in i: person += "3"

        Y[count] = class_label
        P[count] = person
        count   += 1

    # Eliminate the extra space in arrays
    X_image = X_image[:count, :, :, :]
    X_audio = X_audio[:count, :]
    Y       = Y[:count]
    P       = P[:count]

    assert len(X_image) == len(X_audio) and len(X_audio) == len(Y) and \
           len(Y) == len(P)

    Y = np_utils.to_categorical(Y, 26)
    return (X_image, X_audio, P, Y)


def get_avletters_LSTM():
    # returns the data from the AVletters dataset in the form of a tuple
    # (X_image, X_audio, Y, P)

    imgs = list(sorted(list_files(img_path)))
    audios = list(sorted(list_files(audio_path)))

    min_n_frames = 11
    max_n_frames = 39
    total = 780
    # Containers for the data
    X_image = np.zeros(shape=(total, max_n_frames, 80, 60, 1), dtype=np.float32)
    X_audio = np.zeros(shape=(total, max_n_frames, 26, ), dtype=np.float32)
    Y = np.empty(shape=(total,), dtype=np.int)
    P = np.empty(shape=(total,), dtype=np.dtype('a16'))

    count = 0
    min_lines = 100000
    people = ['Anya', 'Bill', 'Faye', 'John', 'Kate', 'Nicola', 'Stephen',
              'Steve', 'Verity', 'Yi']

    # Process each pair of video and audio files
    for i,a in zip(imgs,audios):
        if i.endswith('Store'):
            continue

        person = ""
        for p in people:
            if p in i:
                person = p
                break

        # Label of class == letter
        class_label = ord(i[0]) - ord('A')

        # Load video data
        mat = scipy.io.loadmat(img_path + i)
        n_frames = int(mat['siz'][0,2])
        frames = mat['vid'].reshape(80, 60, n_frames)
        frames = frames[:, :, :-1]

        # Load audio data/MFCCs
        f = open(audio_path + a, "r")
        lines = f.readlines()
        f.close()
        lines = lines[1:-1]
        min_lines = min(min_lines, len(lines))
        n_frames = len(lines) // 6
        line_idx = 1

        # Get MFCC sets
        mfccs = np.zeros(shape=(n_frames, 26), dtype=np.float32)
        for j in range(0, n_frames):
            coefs = []

            # Parse 3 lines corresponding to one MFCC set
            for l in range(0, 3):
                try:
                    line = lines[line_idx]
                    line_idx += 1
                    line = line[8:]
                    coefs += re.findall("[-]?\d+.\d+", line)
                except IndexError:
                    print(line_idx, len(lines), n_frames)
            assert len(coefs) == 26

            mfccs[j, :] = coefs[:]
            # Skip every 2nd frame because there are 2x MFCC coefs than video frames.
            line_idx += 3

        # Write current example to dataset
        for j in range(0, n_frames):
            X_image[count, j, :, :, 0] = frames[:, :, j]
            X_audio[count, j] = mfccs[j]

        if "1" in i: person += "1"
        if "2" in i: person += "2"
        if "3" in i: person += "3"

        Y[count] = class_label
        P[count] = person
        count   += 1

    assert len(X_image) == len(X_audio) and len(X_audio) == len(Y) and \
           len(Y) == len(P)

    Y = np_utils.to_categorical(Y, 26)
    return (X_image, X_audio, P, Y)
