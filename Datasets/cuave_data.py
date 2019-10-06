from keras.utils import np_utils
import numpy as np
import os
import re
import scipy.io
import scipy.misc as misc
import scipy.io.wavfile as wav

# Constants for the CUAVE dataset
min_frames = 6
max_frames = 44
total = 1790

people = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12',
          's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24',
          's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32', 's33', 's34', 's35', 's36']
digits = { 'zero' : 0, 'one' : 1, 'two' : 2, 'three' : 3, 'four' : 4,
           'five' : 5, 'six' : 6, 'seven' : 7, 'eight' : 8, 'nine' : 9 }
img_path = BASE_DIR + 'CUAVE/Video frames/'
mfcc_path = BASE_DIR + 'CUAVE/MFCCs/'


def write_mfccs(dir_path):
    min_mfccs = 100
    max_mfccs = 0

    # Look for WAV files
    for filename in os.listdir(dir_path):
        if filename.endswith(".wav"):
            mfcc_filename = filename[:-4] + '.mfcc'

            # Read the audio signal
            (sr, y) = wav.read(dir_path + '/' + filename)
            # Convert stereo to mono
            y = (y[:,0] + y[:,1]) / 2.0

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y,
                                         sr,
                                         n_mfcc=26,
                                         n_fft=2869,
                                         hop_length=1721)
            mfccs = mfccs.T
            # Need 26 MFCCs per video frame
            assert mfccs.shape[1] == 26
            mfccs.tofile(mfcc_filename)

            if min_mfccs > mfccs.shape[0]:
                min_mfccs = mfccs.shape[0]
            if max_mfccs < mfccs.shape[0]:
                max_mfccs = mfccs.shape[0]


def write_frames(dir_path):
    min_frames = 100
    max_frames = 0
    min_file = ""
    max_file = ""

    # Look for the JPEG files corresponding to the M4V video for each example
    for filename in os.listdir(dir_path):
        if filename.endswith(".mpg"):
            frames = np.empty((100, 90, 120), dtype=float)
            i = 1
            while True:
                img_filename = dir_path + '/' + filename + '_frame'
                img_filename += str(i) + '.bmp'

                try:
                    img_array = misc.imread(img_filename)
                    img_array = np.mean(img_array, axis=2)

                    # Append current frame to current example
                    frames[i] = img_array
                    i += 1
                except IOError:
                    # No more frames for current example
                    frames = frames[:i]
                    if min_frames > i:
                        min_frames = i
                        min_file = filename
                    if max_frames < i:
                        max_frames = i
                        max_file = filename
                    break

            frames.tofile(filename[:-4] + '.npy')


def list_files(path):
    # returns a list of names (with extension, without full path) of all files
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files


def get_cuave_mfccs():
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
        if i.endswith('Store'):
            continue

        person = ""
        for p in people:
            if p in i:
                person = p
                break
        P[count] = person

        # Label of class == letter
        for key in digits:
            if key in i:
                class_label = digits[key]
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


def get_cuave_mfccs_LSTM(with_num_frames=False):

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
        for key in digits:
            if key in i:
                class_label = digits[key]
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
