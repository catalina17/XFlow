import argparse
import sys

import keras
import numpy as np

# CNN x MLP baseline and models
import av_cnn_x_mlp
import cnn_mlp_baseline
import digits_cnn_x_mlp

# CNN x MLP -- LSTM baseline and models
import av_lstm_cnn_x_mlp
import digits_lstm_cnn_x_mlp
import lstm_cnn_mlp_baseline

# Datasets
import avletters_data
import data_config
import digits_data
import cuave_data

parser = argparse.ArgumentParser()

# Model and data
parser.add_argument('--dataset', type=str, choices=['avletters', 'cuave', 'digits'])
parser.add_argument('--model', type=str, choices=['cnn_mlp_baseline', 'cnn_x_mlp',
						  'cnn_mlp_lstm_baseline', 'cnn_mlp_lstm'])

# Optimization hyperparameters
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=300)

args = parser.parse_args()

class ValAccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_accs = []

    def on_epoch_end(self, batch, logs={}):
        self.val_accs.append(logs.get('val_acc'))


def evaluate_single_cv_fold(model, X_image, X_audio, Y, train, test):
    model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
    history = ValAccHistory()

    model.fit([X_image[train], X_audio[train]], Y[train],
                      batch_size=args.batch_size,
                      nb_epoch=args.num_epochs,
                      validation_data=([X_image[test], X_audio[test]], Y[test]),
                      callbacks=[history],
                      verbose=1)

    max_acc = max(history.val_accs)
    print('Result for fold:', max_acc)
    return max_acc


if __name__=='__main__':
    # Prepare model and dataset
    lstm_model = 'lstm' in args.model.lower()
    baseline = 'baseline' in args.model.lower()

    if args.dataset.lower() == 'avletters':
        people = ['Anya', 'Bill', 'Faye', 'John', 'Kate', 'Nicola', 'Stephen', 'Steve', 'Verity',
                  'Yi']
        if lstm_model:
            (X_image, X_audio, P, Y) = avletters_data.get_avletters_LSTM()
            model = lstm_cnn_mlp_baseline.get_model() if baseline else av_lstm_cnn_x_mlp.get_model()
        else:
            (X_image, X_audio, P, Y) = avletters_data.get_avletters()
            model = cnn_mlp_baseline.get_model() if baseline else av_cnn_x_mlp.get_model()

    elif args.dataset.lower() == 'cuave':
        people = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
                  's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22',
                  's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32', 's33',
                  's34', 's35', 's36']
        if lstm_model:
            (X_image, X_audio, P, Y) = cuave_data.get_cuave_mfccs_LSTM()
			# Digits and CUAVE have the same input dimensions
            model = lstm_cnn_mlp_baseline.get_model() if baseline else digits_lstm_cnn_x_mlp.get_model()
        else:
            (X_image, X_audio, P, Y) = cuave_data.get_cuave_mfccs()
            model = cnn_mlp_baseline.get_model() if baseline else digits_cnn_x_mlp.get_model()

    elif args.dataset.lower() == 'digits':
        people = ['Andreea', 'Andrej', 'Catalina', 'Costin', 'Dani', 'Edgar', 'Hugo', 'Ioana',
                  'Laura', 'Lucia', 'Milos', 'Petar', 'Razvan', 'Sisi', 'Tudor']
        if lstm_model:
            (X_image, X_audio, P, Y) = digits_data.get_digits_mfccs_LSTM()
            model = lstm_cnn_mlp_baseline.get_model() if baseline else digits_lstm_cnn_x_mlp.get_model()
        else:
            (X_image, X_audio, P, Y) = digits_data.get_digits_mfccs()
            model = cnn_mlp_baseline.get_model() if baseline else digits_cnn_x_mlp.get_model()

    # Cross-validation
    cvscores = []
    if args.dataset.lower() == 'cuave':
        for fold in range(9):
            test_people = people[4*fold : 4*(fold + 1)]
            train = [i for i in range(len(P)) if not (P[i].decode('ascii') in test_people)]
            test = [i for i in range(len(P)) if P[i].decode('ascii') in test_people]

            print(test_people, '| Train:', len(train), 'Test:', len(test))
            max_acc = evaluate_single_cv_fold(model, X_image, X_audio, Y, train, test)
            cvscores.append(max_acc)
    else:
        for p in people:
            train = [i for i in range(0, len(P)) if not p in P[i].decode('ascii')]
            test = [i for i in range(0, len(P)) if p in P[i].decode('ascii')]

            print(p, '| Train:', len(train), 'Test:', len(test))
            max_acc = evaluate_single_cv_fold(model, X_image, X_audio, Y, train, test)
            cvscores.append(max_acc)
