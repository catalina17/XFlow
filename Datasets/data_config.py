BASE_DIR = '~/'

avletters_config = {
    'min_n_frames': 11,
    'max_n_frames': 39,

    'inputCNNshape': (80, 60, 11),
    'inputMLPshape': (26 * 11,),

    'lstm_inputCNNshape': (39, 80, 60, 1),
    'lstm_inputMLPshape': (39, 26, ),

    'nb_classes': 26,
}

digits_config = {
    'min_n_frames': 6,
    'max_n_frames': 58,

    'inputCNNshape': (90, 120, 6),
    'inputMLPshape': (26 * 6,),

    'lstm_inputCNNshape': (58, 90, 120, 1),
    'lstm_inputMLPshape': (58, 26, ),

    'nb_classes': 10,
}

cuave_config = {
    'min_n_frames': 6,
    'max_n_frames': 44,

    'inputCNNshape': (90, 120, 6),
    'inputMLPshape': (26 * 6,),

    'lstm_inputCNNshape': (44, 90, 120, 1),
    'lstm_inputMLPshape': (44, 26, ),

    'nb_classes': 26,
}

data_constants = {
    'avletters': avletters_config,
    'cuave': cuave_config,
    'digits': digits_config,
}
