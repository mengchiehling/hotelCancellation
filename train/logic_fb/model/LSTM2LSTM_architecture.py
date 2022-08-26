'''
CNN2BiLSTM parameters:

1. encoder:

    Must: encoder_filters_0
    Optional:
        encoder_lstm_units_1

2. decoder:

    Must: decoder_lstm_units_1

    Optional:
        decoder_dense_units
'''

from typing import Dict

from tensorflow.keras.models import Model

from train.logic_fb.model.encoder import LSTM_encoder
from train.logic_fb.model.decoder import LSTM_decoder


def build_model(n_inputs, n_features, n_outputs: int,
                dropout: float=0, recurrent_dropout: float=0, **kwargs):

    encoder_lstm_units = kwargs.get('encoder_lstm_units')
    decoder_lstm_units = kwargs.get('decoder_lstm_units')
    decoder_dense_units = kwargs.get('decoder_dense_units')

    encoder_inputs_layers, _, state_h, state_c = LSTM_encoder(n_inputs, n_features, lstm_units=encoder_lstm_units,
                                                              recurrent_dropout=recurrent_dropout,
                                                              dropout=dropout)

    outputs = LSTM_decoder(state_h, dense_units=decoder_dense_units,
                           lstm_units=decoder_lstm_units, n_outputs=n_outputs,
                           dropout=dropout, recurrent_dropout=recurrent_dropout)

    model = Model(inputs=encoder_inputs_layers, outputs=outputs)

    return model

