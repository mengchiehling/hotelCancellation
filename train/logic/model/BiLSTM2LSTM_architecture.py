'''
BiLSTM2LSTM parameters:

1. encoder:
    Must: encoder_lstm_units

2. decoder:
    Optional:
        decoder_dense_units
'''
from typing import Dict

from tensorflow.keras.models import Model

from train.logic.model.encoder import BiLSTM_encoder
from train.logic.model.decoder import LSTM_decoder


def build_model(n_inputs, n_features, n_outputs: int, decoder_cat_dict: Dict,
                dropout: float=0, recurrent_dropout: float=0, **kwargs):

    encoder_lstm_units = kwargs.get('encoder_lstm_units')
    decoder_dense_units = kwargs.get('decoder_dense_units')

    encoder_inputs_layers, _, state_h, state_c = BiLSTM_encoder(n_inputs, n_features, lstm_units=encoder_lstm_units,
                                                                recurrent_dropout=recurrent_dropout,
                                                                dropout=dropout)

    outputs = LSTM_decoder(state_h, dense_units=decoder_dense_units, lstm_units=encoder_lstm_units, n_outputs=n_outputs,
                           dropout=dropout, recurrent_dropout=recurrent_dropout,
                           state_c=state_c)

    model = Model(inputs=encoder_inputs_layers, outputs=outputs)

    return model

