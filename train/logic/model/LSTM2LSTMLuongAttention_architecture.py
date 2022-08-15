'''
Adding channel attention mechanism to state_h input of decoder

LSTM2LSTMLuongAttention parameters:

1. encoder:

    Must: encoder_filters_0
    Optional:
        encoder_lstm_units_1

2. decoder:

    Optional:
        decoder_lstm_units_1
        decoder_dense_units

'''

from typing import Optional, Dict

from tensorflow.keras.layers import Input, LSTM, Concatenate, TimeDistributed, GlobalAveragePooling1D, Embedding, \
    Dense, Attention, RepeatVector, Dropout
from tensorflow.keras.models import Model

from src.logic.common.functions import parenthesis_striped


def build_encoder(n_inputs, n_features, lstm_units_0: int, lstm_units_1: Optional[int]=None, dropout: float=0,
                  recurrent_dropout: float=0):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0

    x, state_h, state_c = LSTM(lstm_units_0, activation='relu', return_state=True, return_sequences=True, dropout=dropout,
                               recurrent_dropout=recurrent_dropout, name=f'encoder_LSTM_{idx}')(x)

    if lstm_units_1:
        idx += 1
        lstm_units_1 = int(lstm_units_1)
        x, state_h, state_c = LSTM(lstm_units_1, activation='relu', return_state=True, return_sequences=True,
                                   dropout=dropout, recurrent_dropout=recurrent_dropout,
                                   name=f'encoder_LSTM_{idx}')(x)

    return inputs_layers, x, state_h, state_c


def build_decoder(key_tensor, state_h, state_c, dense_units, decoder_cat_dict, lstm_units_1: Optional[int]=None,
                  dropout: float = 0, recurrent_dropout: float = 0):

    inputs_layers = []

    categorical_inputs = []

    for key, item in decoder_cat_dict.items():

        q = item['value']
        _, shape_1, shape_2 = q.shape

        cat_input = Input(shape=(shape_1, shape_2), name=f"{key}_decoder")

        inputs_layers.append(cat_input)

        input_dim = item['input_dim']
        y_embedding = Embedding(input_dim, 1)(cat_input)
        y_embedding = TimeDistributed(GlobalAveragePooling1D(), name=f"{parenthesis_striped(key)}_decoder_embed")(
            y_embedding)

        categorical_inputs.append(y_embedding)

    categorical_inputs.append(RepeatVector(shape_1)(state_h))

    decoder_inputs = Concatenate(axis=2)(categorical_inputs)

    idx = 0

    query_tensor = LSTM(state_h.shape[1], activation='relu', return_sequences=True, dropout=dropout,
                        recurrent_dropout=recurrent_dropout, name=f'decoder_LSTM_{idx}')(decoder_inputs,
                                                                                         initial_state=[state_h, state_c])

    y = Attention()([query_tensor, key_tensor])  # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention

    if lstm_units_1:
        idx += 1
        lstm_units_1 = int(lstm_units_1)
        y, state_h, state_c = LSTM(lstm_units_1, activation='relu', return_state=True, return_sequences=True,
                                   dropout=dropout, recurrent_dropout=recurrent_dropout, name=f'decoder_LSTM_{idx}')(y)

    if dense_units:
        dense_units = int(dense_units)
        y = TimeDistributed(Dense(units=dense_units, activation='relu', dropout=dropout,
                                  name='decoder_dense_layer'))(y)
        if dropout > 0:
            y = TimeDistributed(Dropout(dropout))(y)

    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    return inputs_layers, outputs


def build_model(n_inputs, n_features, decoder_cat_dict: Dict, encoder_lstm_units_0: int,
                dropout: float=0, recurrent_dropout: float=0, **kwargs):

    encoder_lstm_units_0 = int(encoder_lstm_units_0)

    encoder_lstm_units_1 = kwargs.get('encoder_lstm_units_1')
    decoder_lstm_units_1 = kwargs.get('decoder_lstm_units_1')
    decoder_dense_units = kwargs.get('decoder_dense_units')

    encoder_inputs_layers, sequence_outputs, state_h, state_c = build_encoder(n_inputs, n_features,
                                                                              lstm_units_0=encoder_lstm_units_0,
                                                                              lstm_units_1=encoder_lstm_units_1,
                                                                              dropout=dropout,
                                                                              recurrent_dropout=recurrent_dropout)

    decoder_inputs_layers, outputs = build_decoder(sequence_outputs, state_h, state_c, dense_units=decoder_dense_units,
                                                   lstm_units_1=decoder_lstm_units_1,
                                                   decoder_cat_dict=decoder_cat_dict,
                                                   dropout=dropout,
                                                   recurrent_dropout=recurrent_dropout)

    model = Model(inputs=encoder_inputs_layers + decoder_inputs_layers, outputs=outputs)

    return model


