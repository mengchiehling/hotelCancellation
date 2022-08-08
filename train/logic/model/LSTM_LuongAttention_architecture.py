'''
Adding channel attention mechanism to state_h input of decoder
'''

from typing import List, Optional, Dict

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, LSTM, Concatenate, TimeDistributed, GlobalAveragePooling1D, Embedding, \
    Dense, RepeatVector, Layer, Attention
from tensorflow.keras.models import Model

from src.logic.common.functions import parenthesis_striped


def build_encoder(n_inputs, n_features, encoder_lstm_units: List[int]):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    for idx, lstm_units in enumerate(encoder_lstm_units):
        x, state_h, state_c = LSTM(lstm_units, activation='relu', return_state=True, return_sequences=True, dropout=0.1,
                                   recurrent_dropout=0.1, name=f'encoder_LSTM_{idx}')(x)

    return inputs_layers, x, state_h, state_c


def build_decoder(key_tensor, state_h, state_c, decoder_dense_units, decoder_lstm_units: List[int], decoder_cat_dict):

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

    decoder_inputs = Concatenate(axis=2)(categorical_inputs)

    query_tensor = LSTM(state_h.shape[1], activation='relu', return_sequences=True, dropout=0.1,
             recurrent_dropout=0.1, name='decoder_LSTM_0')(decoder_inputs, initial_state=[state_h, state_c])

    y = Attention()([query_tensor, key_tensor])  # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention

    if decoder_lstm_units:
        for idx, lstm_units in enumerate(decoder_lstm_units):
            y = LSTM(lstm_units, activation='relu', return_sequences=True, dropout=0.1,
                     recurrent_dropout=0.1, name=f'decoder_LSTM_{idx+1}')(y)

    for units in decoder_dense_units:
        y = TimeDistributed(Dense(units, activation='relu'))(y)

    outputs = TimeDistributed(Dense(1), name='outputs')(y)

    return inputs_layers, outputs


def build_model(n_inputs, n_features, decoder_cat_dict: Dict, encoder_lstm_units: List[int],
                decoder_dense_units: List[int], decoder_lstm_units: Optional[List[int]]=None):

    encoder_inputs_layers, key_tensor, state_h, state_c = build_encoder(n_inputs, n_features, encoder_lstm_units)

    decoder_inputs_layers, outputs = build_decoder(key_tensor, state_h, state_c, decoder_dense_units=decoder_dense_units,
                                                   decoder_lstm_units=decoder_lstm_units,
                                                   decoder_cat_dict=decoder_cat_dict)

    model = Model(inputs=encoder_inputs_layers + decoder_inputs_layers, outputs=outputs)

    return model

