'''
CNN2BiLSTM parameters:

1. encoder:

    Must: encoder_filters_0
    Optional:
        encoder_filters_1
        encoder_filters_2

2. decoder:

    Must: decoder_lstm_units_0
    Optional:
        decoder_lstm_units_1
        decoder_dense_units
'''


from typing import Optional, Dict

from tensorflow.keras.layers import Input, LSTM, Concatenate, TimeDistributed, GlobalAveragePooling1D, Embedding, \
    Dense, RepeatVector, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dropout, ReLU
from tensorflow.keras.models import Model

from src.logic.common.functions import parenthesis_striped


def build_encoder(n_inputs, n_features, filters_0: int, filters_1: Optional[int] = None,
                  filters_2: Optional[int] = None, dropout: float=0):
    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0
    x = Conv1D(filters=filters_0, kernel_size=3, name=f'encoder_CNN_{idx}')(x)
    x = Dropout(rate=dropout)(x)
    if filters_1:
        idx += 1
        filters_1 = int(filters_1)
        x = Conv1D(filters=filters_1, kernel_size=3, name=f'encoder_CNN_{idx}')(x)
        x = Dropout(rate=dropout)(x)
    if filters_2:
        idx += 1
        filters_2 = int(filters_2)
        x = Conv1D(filters=filters_2, kernel_size=3, name=f'encoder_CNN_{idx}')(x)
        x = Dropout(rate=dropout)(x)

    x = MaxPooling1D(pool_size=2)(x)
    embedding = Flatten()(x)

    return inputs_layers, embedding


def build_decoder(embedding, decoder_cat_dict, lstm_units_0: int, lstm_units_1: Optional[int]=None,
                  dense_units:Optional[int]=None, dropout: float=0, recurrent_dropout: float=0):

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

    categorical_inputs.append(RepeatVector(shape_1)(embedding))

    decoder_inputs = Concatenate(axis=2)(categorical_inputs)

    idx = 0
    lstm_0 = LSTM(lstm_units_0, activation='linear', return_sequences=True, dropout=dropout,
                  recurrent_dropout=recurrent_dropout, name=f'decoder_LSTM_{idx}')
    y = Bidirectional(lstm_0, name=f'decoder_BiLSTM_{idx}')(decoder_inputs)
    y = ReLU()(y)

    if lstm_units_1:
        idx += 1
        lstm_1 = LSTM(lstm_units_1, activation='linear', return_sequences=True, dropout=dropout,
                  recurrent_dropout=recurrent_dropout, name=f'decoder_LSTM_{idx}')
        y = Bidirectional(lstm_1, name=f'decoder_BiLSTM_{idx}')(decoder_inputs)
        y = ReLU()(y)

    if dense_units:
        y = TimeDistributed(Dense(dense_units, activation='relu'))(y)
        y = Dropout(rate=dropout)(y)
        if dropout > 0:
            y = TimeDistributed(Dropout(dropout))(y)

    outputs = TimeDistributed(Dense(1, activation='relu'), name='outputs')(y)

    return inputs_layers, outputs


def build_model(n_inputs, n_features, decoder_cat_dict: Dict, encoder_filters_0,
                decoder_lstm_units_0, dropout: float=0, recurrent_dropout: float=0, **kwargs):

    encoder_filters_0 = int(encoder_filters_0)
    decoder_lstm_units_0 = int(decoder_lstm_units_0)

    encoder_filters_1 = kwargs.get('encoder_filters_1')
    encoder_filters_2 = kwargs.get('encoder_filters_2')
    decoder_lstm_units_1 = kwargs.get('decoder_lstm_units_1')
    decoder_dense_units = kwargs.get('decoder_dense_units')

    encoder_inputs_layers, embedding = build_encoder(n_inputs, n_features, filters_0=encoder_filters_0,
                                                     filters_1=encoder_filters_1, filters_2=encoder_filters_2,
                                                     dropout=dropout)

    decoder_inputs_layers, outputs = build_decoder(embedding=embedding, dense_units=decoder_dense_units,
                                                   lstm_units_0=decoder_lstm_units_0,
                                                   lstm_units_1=decoder_lstm_units_1,
                                                   decoder_cat_dict=decoder_cat_dict,
                                                   dropout=dropout, recurrent_dropout=recurrent_dropout)

    model = Model(inputs=encoder_inputs_layers + decoder_inputs_layers, outputs=outputs)

    return model

