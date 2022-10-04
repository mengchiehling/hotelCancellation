from typing import Optional

from tensorflow.keras.layers import Input, LSTM, Concatenate, ReLU, \
    Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, Add, Bidirectional, LayerNormalization
from tensorflow.keras.regularizers import L2


def residue_block(x, filters: int, idx: int, l2: float, momentum: float):

    # pre-activation residue block
    # https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
    # https://www.researchgate.net/figure/Architecture-of-normal-residual-block-a-and-pre-activation-residual-block-b_fig2_337691625

    regularizer = L2(l2=l2)

    if idx > 0:
        x = BatchNormalization(momentum=momentum)(x)
        x = ReLU()(x)
    x = Conv1D(filters=filters, kernel_size=3, kernel_regularizer=regularizer, name=f'conv1d_block{idx}_1')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = ReLU()(x)
    x = Conv1D(filters=filters, kernel_size=3, kernel_regularizer=regularizer, name=f'conv1d_block{idx}_2')(x)

    return x

def residue_layer(x, filters: int, idx: int, l2: float, momentum: float):

    x_b = residue_block(x, filters=filters, idx=idx, l2=l2, momentum=momentum)

    x = Add()([x, x_b])

    return x


def CNN_encoder(n_inputs, n_features, filters: int, dropout: float=0, l2: float=0, momentum: float=0.99):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0
    x = residue_block(x, filters=filters, idx=idx, l2=l2, momentum=momentum)
    x = Dropout(rate=dropout)(x)

    idx += 1
    x = residue_block(x, filters=filters, idx=idx, l2=l2, momentum=momentum)
    x = Dropout(rate=dropout)(x)

    x = MaxPooling1D(pool_size=2)(x)
    embedding = Flatten()(x)

    return inputs_layers, embedding


def CNNRes_encoder(n_inputs, n_features, filters: int, dropout: float = 0):
    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0
    x = residue_layer(x, filters=filters, idx=idx)
    x = Dropout(rate=dropout)(x)

    idx += 1
    x = residue_layer(x, filters=filters, idx=idx)
    x = Dropout(rate=dropout)(x)

    x = MaxPooling1D(pool_size=2)(x)
    embedding = Flatten()(x)

    return inputs_layers, embedding


def LSTM_block(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int, l2: float):

    regularizer = L2(l2=l2)

    if idx == 0:
        x = LayerNormalization()(x)
        x = ReLU()(x)
    x, state_h, state_c = LSTM(lstm_units, activation='linear', return_state=True, return_sequences=True,
                               dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_regularizer=regularizer,
                               name=f'encoder_LSTM_{idx}_1')(x)
    x = LayerNormalization()(x)
    x = ReLU()(x)
    x, state_h, state_c = LSTM(lstm_units, activation='linear', return_state=True, return_sequences=True,
                               dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_regularizer=regularizer,
                               name=f'encoder_LSTM_{idx}_2')(x)
    return x, state_h, state_c


def LSTMRes_layer(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int, l2: float):

    x_b = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx, l2)

    x = Add()([x, x_b])

    return x


def LSTM_encoder(n_inputs, n_features, lstm_units: int, dropout: float=0, recurrent_dropout: float=0,
                 l2: float=0):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    # categorical_inputs = []

    # x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    x = encoder_numerical_inputs

    idx = 0
    x, state_h, state_c = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx, l2)

    idx += 1
    x, state_h, state_c = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx, l2)

    return inputs_layers, x, state_h, state_c


def LSTMRes_encoder(n_inputs, n_features, lstm_units: int, dropout: float=0, recurrent_dropout: float=0,
                    l2: float=0):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0
    x, state_h, state_c = LSTMRes_layer(x, lstm_units, dropout, recurrent_dropout, idx, l2)

    idx += 1
    x, state_h, state_c = LSTMRes_layer(x, lstm_units, dropout, recurrent_dropout, idx, l2)

    return inputs_layers, x, state_h, state_c


def BiLSTM_block(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int):

    lstm_layer = LSTM(lstm_units, activation='relu', return_state=True, return_sequences=True,
                      dropout=dropout,
                      recurrent_dropout=recurrent_dropout, name=f'encoder_LSTM_{idx}')
    x, forward_h, forward_c, backward_h, backward_c = Bidirectional(lstm_layer, name=f'encoder_BiLSTM_{idx}')(x)

    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])

    return x, state_h, state_c


def BiLSTM_encoder(n_inputs, n_features, lstm_units: int, dropout: float=0, recurrent_dropout: float=0):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0
    x, state_h, state_c = BiLSTM_block(x, lstm_units, dropout, recurrent_dropout, idx)

    return inputs_layers, x, state_h, state_c
