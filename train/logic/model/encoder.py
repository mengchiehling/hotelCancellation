from typing import Optional

from tensorflow.keras.layers import Input, LSTM, Concatenate, ReLU, \
    Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, Add, Bidirectional

from train.logic.data_preparation import generate_categorical_embeddings


def residue_block(x, filters: int, idx: int):

    # pre-activation residue block
    # https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
    # https://www.researchgate.net/figure/Architecture-of-normal-residual-block-a-and-pre-activation-residual-block-b_fig2_337691625

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters=filters, kernel_size=3, name=f'conv1d_block{idx}_1')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters=filters, kernel_size=3, name=f'conv1d_block{idx}_2')(x)

    return x

def residue_layer(x, filters: int, idx: int):

    x_b = residue_block(x, filters=filters, idx=idx)

    x = Add()([x, x_b])

    return x


def CNN_encoder(n_inputs, n_features, filters: int, dropout: float=0):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0
    x = residue_block(x, filters=filters, idx=idx)
    x = Dropout(rate=dropout)(x)

    idx += 1
    x = residue_block(x, filters=filters, idx=idx)
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


def LSTM_block(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int):

    x, state_h, state_c = LSTM(lstm_units, activation='relu', return_state=True, return_sequences=True,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout, name=f'encoder_LSTM_{idx}')(x)

    return x, state_h, state_c


def LSTMRes_layer(x, lstm_units: int, dropout: float, recurrent_dropout: float, idx: int):

    x_b = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx)

    x = Add()([x, x_b])

    return x


def LSTM_encoder(n_inputs, n_features, lstm_units: int, encoder_cat_dict, dropout: float=0, recurrent_dropout: float=0):

    inputs_layers, categorical_embedding = generate_categorical_embeddings(section='encoder',
                                                                        cat_dict=encoder_cat_dict)

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    x = Concatenate(axis=2)(categorical_embedding + [encoder_numerical_inputs])


    idx = 0
    x, state_h, state_c = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx)

    idx += 1
    x, state_h, state_c = LSTM_block(x, lstm_units, dropout, recurrent_dropout, idx)

    return inputs_layers, x, state_h, state_c


def LSTMRes_encoder(n_inputs, n_features, lstm_units: int, dropout: float=0, recurrent_dropout: float=0):

    inputs_layers = []

    encoder_numerical_inputs = Input(shape=(n_inputs, n_features), name='encoder_X_num')

    inputs_layers.append(encoder_numerical_inputs)

    categorical_inputs = []

    x = Concatenate(axis=2)(categorical_inputs + [encoder_numerical_inputs])

    idx = 0
    x, state_h, state_c = LSTMRes_layer(x, lstm_units, dropout, recurrent_dropout, idx)

    idx += 1
    x, state_h, state_c = LSTMRes_layer(x, lstm_units, dropout, recurrent_dropout, idx)

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
