import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def _clstm(vocab_size, embedding_length, seq_len,
           cnn_filters, filter_length, x, pool_size,
           nodes, dropout, lstm_drop):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_length,
                        input_length=seq_len))
    model.add(Conv1D(filters=cnn_filters, kernel_size=filter_length,
                     padding='same', activation='selu'))
    # if user requests no embedding, replace w/ CNN only
    if not embedding_length:
        model.pop()
        model.pop()
        model.add(Conv1D(filters=cnn_filters, kernel_size=filter_length,
                         input_shape=(x.shape[1], x.shape[2]),
                         padding='same', activation='selu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=lstm_drop))
    return model


def _lstm(vocab_size, embedding_length, seq_len,
          nodes, dropout, lstm_drop, x):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_length,
                        input_length=seq_len))
    model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=lstm_drop))
    # if user requests no embedding, replace w/ LSTM only
    if not embedding_length:
        model.pop()
        model.pop()
        model.add(LSTM(nodes, dropout=dropout,
                       recurrent_dropout=lstm_drop,
                       input_shape=(x.shape[1], x.shape[2])))
    return model


def _cnn(vocab_size, embedding_length, seq_len,
         cnn_filters, filter_length, x,
         pool_size, nodes, dropout):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_length,
                        input_length=seq_len))
    model.add(Conv1D(filters=cnn_filters, kernel_size=filter_length,
                     padding='same', activation='selu'))
    # if user requests no embedding, replace w/ CNN only
    if not embedding_length:
        model.pop()
        model.pop()
        model.add(Conv1D(filters=cnn_filters, kernel_size=filter_length,
                         input_shape=(x.shape[1], x.shape[2]),
                         padding='same', activation='selu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(nodes))
    model.add(Dropout(dropout))
    return model


def train_model(x, y, architecture='clstm', test_fraction=0,
                embedding_length=10, batch_size=100, epochs=5,
                verbose=1, save_file=None, cnn_filters=128,
                filter_length=3, pool_size=2, nodes=100,
                lstm_drop=0.2, dropout=0.5):
    # fix random seed for reproducibility
    np.random.seed(7)
    # get embedding parameters from x matrix
    vocab_size = x.max() + 1
    seq_len = x.shape[1]

    if test_fraction:
        # create test-train split
        x, x_test, y, y_test = train_test_split(x, y,
                                                test_size=test_fraction)
    # convert None embedding value to False to prevent error
    if embedding_length is None:
        embedding_length = False

    if not embedding_length:
        x = to_categorical(x)
        if test_fraction:
            x_test = to_categorical(x_test, num_classes=x.shape[-1])

    # create the model
    # select architecture
    if architecture == 'clstm':
        model = _clstm(vocab_size, embedding_length, seq_len,
                       cnn_filters, filter_length, x, pool_size,
                       nodes, dropout, lstm_drop)
    elif architecture == 'lstm':
        model = _lstm(vocab_size, embedding_length, seq_len,
                      nodes, dropout, lstm_drop, x)
    elif architecture == 'cnn':
        model = _cnn(vocab_size, embedding_length, seq_len,
                     cnn_filters, filter_length, x,
                     pool_size, nodes, dropout)

    if np.isscalar(y[0]):
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    print(model.summary())

    fit_args = {
        'epochs': epochs,
        'batch_size': batch_size,
        'verbose': verbose
    }

    if test_fraction:
        fit_args['validation_data'] = (x_test, y_test)

    model.fit(x, y, **fit_args)

    # report the test accuracy if we performed a train_test_split
    if test_fraction:
        # Final evaluation of the model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    if save_file:
        model.save(save_file)

    return model
