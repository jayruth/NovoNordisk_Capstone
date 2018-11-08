import numpy as np
import pandas as pd

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

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


def _clstm(categorical=False, vocab_size=False, seq_len=200,
           embedding_length=10, cnn_filters=128, filter_length=3,
           pool_size=2, nodes=100, lstm_drop=0.2, dropout=0.5):
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
                         input_shape=(seq_len, vocab_size),
                         padding='same', activation='selu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(nodes, dropout=dropout, recurrent_dropout=lstm_drop))
    if not categorical:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(Dense(categorical, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    return model


def _lstm(categorical=False, vocab_size=False, seq_len=200,
          embedding_length=10, nodes=100, lstm_drop=0.2,
          dropout=0.5):
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
                       input_shape=(seq_len, vocab_size)))
    if not categorical:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(Dense(categorical, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


def _cnn(categorical=False, vocab_size=False, seq_len=150,
         embedding_length=10, cnn_filters=128, filter_length=3,
         pool_size=2, nodes=100, dropout=0.5):
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
                         input_shape=(seq_len, vocab_size),
                         padding='same', activation='selu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dense(nodes))
    model.add(Dropout(dropout))
    if not categorical:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    else:
        model.add(Dense(categorical, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model


def train_model(x, y, architecture='clstm', test_fraction=0,
                skip_embedding=False, batch_size=100, epochs=5,
                verbose=1, save_file=None, **kwargs):
    # fix random seed for reproducibility
    np.random.seed(7)

    if test_fraction:
        # create test-train split
        x, x_test, y, y_test = train_test_split(
            x, y, test_size=test_fraction
        )

    kwargs['vocab_size'] = int(x.max() + 1)
    # convert None embedding value to False to prevent error
    if skip_embedding:
        kwargs['embedding_length'] = False
        x = to_categorical(x)
        if test_fraction:
            x_test = to_categorical(x_test, num_classes=x.shape[-1])

    # get embedding parameters from x matrix
    kwargs['seq_len'] = int(x.shape[1])

    if not np.isscalar(y[0]):
        kwargs['categorical'] = y.shape[1]

    # select model architecture
    if architecture == 'clstm':
        model = _clstm(**kwargs)
    elif architecture == 'lstm':
        model = _lstm(**kwargs)
    elif architecture == 'cnn':
        model = _cnn(**kwargs)

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


def cross_validate(x, y, architecture='clstm', save_file=None,
                   skip_embedding=False, batch_size=100, epochs=35,
                   verbose=10, k=3, params=None):
    # fix random seed for reproducibility
    np.random.seed(7)
    params['vocab_size'] = [int(x.max() + 1)]

    if skip_embedding:
        params['embedding_length'] = [False]
        x = to_categorical(x)

    # get embedding parameters from x matrix
    params['seq_len'] = [int(x.shape[1])]

    if not np.isscalar(y[0]):
        params['categorical'] = [y.shape[1]]

    # print(model.summary())
    if architecture == 'clstm':
        model = KerasClassifier(build_fn=_clstm, batch_size=batch_size,
                                epochs=epochs, verbose=verbose)
    if architecture == 'lstm':
        model = KerasClassifier(build_fn=_lstm, batch_size=batch_size,
                                epochs=epochs, verbose=verbose)
    if architecture == 'cnn':
        model = KerasClassifier(build_fn=_cnn, batch_size=batch_size,
                                epochs=epochs, verbose=verbose)

    grid = GridSearchCV(estimator=model, param_grid=params, cv=k)  # , n_jobs=28)
    grid_result = grid.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    if save_file:
        grid_df = pd.DataFrame(grid_result.cv_results_['params'])
        grid_df['means'] = grid_result.cv_results_['mean_test_score']
        grid_df['stddev'] = grid_result.cv_results_['std_test_score']
        # print results to csv file
        grid_df.to_csv(save_file)

    return
