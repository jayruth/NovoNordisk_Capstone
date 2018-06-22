import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.model_selection import train_test_split


def train_clstm(x, y, test_fraction=0, embedding_length=4,
                batch_size=100, epochs=5, verbose=1,
                save_file=None):
    # fix random seed for reproducibility
    np.random.seed(7)
    # get embedding parameters from x matrix
    vocab_size = x.max() + 1
    seq_len = x.shape[1]

    if test_fraction:
        # create test-train split
        x, x_test, y, y_test = train_test_split(x, y,
                                                test_size=test_fraction)

    # create the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_length,
                        input_length=seq_len))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='selu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
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
    model.fit(x, y, epochs=epochs,
              batch_size=batch_size, verbose=verbose)
    # report the test accuracy if we performed a train_test_split
    if test_fraction:
        # Final evaluation of the model
        scores = model.evaluate(x_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    if save_file:
        model.save(save_file)

    return model
