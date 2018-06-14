import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.layers.embeddings import Embedding

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# read original data from 
data = pd.read_csv('../../dataframes/DF_prest.csv', index_col=0)

# setup 'docs' for use with Tokenizer
def nt_seq_doc(nt_sequence):
    if 'GACAAGCTTGCGGCCGCA' not in nt_sequence:
        return None
    true_nt = nt_sequence.split('GACAAGCTTGCGGCCGCA')[1]
    if len(true_nt) % 3 != 0:
        return None
    return ' '.join([true_nt[i:i+3]
                     for i in range(0, len(true_nt), 3)])
# split quantiles
def assign_class(conc):
    if conc <= low_cut:
        return 0
    elif conc >= high_cut:
        return 1
    return

data['nt_seq_doc'] = data['nt_seq'].apply(nt_seq_doc)
data = data[pd.notnull(data['nt_seq_doc'])]

# identify high and low classes by conc_cf quantiles
low_cut = data['conc_cf'].quantile(0.25)
high_cut = data['conc_cf'].quantile(0.75)

data['class'] = data['conc_cf'].apply(assign_class)
data = data[pd.notnull(data['class'])]
# check shape
print('data shape: ', data.shape)

# define sequence documents
docs = list(data['nt_seq_doc'])
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)

# integer encode documents
X = t.texts_to_sequences(docs)
y = data['class'].values

# create test-train split
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.3, random_state=42)

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = len(t.word_index) + 1

# truncate and pad input sequences
seq_lengths = [len(seq) for seq in X]
max_seq_length = max(seq_lengths)
#X_train = sequence.pad_sequences(X_train, maxlen=max_seq_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_seq_length)
X = sequence.pad_sequences(X, maxlen=max_seq_length)

# tune hyperparameters for simple model

# define simple model per Yoon Kim (2014)
def create_model(embedding_length=16, num_filters=128, pool_size=2,
                 lstm_nodes=100, drop=0.5, recurrent_drop=0.5, filter_length=3):
    # create the model
    model = Sequential()
    model.add(Embedding(top_words, embedding_length, 
                        input_length=max_seq_length))
    model.add(Conv1D(filters=num_filters, kernel_size=filter_length, 
                     padding='same', activation='selu'))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_nodes, dropout=drop, 
              recurrent_dropout=recurrent_drop))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', 
                  metrics=['accuracy'])

    return model


model = KerasClassifier(build_fn=create_model, batch_size=64,
                        epochs=30, verbose=10)
# define the grid search parameters
# model hyperparameters
embedding_length = [4, 8, 16] #maybe use [3, 6, 9] for AA sequence
num_filters = [100, 200]
filter_length = [2, 3, 4, 5] #maybe longer? [3, 5, 8]
pool_size = [3, 4]
lstm_nodes = [100, 200]
# drop = [0.5, 0.8] just do 0.5
#recurrent_drop = [0.2, 0.5, 0.8]

param_grid = dict(num_filters=num_filters, pool_size=pool_size,
                  lstm_nodes=lstm_nodes, filter_length=filter_length)

grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    cv=3)#, n_jobs=28)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

grid_df = pd.DataFrame(grid_result.cv_results_['params'])
grid_df['means'] = grid_result.cv_results_['mean_test_score']
grid_df['stddev'] = grid_result.cv_results_['std_test_score']

# print results to csv file
grid_df.to_csv('test_output_jay.csv')
