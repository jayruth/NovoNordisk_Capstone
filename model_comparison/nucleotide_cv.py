import numpy as np
import pandas as pd

#the following 2 lines are required if ndac is not installed
import sys
sys.path.append("..") 
import ndac

from ndac.data_processing import quantile_classify, encode_sequence
from ndac.predict import cross_validate


#####################################################################
################## AMINO ACID SEQUENCE ##############################
#####################################################################

BATCH_SIZE = 100
EPOCHS = 40
K_FOLDS = 3

# read and encode data, splitting top and bottom quartiles into classes
data = pd.read_csv('../dataframes/DF_prest.csv', index_col=0)
df, hist = quantile_classify(data['conc_cf'], data['nt_seq'], 
                             [0.25, 0.75], drop_class=[1])
X, y = encode_sequence(df['nt_seq'], df['class'], max_length=150)

######################
####### CLSTM ########
######################

embedding_length = [5, 10, 20, 30]
cnn_filters = [32, 64, 128] 
filter_length =[3, 5, 7]
pool_size = [2, 4]
nodes = [10, 50, 100]
lstm_drop = [0.2]
dropout = [0.5]

# w/o embedding
param_grid = dict(cnn_filters=cnn_filters, filter_length=filter_length,
                  pool_size=pool_size, nodes=nodes, lstm_drop=lstm_drop,
                  dropout=dropout, embedding_length=embedding_length)
cross_validate(X, y, architecture='clstm', save_file='clstm_no_embedding.csv', 
               skip_embedding=True, batch_size=BATCH_SIZE, epochs=EPOCHS, 
               verbose=10, k=K_FOLDS, params=param_grid) 
# w/ embedding
param_grid = dict(cnn_filters=cnn_filters, filter_length=filter_length,
                  pool_size=pool_size, nodes=nodes, lstm_drop=lstm_drop,
                  dropout=dropout, embedding_length=embedding_length)
cross_validate(X, y, architecture='clstm', save_file='clstm.csv',
               skip_embedding=False, batch_size=BATCH_SIZE, epochs=EPOCHS,
               verbose=10, k=K_FOLDS, params=param_grid)

######################
####### LSTM #########
######################

embedding_length = [5, 10, 20, 30]
nodes = [10, 50, 100]
lstm_drop = [0.2]
dropout = [0.5]

# w/o embedding
param_grid = dict(nodes=nodes, lstm_drop=lstm_drop, dropout=dropout,
                  embedding_length=embedding_length)
cross_validate(X, y, architecture='lstm', save_file='lstm_no_embedding.csv',
               skip_embedding=True, batch_size=BATCH_SIZE, epochs=EPOCHS, 
               verbose=10, k=K_FOLDS, params=param_grid)
# w/ embedding
param_grid = dict(nodes=nodes, lstm_drop=lstm_drop, dropout=dropout,
                  embedding_length=embedding_length)
cross_validate(X, y, architecture='lstm', save_file='lstm.csv',
               skip_embedding=False, batch_size=BATCH_SIZE, epochs=EPOCHS, 
               verbose=10, k=K_FOLDS, params=param_grid)


######################
####### CNN ##########
######################

embedding_length = [5, 10, 20, 30]
cnn_filters = [32, 64, 128]
filter_length =[3, 5, 7]
pool_size = [2, 4]
nodes = [10, 50, 100]
dropout = [0.5]

# w/o embedding
param_grid = dict(cnn_filters=cnn_filters, filter_length=filter_length,
                  pool_size=pool_size, nodes=nodes, dropout=dropout,
                  embedding_length=embedding_length)
cross_validate(X, y, architecture='cnn', save_file='cnn_no_embedding.csv',
               skip_embedding=True, batch_size=BATCH_SIZE, epochs=EPOCHS, 
               verbose=10, k=K_FOLDS, params=param_grid)
# w/ embedding
param_grid = dict(cnn_filters=cnn_filters, filter_length=filter_length,
                  pool_size=pool_size, nodes=nodes, dropout=dropout,
                  embedding_length=embedding_length)
cross_validate(X, y, architecture='cnn', save_file='cnn.csv',
               skip_embedding=False, batch_size=BATCH_SIZE, epochs=EPOCHS,
               verbose=10, k=K_FOLDS, params=param_grid)

